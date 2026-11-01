"""A FastAPI server that exposes the chatbot from main.py over HTTP.

Run it with:  uvicorn server:app --reload

The chat logic itself lives in main.py, so the terminal chatbot and this API
always behave the same way. The .env values and the API key never leave the
server: they are only read inside main.get_config().

A turn may also end with a save proposal: when the model has researched
something, main.collect_saves() hands the server the Markdown draft it prepared.
POST /chat never writes that draft -- it parks it in the in-memory pending store
below and returns it as ``pending_save``, so the user can review it. Only
POST /save, after that review, hands the proposal to save_tool.write_proposal(),
and POST /save/cancel drops it again. Nothing is persisted: a restart forgets
every proposal that was still waiting.
"""

import re
import secrets
import threading
import time

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator

import main
import save_tool

app = FastAPI(title="Chatbot API")

# Roles a client may send. "system" is deliberately missing: the system prompt
# must always come from main.SYSTEM_PROMPT, never from the request.
ALLOWED_ROLES = ("user", "assistant")

# Request-size guards. A single call must not be able to spend an unreasonable
# amount of provider budget, so the message, the number of history turns and the
# length of each turn are all bounded. These values mirror the frontend limits in
# frontend/src/api/chat.ts and frontend/src/lib/history.ts.
#
# MAX_HISTORY_MESSAGES follows main.HISTORY_LIMIT (the number of messages the
# model actually keeps), so a client never uploads history the server would only
# trim away anyway.
MAX_MESSAGE_LENGTH = 4000
MAX_HISTORY_MESSAGES = main.HISTORY_LIMIT
MAX_HISTORY_CONTENT_LENGTH = 4000


# --------------------------------------------------------------------------
# Pending save proposals
#
# A save proposal is prepared inside one /chat turn (main.collect_saves) and is
# only text at that point: no file exists yet. /chat moves the proposals of the
# turn into the store below, where they wait for the user's decision, and
# POST /save writes exactly one of them after that decision -- with the lock of
# the store *released*, because a write may block for a while.
#
# Every entry lives in this process' memory only, so nothing is persisted and a
# restart forgets every proposal that was still waiting.
# --------------------------------------------------------------------------

# How long a waiting proposal stays usable, and how many of them may wait at
# once. An abandoned proposal (the user simply closed the tab) would otherwise
# live forever, and a client that never confirms would otherwise grow the store
# without bound, so both values are deliberately small.
PENDING_SAVE_TTL_SECONDS = 300.0  # 5 minutes
MAX_PENDING_SAVES = 16

# The two states a stored proposal can be in: waiting for its user, or handed to
# the writer of exactly one request. There is no entry for the third one, because
# leaving a state is what removes it ("pending -> saving -> removed", and
# "pending -> removed" for a cancellation).
PENDING_STATE = "pending"
SAVING_STATE = "saving"

# The id is generated here with secrets.token_urlsafe() and never comes from a
# client. 32 random bytes make guessing a valid id impractical; the length limit
# only bounds what a request may even look like.
SAVE_ID_BYTES = 32
MAX_SAVE_ID_LENGTH = 128

# The preview is the exact Markdown that would be written, so it is only attached
# to a response while the whole document still fits here.
MAX_PREVIEW_LENGTH = 20_000

# A raw HTML tag: "<" followed by a letter, "/", "!" or "?". The preview travels
# as text -- this server never renders or executes HTML -- but should the
# document contain markup anyway, only the title, the summary and the sources are
# returned, so a client is never asked to render markup. Markdown links and
# comparisons like "5 < 10" are unaffected.
RAW_HTML_PATTERN = re.compile(r"<[/!?a-zA-Z]")

# What a proposal must carry before it is stored is defined in exactly one place,
# save_tool.copy_proposal(); this server only calls it, so the stored contract and
# the writer's contract can never drift apart.

# Longest file name that may be echoed back. save_tool builds far shorter names,
# so this is only a bound on what one response can contain.
MAX_FILENAME_LENGTH = 255

# The only fields of a stored proposal that may leave the server. The rest (the
# byte count, the state, the internal proposal mapping) stays here.
PUBLIC_SAVE_FIELDS = ("save_id", "title", "summary", "filename", "sources")

# One generic answer for every unusable id, so a caller cannot tell an unknown id
# from an expired, cancelled or already-used one.
SAVE_NOT_FOUND_DETAIL = "No save proposal is waiting for this id."
SAVE_FAILED_DETAIL = "The research file could not be saved."


def public_text(value, limit):
    """Return a bounded, plain copy of one proposal text field."""
    if not isinstance(value, str):
        return ""
    # The stored text is already bounded by save_tool; this is only a guard.
    return value[:limit]


def public_sources(raw_sources):
    """Return the safe source list of one proposal.

    save_tool.select_sources() is reused rather than re-implemented: it keeps
    only real ``http``/``https`` URLs, drops duplicates, bounds the list and
    falls back to the URL when a title is missing, so a client can never be handed
    a fabricated or dangerous link.
    """
    try:
        return save_tool.select_sources(raw_sources)
    except save_tool.SaveError:
        return []


def pending_save_view(save_id, proposal):
    """Return the client-facing view of one stored proposal.

    Only what the confirmation card needs leaves the server: the id the client
    has to send back, the title, the summary, the file name (a plain name, never
    a path) and the validated sources. ``preview`` is added as well when it is
    safe and practical -- see MAX_PREVIEW_LENGTH and RAW_HTML_PATTERN. Nothing
    else of the internal proposal is exposed.
    """
    view = {
        "save_id": save_id,
        "title": public_text(proposal.get("title"), save_tool.MAX_TITLE_LENGTH),
        "summary": public_text(
            proposal.get("summary"), save_tool.MAX_SUMMARY_LENGTH
        ),
        "filename": public_text(proposal.get("filename"), MAX_FILENAME_LENGTH),
        "sources": public_sources(proposal.get("sources")),
    }

    content = proposal.get("content")
    if (
        isinstance(content, str)
        and len(content) <= MAX_PREVIEW_LENGTH
        and not RAW_HTML_PATTERN.search(content)
    ):
        view["preview"] = content
    return view


def storable_proposal(proposal):
    """Return a private copy of a proposal worth storing, or None.

    The rules of what a valid proposal is live in save_tool.copy_proposal(), so
    this server never restates them: only a full build_proposal() result -- the
    text fields, a plain Markdown file name and the validated source mappings --
    is accepted at all. The mapping that comes back is an independent copy,
    nested sources included, so a later change to the original can never alter
    what will eventually be written.
    """
    return save_tool.copy_proposal(proposal)


class PendingSaveStore:
    """Thread-safe, in-memory store of save proposals waiting for confirmation.

    A client can only ever name an id: the proposal behind it -- content, file
    name, sources -- is prepared by this server and is never read from a request.

    Every transition happens under one lock and is therefore atomic:

    - ``register``      new entry, state "pending"
    - ``begin_save``    "pending" -> "saving"; only the first caller wins
    - ``cancel``        "pending" -> removed
    - ``restore``       "saving" -> "pending", unless it expired meanwhile
    - ``discard``       removed, whatever the state (after a successful write)

    No file system call and no other blocking work happens while the lock is
    held: a writer takes its proposal out with begin_save(), writes it after the
    lock has been released, and reports back with discard() or restore(). That is
    also what keeps two writers apart -- the loser of begin_save() gets None and
    never reaches save_tool.write_proposal(). ``clock`` is injectable so the
    tests can expire an entry without sleeping.
    """

    def __init__(
        self,
        ttl=PENDING_SAVE_TTL_SECONDS,
        capacity=MAX_PENDING_SAVES,
        clock=None,
    ):
        self._lock = threading.Lock()
        self._entries = {}
        self._ttl = ttl
        self._capacity = capacity
        self._clock = clock or time.monotonic

    def register(self, proposal):
        """Store one proposal and return its client-facing view, or None.

        None means the proposal was refused: this store keeps only what looks
        like a build_proposal() result, and only while it is below its capacity.
        Expired entries are dropped first, so a full store means genuinely
        waiting proposals. The returned view carries the one and only handle for
        the stored proposal, its ``save_id``.
        """
        stored = storable_proposal(proposal)
        if stored is None:
            return None

        now = self._clock()
        with self._lock:
            self._purge(now)
            if len(self._entries) >= self._capacity:
                return None
            save_id = self._new_id()
            self._entries[save_id] = {
                "proposal": stored,
                "state": PENDING_STATE,
                "expires_at": now + self._ttl,
            }
        # Built outside the lock: the entry above owns the private copy already,
        # so nothing can change under us and the critical section stays short.
        return pending_save_view(save_id, stored)

    def view(self, save_id):
        """Return the client-facing view of a waiting proposal, or None."""
        with self._lock:
            entry = self._live_entry(save_id, self._clock())
            if entry is None or entry["state"] != PENDING_STATE:
                return None
            return pending_save_view(save_id, entry["proposal"])

    def begin_save(self, save_id):
        """Take a waiting proposal out of the store for exactly one writer.

        Returns the stored proposal (ready for save_tool.write_proposal()) or None
        when the id is unknown, expired, already taken by another request or
        already used. The caller owns the proposal now and has to finish with
        discard() after a successful write, or give it back with restore() when
        the write failed.
        """
        with self._lock:
            entry = self._live_entry(save_id, self._clock())
            if entry is None or entry["state"] != PENDING_STATE:
                return None
            entry["state"] = SAVING_STATE
            return dict(entry["proposal"])

    def cancel(self, save_id):
        """Remove a waiting proposal. False when it is not usable any more.

        A proposal that is being written cannot be cancelled, so a cancellation
        can never pull it out from under a writer that already started.
        """
        with self._lock:
            entry = self._live_entry(save_id, self._clock())
            if entry is None or entry["state"] != PENDING_STATE:
                return False
            del self._entries[save_id]
            return True

    def restore(self, save_id):
        """Make a taken proposal wait again after a failed write.

        False when the entry is gone, was never taken, or expired while the write
        was running: an expired proposal must never become usable again.
        """
        if not isinstance(save_id, str):
            return False
        with self._lock:
            entry = self._entries.get(save_id)
            if entry is None or entry["state"] != SAVING_STATE:
                return False
            if entry["expires_at"] <= self._clock():
                del self._entries[save_id]
                return False
            entry["state"] = PENDING_STATE
            return True

    def discard(self, save_id):
        """Forget one proposal, whatever state it is in. True when it was there."""
        if not isinstance(save_id, str):
            return False
        with self._lock:
            return self._entries.pop(save_id, None) is not None

    def count(self):
        """Number of proposals currently waiting (expired ones are dropped)."""
        with self._lock:
            self._purge(self._clock())
            return len(self._entries)

    # -- internals; every one of them runs with the lock held ---------------

    def _purge(self, now):
        """Drop every *waiting* entry whose lifetime is over.

        A proposal in "saving" state is deliberately kept: a writer already owns
        it and still has to finish with discard() or restore(). restore() is what
        retires such an entry after a failed write once the TTL has passed, so a
        proposal that is being written can neither be purged out from under its
        writer nor become usable again.
        """
        expired = [
            save_id
            for save_id, entry in self._entries.items()
            if entry["state"] == PENDING_STATE and entry["expires_at"] <= now
        ]
        for save_id in expired:
            del self._entries[save_id]

    def _live_entry(self, save_id, now):
        """Return an entry, dropping a waiting one when it has expired.

        A "saving" entry is returned even after its TTL: the proposal belongs to
        a running writer and only restore() or discard() may retire it. Every
        caller rejects a "saving" entry by its state, so handing it back is
        harmless and keeps an in-flight write from being cancelled or replayed.
        """
        if not isinstance(save_id, str):
            return None
        entry = self._entries.get(save_id)
        if entry is None:
            return None
        if entry["state"] == PENDING_STATE and entry["expires_at"] <= now:
            del self._entries[save_id]
            return None
        return entry

    def _new_id(self):
        """Return an unused id; a collision is practically impossible."""
        save_id = secrets.token_urlsafe(SAVE_ID_BYTES)
        while save_id in self._entries:
            save_id = secrets.token_urlsafe(SAVE_ID_BYTES)
        return save_id


# The one store this process uses. The tests replace it with their own instance
# so no state leaks from one test into the next.
PENDING_SAVES = PendingSaveStore()


class HistoryMessage(BaseModel):
    """One turn of the conversation sent by the client."""

    role: str
    content: str = Field(max_length=MAX_HISTORY_CONTENT_LENGTH)

    @field_validator("role")
    @classmethod
    def role_must_be_user_or_assistant(cls, value):
        """Reject anything that is not a normal conversation turn."""
        if value not in ALLOWED_ROLES:
            raise ValueError("role must be 'user' or 'assistant'")
        return value


class ChatRequest(BaseModel):
    """The body of a POST /chat request."""

    message: str = Field(max_length=MAX_MESSAGE_LENGTH)
    history: list[HistoryMessage] = Field(
        default_factory=list, max_length=MAX_HISTORY_MESSAGES
    )

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value):
        """Trim the message and make sure something is left."""
        value = value.strip()
        if not value:
            raise ValueError("message must not be empty")
        return value


class SaveRequest(BaseModel):
    """The body of POST /save and POST /save/cancel.

    It carries the id of a waiting proposal and nothing else, because a client
    may not supply content, a file name, a folder, a path or sources: all of that
    is read from this server's own pending store. An unknown field is refused
    (422) instead of being quietly ignored, so an attempt to pass a location is a
    visible error and never a silent no-op.
    """

    model_config = ConfigDict(extra="forbid")

    save_id: str = Field(max_length=MAX_SAVE_ID_LENGTH)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Return one short, generic message instead of pydantic's field details.

    The default 422 body lists internal field names, locations and limits, which
    the client does not need and should not see, so it is replaced here.
    """
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid request payload."},
    )


@app.get("/")
def read_root():
    """Return a simple status message to show the server is up."""
    return {"status": "ok", "message": "Chatbot API is running"}


@app.get("/health")
def read_health():
    """Health check endpoint."""
    return {"status": "healthy"}


def chat_with_saves(message, history):
    """Run one turn and return (reply, updated_history, proposals).

    main.chat() keeps its signature and its behaviour: the collector only means
    that save proposals prepared during this turn are handed back instead of
    being dropped. No file is written here -- the caller decides what to do with
    the proposals (read_chat parks them in the pending store).
    """
    with main.collect_saves() as proposals:
        reply, updated_history = main.chat(message, history)
    return reply, updated_history, proposals


@app.post("/chat")
def read_chat(request: ChatRequest):
    """Send one message to the model and return the reply and new history.

    Errors are translated to plain HTTP status codes. Only short, generic
    messages are returned, so the API key, the .env values and tracebacks are
    never exposed. The returned history never contains the system message,
    because the server adds it itself on every request.

    A turn may also end with save proposals (the model called the save_research
    tool; see main.collect_saves). No file is written here: only the last
    proposal of the turn is parked in the pending store and returned as
    ``pending_save``, so the user can review it and then confirm or cancel it;
    any earlier draft of the same turn is superseded and never stored. Without an
    accepted proposal the response is exactly the one from before --
    ``{"reply": ..., "history": ...}`` and nothing else.
    """
    history = [turn.model_dump() for turn in request.history]

    try:
        reply, updated_history, proposals = chat_with_saves(
            request.message, history
        )
    except main.ConfigError:
        raise HTTPException(
            status_code=503,
            detail="Server configuration is incomplete.",
        )
    except AuthenticationError:
        raise HTTPException(
            status_code=502,
            detail="The model provider rejected our credentials.",
        )
    except RateLimitError:
        raise HTTPException(
            status_code=429,
            detail="Rate limit reached. Please try again later.",
        )
    except APITimeoutError:
        # The provider *was* reached: it simply did not answer inside the time
        # this server allows (see TOOL_ANSWER_TIMEOUT_SECONDS and
        # TOOL_DEADLINE_SECONDS in main.py). The SDK derives APITimeoutError from
        # APIConnectionError, so this branch has to come first -- otherwise a slow
        # answer is reported as an unreachable provider (503) instead of a
        # timeout (504).
        raise HTTPException(
            status_code=504,
            detail="The model provider took too long to answer.",
        )
    except APIConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Could not reach the model provider.",
        )
    except APIStatusError:
        raise HTTPException(
            status_code=502,
            detail="The model provider returned an error.",
        )
    except main.ToolDeadlineExceeded:
        raise HTTPException(
            status_code=504,
            detail="The web search turn took too long to finish.",
        )
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Unexpected server error.",
        )

    if not reply:
        # The chat() helper keeps the history unchanged in this case.
        raise HTTPException(
            status_code=502,
            detail="The model returned an empty reply.",
        )

    response = {"reply": reply, "history": updated_history}

    # Only the last proposal of a turn is offered: an earlier draft of the same
    # turn is superseded and never reaches the store. If that last proposal is
    # refused (it is not a build_proposal() result, or the store is full), the
    # plain contract above still holds and no pending_save is returned.
    pending = PENDING_SAVES.register(proposals[-1]) if proposals else None
    if pending is not None:
        response["pending_save"] = pending

    return response


@app.post("/save")
def confirm_save(request: SaveRequest):
    """Write the research document of one confirmed proposal.

    The request only names a proposal. Everything that is written comes from the
    pending store of this server, which only /chat fills, so a client can neither
    inject content nor choose a file name or a location. Moving the proposal from
    "pending" to "saving" is atomic: a second request with the same id -- or one
    arriving while the first is still writing -- finds nothing and gets the same
    generic 404 as an unknown id. The file system is only touched after that
    transition and never with the store's lock held.

    The answer is small and path-free: the status and the plain file name that
    save_tool chose. A failed write gives the proposal back to the store, unless
    it expired meanwhile, so the user can simply try again. Once the writer
    returns, the document exists and the id is consumed for good, even if its
    answer cannot be echoed safely.
    """
    save_id = request.save_id

    proposal = PENDING_SAVES.begin_save(save_id)
    if proposal is None:
        raise HTTPException(status_code=404, detail=SAVE_NOT_FOUND_DETAIL)

    try:
        result = save_tool.write_proposal(proposal)
    except Exception:
        # Nothing was written: write_proposal() validates the name and removes
        # its own partial file before it raises, so the proposal may simply wait
        # again -- unless it expired meanwhile, in which case restore() retires it.
        # save_tool.SaveError already has a path-free message, but no exception
        # text is passed on at all here: a client gets the same generic 500
        # whatever went wrong inside the writer.
        PENDING_SAVES.restore(save_id)
        raise HTTPException(status_code=500, detail=SAVE_FAILED_DETAIL)

    # write_proposal() returned, so the document is really on disk: the save
    # happened and the id is used up from here on. It must never go back to
    # "pending", because a retry would find the file already there and the same
    # proposal could be handed to a second writer.
    PENDING_SAVES.discard(save_id)

    filename = result.get("file") if isinstance(result, dict) else None

    # The name is chosen by save_tool, never by the client, and is checked once
    # more before it is echoed: only a plain name inside the output folder is
    # ever reported back, so no path can reach the response. An unusable answer
    # cannot undo a write that already happened, so it only refuses the response
    # -- the id was consumed above and is not given back.
    try:
        save_tool.resolve_target(filename)
    except (save_tool.SaveError, TypeError):
        raise HTTPException(status_code=500, detail=SAVE_FAILED_DETAIL)

    return {"status": "saved", "filename": filename}


@app.post("/save/cancel")
def cancel_save(request: SaveRequest):
    """Drop one waiting proposal, so it can never be written any more.

    Only a proposal that is still waiting can be cancelled. An unknown, expired,
    already-cancelled, already-saved or currently-saving id all get the same
    generic 404. Cancelling and confirming are atomic with respect to each other,
    so a race between the two always has exactly one winner.
    """
    if not PENDING_SAVES.cancel(request.save_id):
        raise HTTPException(status_code=404, detail=SAVE_NOT_FOUND_DETAIL)
    return {"status": "cancelled"}

