"""A tiny chatbot backed by an OpenAI-compatible API.

Run it with:  python main.py

The chat logic is also reusable from other code (see server.py): call
chat(message, history) to get (reply, updated_history) without the terminal
loop. Configuration is read from a local .env file (see .env for the keys).

When the optional SEARCH_API_KEY variable is configured, the model may use the
web_search tool implemented in web_search.py; without it, the chatbot behaves
exactly as before and never tries to search.

A caller may additionally wrap a turn in collect_saves() (see below). Only then is
the save_research tool from save_tool.py offered to the model, and only after a
successful search has produced a real source URL. That tool merely prepares a
Markdown proposal for the current request and hands it back through the
collector; this module never writes a file.
"""

import contextlib
import contextvars
import json
import os
import time

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

import save_tool
import web_search

load_dotenv()

# Required configuration keys, in the order we want to report them.
REQUIRED_ENV_VARS = ("API_KEY", "API_BASE_URL", "MODEL_NAME")

# The system prompt is added by build_prompt() on every request and is never
# taken from the caller's history, so a client cannot change it.
#
# Besides the language rule it states the save contract, because the model must
# not confuse a prepared draft with a written file: save_research answers with
# status "ready" (see SAVE_READY_STATUS), which only means the proposal is ready,
# so the model has to ask for an explicit confirmation and must never claim that
# the file was saved on its own.
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer in Persian. "
    "You may save a finished web-research answer by calling the save_research "
    "tool with a title, a summary, the analysis and the sources the search tool "
    "returned. That tool only prepares a draft: it writes no file to disk. When "
    "it answers with status 'ready', that means the draft is ready, not that a "
    "file was saved. After 'ready', ask the user for a clear and unambiguous "
    "confirmation before saving, only treat the document as saved once the user "
    "has explicitly confirmed, and never claim that a file was saved otherwise."
)

# How many recent messages (besides the system message) are kept in memory
# and sent to the model. 10 messages is about 5 turns of conversation.
HISTORY_LIMIT = 10

EXIT_COMMANDS = {"exit", "quit"}

# Commands that reset the conversation. The system message is kept so the
# assistant's behaviour and language stay the same after clearing.
CLEAR_COMMANDS = {"clear"}

# Tool-calling budget. They only apply when SEARCH_API_KEY is configured (see
# web_search.py) and keep the extra work per message bounded, so the loop cannot
# run forever:
#
# - MAX_TOOL_ROUNDS rounds in which the model may ask for a tool, followed by
#   MAX_ANSWER_ROUNDS rounds that exist only to get the text answer the user
#   reads. The tools are offered in both, because the system prompt tells the
#   model it may call save_research: taking the tool channel away while that
#   instruction still stands is what made the provider write its tool call into
#   message.content as markup (see TOOL_CALL_MARKUP) instead of using
#   message.tool_calls, leaving the call unexecuted and the markup in the reply.
# - at most MAX_TOOL_CALLS web searches and MAX_SAVE_CALLS save proposals may
#   actually be executed. The two budgets are separate, so repeated searches can
#   never use up the room held back for a save proposal.
#
# One turn therefore costs at most MAX_TOOL_ROUNDS + MAX_ANSWER_ROUNDS model calls
# plus MAX_TOOL_CALLS + MAX_SAVE_CALLS tool calls, all inside the time budget
# below.
MAX_TOOL_ROUNDS = 2
MAX_ANSWER_ROUNDS = 2
MAX_TOOL_CALLS = 3
MAX_SAVE_CALLS = 2

# Message a tool answers with once its own budget is used up. Each budget has its
# own wording, because telling the model that "the budget" is gone would also rule
# out the other tool for no reason. It still says "tool", so the model reads it as
# a tool answer and not as a search result.
SEARCH_BUDGET_SPENT = "the web search tool budget for this message is used up"
SAVE_BUDGET_SPENT = "the save tool budget for this message is used up"

# The configured provider sometimes writes its own tool-call syntax into
# ``message.content`` instead of the structured ``message.tool_calls`` (observed as
# ``<||DSML|| calls>...``). A reply like that is a tool request that arrived on the
# wrong channel: it is not an answer, so it is never shown to the user and never
# parsed into an executable call (see is_tool_call_markup / reply_text).
TOOL_CALL_MARKUP = "<||DSML||"

# Shown instead of that markup when the model never produced a readable answer,
# even after MAX_TOOL_ROUNDS + MAX_ANSWER_ROUNDS chances. It is a short, plain
# message: the markup itself must never reach the caller.
TOOL_MARKUP_ERROR = (
    "The model asked for a tool instead of answering. Please try again."
)

# Status the model sees once save_research has prepared a proposal. It is
# deliberately not "saved": this module writes nothing, the proposal waits for
# the caller (server.py) to ask the user, so the word must not imply a file.
SAVE_READY_STATUS = "ready"

# One shared wall-clock budget for the whole tool-calling turn
# (get_reply_with_tools). The clock starts when that path is entered, so this
# covers *all* of the turn's model calls and searches together: every step only
# gets a slice of what is left, never a fresh full allowance.
#
# Why 100s: it is the ceiling for the slowest honest turn, not the expected one.
# A measured search turn on the configured provider needs ~25-40s in total (three
# searches, two rounds and a long answer), so the budget only comes into play for
# the slow tail. The browser aborts POST /chat after 130s
# (frontend/src/api/chat.ts, REQUEST_TIMEOUT_MS), so the server still gives up
# first and the user gets a real status code instead of a dead socket.
#
# It also has to hold back the whole answer reserve *and* leave the rounds their
# share: 100 - 40 = 60s of rounds and searches. The plain path (no search key) is
# untouched: it keeps the client's own settings from build_client().
TOOL_DEADLINE_SECONDS = 100.0

# Held back from the work rounds for the final answer, which is what
# the user actually reads. It is also exactly the answer's own cap (below), since
# a reserve smaller than the cap would not protect the answer it is held back for.
#
# Sized from measurement, not from taste: an analytical request ("search for X
# and analyse the results") came back as an 8700-character answer that took 24.9s
# to write on the configured model, against ~9s for a short factual one. A reserve
# below that is not a safety margin -- it is what turns a slow answer into a
# failed request (see TOOL_ANSWER_TIMEOUT_SECONDS below).
TOOL_ANSWER_RESERVE_SECONDS = 40.0

# Longest a single model call *with tools* (one round) may take. A round is not
# necessarily short: the model may answer at length instead of asking for a tool,
# and one measured round needed over 25s. This is a per-request override
# (bounded_client), so it does not depend on the client's own 20s setting.
#
# A round that still runs out of time does not fail the turn: the loop drops that
# round and answers from what the earlier rounds already gathered (see
# get_reply_with_tools).
TOOL_STEP_TIMEOUT_SECONDS = 30.0

# Longest one answer round may take: it is the one reply the user reads, so it gets
# the reserve held back above. Every step is also bounded by the time left
# (step_timeout), so a slow round shrinks this step instead of pushing the turn
# past TOOL_DEADLINE_SECONDS.
TOOL_ANSWER_TIMEOUT_SECONDS = 40.0

# A step is only started while at least this much time is left. Below it the
# deadline has effectively passed, so the loop stops instead of starting a call
# that is bound to be cut off immediately.
TOOL_MIN_STEP_SECONDS = 2.0


class ConfigError(Exception):
    """Raised when the required environment variables are missing."""


class ToolDeadlineExceeded(Exception):
    """The tool-calling turn ran out of its shared time budget.

    Raised instead of returning a partial or empty answer, so both entry points
    report a controlled error: the terminal loop prints it through
    describe_error(), and server.py maps it to HTTP 504. A plain turn (no search
    key) never raises this, because it is not time-boxed here.
    """


def get_config():
    """Return the (api_key, base_url, model_name) tuple we need.

    Raises ConfigError when something is missing. Only the variable *names*
    are ever put in the message, never their values.
    """
    values = {name: os.getenv(name) for name in REQUIRED_ENV_VARS}
    missing = [name for name in REQUIRED_ENV_VARS if not values[name]]

    if missing:
        raise ConfigError(
            "Missing required environment variable(s): " + ", ".join(missing)
        )

    return values["API_KEY"], values["API_BASE_URL"], values["MODEL_NAME"]


def load_config():
    """Command-line helper: return the config, or print an error and None.

    This keeps the terminal chatbot friendly: a missing .env value shows a
    short message instead of a traceback.
    """
    try:
        return get_config()
    except ConfigError as error:
        print(str(error))
        print("Please add them to your .env file and try again.")
        return None


def keep_recent(history):
    """Keep only the most recent HISTORY_LIMIT messages."""
    return history[-HISTORY_LIMIT:]


def trim_history(history):
    """Keep the system message plus the most recent HISTORY_LIMIT messages."""
    system_message, *rest = history
    return [system_message] + keep_recent(rest)


def build_prompt(history, user_message):
    """Build the message list sent to the model.

    The system message always comes from SYSTEM_PROMPT in this file, so a
    caller can never replace it, and the HISTORY_LIMIT rule is applied here.
    """
    system_message = {"role": "system", "content": SYSTEM_PROMPT}
    prompt = [system_message] + history + [{"role": "user", "content": user_message}]
    return trim_history(prompt)


def build_client(api_key, base_url):
    """Create a configured OpenAI client.

    ``timeout=20`` is the per-operation budget of the underlying HTTP client
    (connect/read/write), *not* a hard deadline for the whole request: the SDK
    also retries once (``max_retries=1``) and sleeps a short backoff between
    attempts, and a provider-sent ``Retry-After`` header can extend the wait
    further. Worst case for a single /chat call is therefore roughly
    ``2 * 20s + backoff`` (~41s) and may be longer, which is why the frontend
    timeout in frontend/src/api/chat.ts is set above it.
    """
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        max_retries=1,
        timeout=20,
    )


def get_reply(client, model_name, messages):
    """Send the messages to the model and return the reply text (or None)."""
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
    )
    if not response.choices:
        return None
    return response.choices[0].message.content


def tool_clock(now=None):
    """Return the clock used by the tool-calling time budget.

    ``now`` is injectable so the tests can move time forward without sleeping.
    """
    return now or time.monotonic


def bounded_client(client, timeout):
    """Return a view of ``client`` whose one request cannot outlive ``timeout``.

    ``timeout`` is the per-request override of the SDK; ``max_retries=0`` drops
    the client's retry, because a retry would repeat the same wait and turn a
    slice of the budget into twice that slice. ``with_options`` shares the
    underlying HTTP client, so no extra connection pool is created.

    Note the limit of what this can do: it bounds how long a request may *stay*
    open, not the ability to cancel one. Neither the SDK nor urllib can abort a
    request that is already in flight, so the budget below stops the loop from
    starting new work; it cannot make network I/O stop instantly.
    """
    return client.with_options(timeout=timeout, max_retries=0)


def step_timeout(deadline, clock, cap, reserve=0.0):
    """Return the timeout for one step, or None when it must not be started.

    The value is the smaller of the time left (after ``reserve``) and ``cap``, so
    no step ever gets a timeout longer than the budget it belongs to. None means
    less than TOOL_MIN_STEP_SECONDS is left, i.e. starting the step would only
    burn the rest of the budget on work that is bound to be cut off.
    """
    left = deadline - reserve - clock()
    if left < TOOL_MIN_STEP_SECONDS:
        return None
    return min(left, cap)


def request_message(client, model_name, messages, tools=None, timeout=None):
    """Make one Chat Completions call and return the assistant message or None.

    ``tools`` is left out of the request entirely when it is None, so the plain
    call is the same one get_reply() already sends. ``timeout`` is only used by
    the time-boxed tool path: when it is None the client keeps its own settings.
    """
    request = {"model": model_name, "messages": messages}
    if tools:
        request["tools"] = tools
    if timeout is not None:
        client = bounded_client(client, timeout)

    response = client.chat.completions.create(**request)
    if not response.choices:
        return None
    return response.choices[0].message


def tool_call_parts(call):
    """Return (call_id, name, raw_arguments) for one tool call, or None.

    Only the shape the SDK models declare is accepted. A call we cannot read is
    dropped instead of guessed at, and because the dropped call is left out of
    the assistant message as well, the tool results stay consistent with it.
    """
    function = getattr(call, "function", None)
    call_id = getattr(call, "id", None)
    name = getattr(function, "name", None)
    arguments = getattr(function, "arguments", None)

    if not isinstance(call_id, str) or not isinstance(name, str):
        return None
    if not isinstance(arguments, (str, dict)):
        arguments = ""
    return call_id, name, arguments


def collect_tool_calls(message):
    """Return the readable tool calls of an assistant message."""
    calls = []
    for call in getattr(message, "tool_calls", None) or []:
        parts = tool_call_parts(call)
        if parts is not None:
            calls.append(parts)
    return calls


def build_assistant_tool_message(message, calls):
    """Build the assistant turn that must precede the tool results."""
    turn = {"role": "assistant"}
    if message.content:
        turn["content"] = message.content
    turn["tool_calls"] = [
        {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": raw_arguments or ""},
        }
        for call_id, name, raw_arguments in calls
    ]
    return turn


def is_tool_call_markup(content):
    """True when a reply is the model's own tool-call markup, not an answer.

    The configured provider sometimes writes its tool-call syntax into
    ``message.content`` -- ``<||DSML|| calls>`` and friends -- instead of filling
    in ``message.tool_calls``. Such a reply is a tool request that arrived on the
    wrong channel, so it is not an answer. It is never parsed into an executable
    call either: reading arguments out of free text is as unsafe as trusting a
    fabricated source. The model instead gets further rounds with the tools still
    offered, so it can repeat the request properly (see get_reply_with_tools).
    """
    return isinstance(content, str) and TOOL_CALL_MARKUP in content


def reply_text(content):
    """Return the user-visible text of a model reply, or None.

    The one place a model reply is turned into the caller's reply, so the
    tool-call markup can never reach the user: a reply that is (or contains) the
    markup becomes the short TOOL_MARKUP_ERROR message instead.
    """
    if is_tool_call_markup(content):
        return TOOL_MARKUP_ERROR
    return content


# --------------------------------------------------------------------------
# save_research: proposals belong to the current request, not to the module
# --------------------------------------------------------------------------

# A save proposal is per-request state, so it lives in a ContextVar instead of a
# module global: every request runs in its own context (its own thread or task),
# so concurrent requests can never read each other's proposals, and code that
# never calls collect_saves() -- the terminal loop, a plain chat() turn -- keeps
# the default of None, i.e. "no collector" and therefore no save tool at all.
#
# Nothing in this module writes a file: the collected proposals are handed to the
# caller (server.py), which owns the user's decision.
_PENDING_SAVES = contextvars.ContextVar("pending_saves", default=None)


@contextlib.contextmanager
def collect_saves():
    """Collect the save proposals made while the with-block runs.

    Yields the list the proposals are appended to, so the caller can read them
    once the turn is over. The previous value of the ContextVar is always
    restored, both on the normal path and when the body raises, so a later turn
    never inherits a stale collector.
    """
    proposals = []
    token = _PENDING_SAVES.set(proposals)
    try:
        yield proposals
    finally:
        _PENDING_SAVES.reset(token)


def record_proposal(proposal):
    """Append one proposal to the collector of the current request.

    Returns True when it was recorded and False when this context has no active
    collector (or was handed something that is not a build_proposal() result).
    A False result means the proposal is dropped: no global state is created and
    it is never passed on to another request.
    """
    collector = _PENDING_SAVES.get()
    if collector is None or not isinstance(proposal, dict):
        return False
    collector.append(proposal)
    return True


def known_source_urls(content):
    """Return the real URLs inside one web_search tool result.

    The search tool answers with a small JSON object -- ``{"query": ...,
    "results": [...]}`` when it worked and ``{"error": ...}`` when it did not --
    so a failed, empty or unreadable result simply yields no URL. Each URL is
    validated by save_tool.normalize_source_url() and duplicates are dropped, so
    only URLs the search really returned can ever become a source; a URL the
    model made up is never in this list.
    """
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        return []
    if not isinstance(payload, dict):
        return []

    results = payload.get("results")
    if not isinstance(results, list):
        return []

    urls = []
    for item in results:
        if not isinstance(item, dict):
            continue
        url = save_tool.normalize_source_url(item.get("url"))
        if url and url not in urls:
            urls.append(url)
    return urls


def remember_source_urls(known_urls, content):
    """Add the real URLs of one search result to ``known_urls``, in place.

    ``known_urls`` accumulates across the rounds of one turn: the sources offered
    to save_research are the URLs this turn's searches returned, and nothing else.
    """
    for url in known_source_urls(content):
        if url not in known_urls:
            known_urls.append(url)
    return known_urls


def search_query_key(raw_arguments):
    """Return the canonical query of a web_search call, or None.

    Only used for a call whose name is already known to be web_search (see
    get_reply_with_tools). It recognizes a query that was already searched in this
    turn, so an identical query is answered from the result the model already has
    instead of being sent to the service a second time; genuinely different
    queries are never affected.

    The query is normalized exactly as web_search.normalize_query() does it, so
    extra whitespace cannot hide a repeat, and the result is case-folded so
    "Tehran" and "tehran" count as the same search.

    None means "not readable as a web_search query": a malformed argument payload,
    or arguments that are not a mapping. The call is then passed to the tool
    unchanged and keeps its normal error answer, so this helper can never turn a
    malformed call into a different outcome.
    """
    if isinstance(raw_arguments, (bytes, bytearray)):
        try:
            raw_arguments = bytes(raw_arguments).decode("utf-8")
        except UnicodeDecodeError:
            return None

    if isinstance(raw_arguments, str):
        try:
            raw_arguments = json.loads(raw_arguments)
        except (TypeError, ValueError):
            return None

    if not isinstance(raw_arguments, dict):
        return None

    try:
        return web_search.normalize_query(raw_arguments.get("query")).casefold()
    except web_search.SearchError:
        return None


def save_tool_offered(known_urls):
    """True when save_research may be offered for this round.

    Both halves are required: a collector is active for this request, and at
    least one real source URL came back from a successful search in this turn.
    Without that, either the caller has nowhere to put a proposal (CLI, plain
    turn) or the model would have nothing real to cite.
    """
    return bool(known_urls) and _PENDING_SAVES.get() is not None


def build_tools(known_urls=None):
    """Return the tools the model may use in one round.

    web_search is always offered (this path only runs when it is configured).
    save_research is appended only when save_tool_offered() says so, so it is
    absent at the start of a turn, after a failed search and whenever no source
    came back. The list is rebuilt every round, because it changes as soon as a
    search has produced a usable URL.
    """
    tools = list(web_search.build_tools())
    if save_tool_offered(known_urls):
        tools.append(save_tool.build_tool_spec())
    return tools


def execute_save_tool(raw_arguments, known_urls):
    """Prepare one save proposal and return the JSON answer for the model.

    The arguments are validated, the Markdown is rendered and the sources are
    filtered by save_tool.build_proposal(), which is also what selects the file
    name: none of that is repeated here. ``known_urls`` is passed explicitly, so
    a source the model invented is dropped instead of being written down.

    The finished proposal is handed to the current request with
    record_proposal(). No file is created: write_proposal() is never called from
    this module; the caller decides what happens after the user is asked.

    Every failure becomes a small JSON error object from save_tool, so the model
    can correct its arguments and the conversation continues; the message never
    contains an absolute path or an internal value.
    """
    if _PENDING_SAVES.get() is None:
        return save_tool.error_content("saving is not available for this request")
    if not known_urls:
        return save_tool.error_content(
            "no source from the web search is available yet"
        )

    try:
        proposal = save_tool.build_proposal(raw_arguments, known_urls=known_urls)
    except save_tool.SaveError as error:
        return save_tool.error_content(str(error))
    except Exception:
        # A bug inside the tool must not break the conversation: the model gets a
        # short error object and can still answer. It is one attempt, no retry.
        return save_tool.error_content("the save tool failed")

    if not record_proposal(proposal):
        return save_tool.error_content("saving is not available for this request")

    return save_tool.ack_content(
        {
            "status": SAVE_READY_STATUS,
            "file": proposal["filename"],
            "title": proposal["title"],
            "sources": len(proposal["sources"]),
            "bytes": proposal["bytes"],
        }
    )


def tool_error(name, message):
    """Return one tool's error object, in the shape that tool's model expects."""
    if name == save_tool.TOOL_NAME:
        return save_tool.error_content(message)
    return web_search.error_content(message)


def get_reply_with_tools(client, model_name, messages, now=None):
    """get_reply() plus a bounded web_search/save_research tool-calling loop.

    The turn has two kinds of round, and the tools are offered in both:

    - at most MAX_TOOL_ROUNDS work rounds, in which a call the model asks for is
      executed (subject to the budgets below);
    - at most MAX_ANSWER_ROUNDS rounds that exist only to obtain the text answer
      the user reads. A call asked for in one of those is answered with the
      spent-budget message instead of being run, which is what ends the turn.

    Offering the tools in the answer rounds too is the point of that shape: the
    system prompt tells the model it may call save_research, so a request that
    takes the tool channel away contradicts the model's own instructions -- and
    that contradiction is what made the provider write its call into
    message.content as markup (see TOOL_CALL_MARKUP) instead of using
    message.tool_calls. The save was then never executed and the markup ended up
    in the user's reply. With the tools still offered, a model that wants to save
    says so on the structured channel and the call is executed normally.

    The tool budgets are separate: at most MAX_TOOL_CALLS web searches and at most
    MAX_SAVE_CALLS save proposals are executed per message, so repeated searches
    can never use up the room held back for a save, and a query already searched in
    this turn is never searched again (see search_query_key).

    Tool requests, their arguments and their raw results live only in the temporary
    list built here: they are never added to the history the caller keeps, and
    neither is a save proposal -- that goes to the request's collector instead (see
    collect_saves).

    The offered tools are rebuilt before every round by build_tools(): web_search
    is always there, save_research only while a collector is active for this
    request and a successful search has already produced a real source URL.

    The whole turn runs under one shared budget of TOOL_DEADLINE_SECONDS, counted
    from entry with ``now`` (a monotonic clock by default, injectable for tests).
    Before every model call and every search the time left is checked, and the
    step's own timeout is capped by what remains, so a slow provider cannot
    restart the budget on each round. A step that cannot be started any more is
    skipped; when the budget is gone altogether, ToolDeadlineExceeded is raised
    instead of returning a partial answer.

    This bounds when work *starts*; it cannot cancel a request that is already in
    flight (see bounded_client). A round whose own call times out is the one
    exception to that hard edge: it is dropped instead of failing the turn, so a
    single slow round costs the turn its next round rather than its answer.

    Every call the model asks for is answered, so the API contract holds: a call
    that cannot be run (budget spent, a query already searched, no time left, an
    unknown name) becomes a short JSON error object instead, and the loop keeps
    making progress.

    Returns the final reply text, or None when the model produced no text at all.
    A reply that is only the model's own tool-call markup becomes the short
    TOOL_MARKUP_ERROR message, so that markup never reaches the caller.
    """
    clock = tool_clock(now)
    deadline = clock() + TOOL_DEADLINE_SECONDS

    conversation = list(messages)
    # Two separate tool budgets, so repeated searches can never spend the room
    # held back for a save proposal.
    searches_left = MAX_TOOL_CALLS
    saves_left = MAX_SAVE_CALLS
    # The real URLs this turn's searches returned. It starts empty, so
    # save_research is not offered until a search has actually produced a source.
    known_urls = []
    # The queries already searched in this turn, in search_query_key() form.
    searched_queries = []

    work_rounds_left = MAX_TOOL_ROUNDS
    answer_rounds_left = MAX_ANSWER_ROUNDS
    # True when a round could no longer be started because the turn's time was
    # gone: that is reported as a deadline, not as an empty answer.
    deadline_hit = False
    # The last reply that asked for a tool on the text channel, if any.
    unanswered_markup = None

    while work_rounds_left > 0 or answer_rounds_left > 0:
        if work_rounds_left > 0:
            timeout = step_timeout(
                deadline, clock, TOOL_STEP_TIMEOUT_SECONDS, TOOL_ANSWER_RESERVE_SECONDS
            )
            if timeout is None:
                # Not enough time is left for another work round: keep the rest for
                # the answer instead of starting work we cannot finish.
                work_rounds_left = 0
                continue
            work_rounds_left -= 1
        else:
            # The answer is the one reply the user reads, so it gets the answer cap
            # (backed by the reserve the work rounds held back) rather than the
            # smaller per-round one. When not even that can be started, the whole
            # turn is spent and there is nothing left to answer with.
            timeout = step_timeout(deadline, clock, TOOL_ANSWER_TIMEOUT_SECONDS)
            if timeout is None:
                deadline_hit = True
                break
            answer_rounds_left -= 1

        # Rebuilt every round: save_research appears only once known_urls is
        # non-empty, i.e. only after a search has succeeded in this turn.
        tools = build_tools(known_urls)
        try:
            message = request_message(
                client, model_name, conversation, tools, timeout=timeout
            )
        except APITimeoutError:
            # This one round was too slow to finish inside its slice. That is not
            # worth failing the whole turn for: the searches of the earlier rounds
            # are already in the conversation, so the loop moves straight to the
            # answer rounds, which still have the whole reserve to write it. Every
            # other failure (an unreachable provider, a rejected key, a rate limit)
            # still reaches the caller unchanged.
            work_rounds_left = 0
            continue
        if message is None:
            return None

        calls = collect_tool_calls(message)
        if not calls:
            if not is_tool_call_markup(message.content):
                # A real answer: no tool was asked for in a round where the tools
                # were offered, so the turn is finished.
                return message.content
            # The model asked for a tool on the wrong channel instead. That markup
            # is not an answer and is never parsed into a call: the loop gives the
            # model further rounds to repeat the request on the structured channel,
            # and the turn ends with TOOL_MARKUP_ERROR if it never does (see
            # reply_text). The round budgets above bound those retries.
            unanswered_markup = message.content
            continue

        conversation.append(build_assistant_tool_message(message, calls))
        for call_id, name, raw_arguments in calls:
            if name == save_tool.TOOL_NAME:
                if saves_left <= 0:
                    # The save budget is used up. Every requested call still gets
                    # an answer, so the API contract holds.
                    content = tool_error(name, SAVE_BUDGET_SPENT)
                else:
                    # Preparing a proposal is cheap and touches no file, so it needs
                    # no time slice, and its own budget is what keeps it out of reach
                    # of the searches of this turn.
                    saves_left -= 1
                    content = execute_save_tool(raw_arguments, known_urls)
            else:
                # web_search -- or a name the model made up, which execute_tool()
                # answers with a short error object instead of a search.
                query = (
                    search_query_key(raw_arguments)
                    if name == web_search.TOOL_NAME
                    else None
                )
                if query is not None and query in searched_queries:
                    # The same query was already searched in this turn: answering it
                    # from the earlier result costs no budget and no time, and it is
                    # what stops a turn from being spent on one search over and
                    # over. Genuinely different queries are never affected.
                    content = web_search.error_content(
                        f"the query '{query}' was already searched in this message"
                    )
                else:
                    search_timeout = step_timeout(
                        deadline,
                        clock,
                        web_search.REQUEST_TIMEOUT_SECONDS,
                        TOOL_ANSWER_RESERVE_SECONDS,
                    )
                    if search_timeout is None:
                        content = web_search.error_content(
                            "the time budget for this message is used up"
                        )
                    elif searches_left <= 0:
                        content = tool_error(name, SEARCH_BUDGET_SPENT)
                    else:
                        searches_left -= 1
                        if query is not None:
                            searched_queries.append(query)
                        try:
                            content = web_search.execute_tool(
                                name, raw_arguments, timeout=search_timeout
                            )
                        except Exception:
                            # A bug inside the tool must not break the conversation:
                            # the model gets a short error object and can still
                            # answer. It is one attempt, never a retry loop.
                            content = web_search.error_content(
                                "the web search tool failed"
                            )
                        else:
                            # Only URLs the search really returned are remembered,
                            # so save_research can never cite an invented source.
                            remember_source_urls(known_urls, content)
            conversation.append(
                {"role": "tool", "tool_call_id": call_id, "content": content}
            )

    # The loop is over without a text answer.
    if deadline_hit:
        # Nothing could be started any more: the turn's own time budget is gone.
        raise ToolDeadlineExceeded(
            "the web search turn ran out of time before an answer could be made"
        )
    if unanswered_markup is not None:
        # A tool request that stayed on the text channel until the last round: the
        # caller gets a short, readable message, never the raw markup.
        return reply_text(unanswered_markup)
    # The model replied, but never with text: its last answer was a tool call the
    # turn had no room left for. That is the same empty answer as before.
    return None


def chat(message, history=None, client=None, model_name=None):
    """Send one message to the model and return (reply, updated_history).

    This is the reusable entry point (used by server.py). The terminal loop in
    main() uses it too, passing the client it already built.

    The returned history holds only "user" and "assistant" messages, never the
    system message, never any internal tool-calling message and never a save
    proposal (those go to the request's collector; see collect_saves). When the
    model replies with nothing, or when the request fails, the history is
    returned unchanged so a failed turn is never stored as a successful exchange.

    When a search key is configured the tool turn also runs under its own time
    budget; if that budget runs out, ToolDeadlineExceeded is raised (see
    get_reply_with_tools) so callers report the timeout instead of an empty turn.
    """
    if history is None:
        history = []

    if client is None or model_name is None:
        api_key, base_url, model_name = get_config()
        client = build_client(api_key, base_url)

    prompt = build_prompt(history, message)

    if web_search.is_available():
        # A search key is configured, so the model may use the web_search tool
        # for this turn. Without it, the original single-call path below runs
        # unchanged, so behaviour is identical when search is not set up.
        reply = get_reply_with_tools(client, model_name, prompt)
    else:
        reply = get_reply(client, model_name, prompt)

    if not reply:
        return None, keep_recent(history)

    updated_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]
    return reply, keep_recent(updated_history)


def describe_error(error):
    """Turn an exception into a short, friendly message."""
    if isinstance(error, AuthenticationError):
        return "Authentication failed. Check your API key."
    if isinstance(error, RateLimitError):
        return "Rate limit exceeded. Check your provider limits."
    if isinstance(error, APITimeoutError):
        # Not a dead provider: the tool path caps every call, so this is usually
        # our own cap expiring on a genuinely slow answer (see
        # TOOL_ANSWER_TIMEOUT_SECONDS / TOOL_DEADLINE_SECONDS). It must be checked
        # before APIConnectionError, because the SDK derives it from there.
        return "The model took too long to answer. Please try again."
    if isinstance(error, APIConnectionError):
        # Reaching here means the provider was never reached at all.
        return "Connection failed. Check your internet or API endpoint."
    if isinstance(error, APIStatusError):
        return f"API returned HTTP {error.status_code}."
    if isinstance(error, ToolDeadlineExceeded):
        # Our own signal for the tool turn's budget, not a provider failure.
        return (
            "The web search turn took too long and was stopped. "
            "Please try again with a simpler question."
        )
    return f"Unexpected error ({type(error).__name__}): {error}"


def main():
    config = load_config()
    if config is None:
        return

    api_key, base_url, model_name = config
    client = build_client(api_key, base_url)

    # Conversation memory: "user" and "assistant" messages only. The system
    # message is added by build_prompt() on every request, so the assistant's
    # behaviour and language always come from SYSTEM_PROMPT above.
    history = []

    print("Chatbot is ready! Type 'exit' to quit.")

    # A single KeyboardInterrupt handler wraps the whole loop, so Ctrl+C is
    # handled cleanly both while typing a message and while waiting for the
    # API response (KeyboardInterrupt is a BaseException, so the API
    # "except Exception" blocks below intentionally do not swallow it).
    try:
        while True:
            try:
                user_message = input("You: ").strip()
            except EOFError:
                print("\nGoodbye!")
                break

            if user_message.lower() in EXIT_COMMANDS:
                print("Goodbye!")
                break

            if user_message.lower() in CLEAR_COMMANDS:
                # Reset the conversation. The system message is re-added by
                # build_prompt(), so the assistant keeps behaving the same way.
                history = []
                print("Conversation history cleared.")
                continue

            if not user_message:
                continue

            # chat() only returns an updated history for a successful turn, so
            # a failed request never leaves a dangling user message in memory.
            try:
                bot_message, history = chat(user_message, history, client, model_name)
            except Exception as error:
                # describe_error() maps every exception to one short, friendly
                # message, so the user never sees a raw traceback.
                print(describe_error(error))
                continue

            if not bot_message:
                print("Bot: [empty response]")
                continue

            print(f"Bot: {bot_message}")
    except KeyboardInterrupt:
        print("\nChatbot stopped.")


if __name__ == "__main__":
    main()
