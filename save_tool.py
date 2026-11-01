"""Optional save tool for the chatbot: stores finished research as Markdown.

The model may ask for one extra tool, ``save_research``, that writes the result
of a web-research turn into a Markdown document inside the project's
``search_results`` folder. The model only ever supplies *content* -- a title, a
summary, the analysis and the sources it was actually given by web_search. It can
never choose a path, a folder or a file name, and nothing it produces is executed.

Rules enforced here:

- Only the standard library is used, so no new dependency is required.
- The output folder is fixed: ``PROJECT_ROOT/search_results``. A different folder
  can only be passed by the embedding code (tests), never by the model.
- The file name is derived from the title (UTF-8 safe slug) plus a UTC timestamp
  and a short random token, and is re-validated against path traversal before any
  write. No ``..``, no separator, no absolute or drive-qualified path survives.
- Everything is UTF-8. Title, summary, analysis, every source, the whole file and
  each acknowledgement have hard limits; an over-long value is refused instead of
  being silently truncated.
- A source only survives when it is a valid ``http``/``https`` URL with a host;
  duplicates are dropped, and when the caller knows which URLs the search really
  returned, any other URL (a fabricated source) is dropped as well.
- ``build_proposal()`` never touches the file system: it only renders text.
  ``write_proposal()`` is the single function allowed to create a file -- it
  creates it exclusively (an existing file is never replaced) and removes its own
  partial file when a write fails.
- Errors are small JSON objects the model can read; they never contain an
  absolute system path, file contents or any other internal value.
"""

import json
import os
import secrets
from datetime import datetime, timezone
from urllib.parse import urlsplit

# The one tool name the model is allowed to ask for.
TOOL_NAME = "save_research"

# The output folder is fixed and derived from this file's own location, so it can
# never be pointed somewhere else by the model.
OUTPUT_DIR_NAME = "search_results"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, OUTPUT_DIR_NAME)

# Extension used for every saved research document.
FILE_EXTENSION = ".md"

# Hard limits for a single save. Larger values are refused, so one tool call can
# neither store a huge file nor flood the model's context.
MAX_TITLE_LENGTH = 200
MAX_SUMMARY_LENGTH = 1000
MAX_ANALYSIS_LENGTH = 8000
MAX_SOURCE_TITLE_LENGTH = 200
MAX_URL_LENGTH = 500
MAX_SOURCES = 5
MAX_FILE_BYTES = 200_000
MAX_ACK_LENGTH = 600
MAX_ACK_FIELD_LENGTH = 200
MAX_ERROR_LENGTH = 200

# File-name budget: UTC timestamp label, title slug and random token.
STAMP_FORMAT = "%Y%m%d-%H%M%S"
MAX_LABEL_LENGTH = 40
MAX_SLUG_LENGTH = 60
MAX_TOKEN_LENGTH = 16
TOKEN_BYTES = 3
SLUG_FALLBACK = "research"

# Names Windows refuses (or reinterprets) even when an extension is appended. The
# generated stem always starts with the timestamp, so this is only a safety net.
RESERVED_NAMES = frozenset(
    ["con", "prn", "aux", "nul"]
    + [f"com{index}" for index in range(1, 10)]
    + [f"lpt{index}" for index in range(1, 10)]
)

# The only fields that may travel back to the model in an acknowledgement.
ACK_FIELDS = ("status", "file", "title", "sources", "bytes", "error")

# The exact shape build_proposal() produces. A proposal is validated against this
# in one place only -- copy_proposal() below -- and both server.py's pending store
# and write_proposal() go through it, so the contract cannot drift between them.
PROPOSAL_TEXT_FIELDS = ("title", "summary", "analysis", "filename", "content")
PROPOSAL_SIZE_FIELD = "bytes"


class SaveError(Exception):
    """A research document could not be prepared or saved.

    Messages are short, model-readable and never contain an absolute system path,
    a file's contents or any other internal value.
    """


def strip_control(text):
    """Drop control characters but keep tabs and newlines, for multi-line text."""
    return "".join(
        char for char in text if (char >= " " and char != "\x7f") or char in "\n\t"
    )


def single_line(value):
    """Return ``value`` as one collapsed line. Non-text values become ""."""
    if not isinstance(value, str):
        if isinstance(value, (int, float)):
            value = str(value)
        else:
            return ""
    return " ".join(strip_control(value).split())


def multi_line(value):
    """Return ``value`` with normalized line breaks and no control characters."""
    if not isinstance(value, str):
        return ""
    text = strip_control(value.replace("\r\n", "\n").replace("\r", "\n"))
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def normalize_text(value, limit, name, multiline=False):
    """Validate one text argument and return it cleaned, or raise SaveError.

    Empty, non-text and over-long values are refused, so a malformed tool call can
    never get near a file.
    """
    if not isinstance(value, str):
        raise SaveError(f"the '{name}' argument is required and must be a string")

    text = multi_line(value) if multiline else single_line(value)
    if not text:
        raise SaveError(f"the '{name}' argument was empty")
    if len(text) > limit:
        raise SaveError(f"the '{name}' argument must be at most {limit} characters")
    return text


def build_tool_spec():
    """Return the function-tool schema the model sees for save_research.

    The schema deliberately has no ``path``, ``folder`` or ``filename`` property:
    the model can only supply content, so it can never ask for a location.

    The description is worded just as carefully, because it is the model's only
    instruction for this tool: it says the tool merely *prepares a proposal* and
    writes no file to disk, that a successful call answers with status "ready",
    and that the model has to ask the user for an explicit confirmation after
    that -- it must never claim that a file was saved by itself.
    """
    return {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": (
                "Prepare a save proposal for a finished web-research answer: a "
                "Markdown draft with a title, a summary, the analysis and the "
                "sources the search tool returned. This tool only prepares the "
                "proposal -- it writes no file to disk. Pass only the text and "
                "the sources; the file name and the folder are chosen "
                "automatically and are used later, once the user confirms. A "
                "successful call answers with status 'ready', meaning the draft "
                "is ready, not that a file was saved. After 'ready', ask the "
                "user for a clear, explicit confirmation before saving, and "
                "never claim that a file has been saved."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Short title of the research, on one line.",
                    },
                    "summary": {
                        "type": "string",
                        "description": "A short summary of the findings.",
                    },
                    "analysis": {
                        "type": "string",
                        "description": "The full research text, in Markdown.",
                    },
                    "sources": {
                        "type": "array",
                        "description": (
                            "The sources exactly as the search tool returned "
                            "them. Anything the search did not return is dropped."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "url": {"type": "string"},
                            },
                            "required": ["url"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["title", "summary", "analysis", "sources"],
                "additionalProperties": False,
            },
        },
    }


def build_tools():
    """Return the tool list passed to the Chat Completions API."""
    return [build_tool_spec()]


def normalize_source_url(value):
    """Return a clean http(s) URL, or None when the value cannot be trusted.

    Only absolute ``http``/``https`` URLs with a host survive. Anything else --
    another scheme, a host-less or relative URL, embedded credentials, a
    backslash, whitespace, control characters, an over-long value -- is refused,
    so a fabricated or dangerous link is simply dropped instead of written.
    """
    if isinstance(value, (bytes, bytearray)):
        try:
            value = bytes(value).decode("utf-8")
        except UnicodeDecodeError:
            return None
    if not isinstance(value, str):
        return None

    url = value.strip()
    if not url or len(url) > MAX_URL_LENGTH:
        return None
    if "\\" in url or any(char <= " " or char == "\x7f" for char in url):
        return None

    try:
        parts = urlsplit(url)
        hostname = parts.hostname
    except ValueError:
        return None

    if parts.scheme.lower() not in ("http", "https"):
        return None
    if not parts.netloc or not hostname:
        return None
    if "@" in parts.netloc:
        return None
    return url


def select_sources(raw_sources, known_urls=None):
    """Return the sources that will be written into the document.

    ``raw_sources`` is what the model produced: a list whose entries are either a
    URL string or a mapping with at least a ``url``. Every entry must be a real
    ``http``/``https`` URL; unusable entries and duplicates are dropped and at
    most MAX_SOURCES survive, so a hallucinated or repeated list can neither flood
    the file nor invent a link.

    ``known_urls`` is optional. When it is given it must be the collection of URLs
    the search really returned: a source outside it is fabricated and is dropped
    as well. ``None`` means "there is no search grounding to compare with", and
    only the URL checks above apply.
    """
    if raw_sources is None:
        return []
    if not isinstance(raw_sources, list):
        raise SaveError("the 'sources' argument must be a list of search results")

    if isinstance(known_urls, str):
        known_urls = [known_urls]
    allowed = None
    if known_urls is not None:
        known_urls = list(known_urls) if isinstance(
            known_urls, (list, tuple, set, frozenset)
        ) else None
        if known_urls is None:
            raise SaveError("the expected source list must be a collection of URLs")
        allowed = set()
        for candidate in known_urls:
            url = normalize_source_url(candidate)
            if url is not None:
                allowed.add(url)

    selected = []
    seen = set()
    for item in raw_sources:
        if len(selected) >= MAX_SOURCES:
            break

        if isinstance(item, str):
            candidate_url, candidate_title = item, ""
        elif isinstance(item, dict):
            candidate_url, candidate_title = item.get("url"), item.get("title")
        else:
            continue

        url = normalize_source_url(candidate_url)
        if url is None or url in seen:
            continue
        if allowed is not None and url not in allowed:
            continue

        seen.add(url)
        title = single_line(candidate_title)[:MAX_SOURCE_TITLE_LENGTH]
        selected.append({"title": title or url, "url": url})
    return selected


def normalize_arguments(arguments, known_urls=None):
    """Validate a save_research argument mapping and return its clean form.

    Returns ``{"title", "summary", "analysis", "sources"}`` where every value has
    been cleaned and bounded and ``sources`` already went through
    select_sources(). A missing or malformed field raises SaveError.
    """
    if not isinstance(arguments, dict):
        raise SaveError("the tool arguments must be a JSON object")

    if "sources" not in arguments:
        raise SaveError("the 'sources' argument is required and must be a list")

    return {
        "title": normalize_text(arguments.get("title"), MAX_TITLE_LENGTH, "title"),
        "summary": normalize_text(
            arguments.get("summary"), MAX_SUMMARY_LENGTH, "summary", multiline=True
        ),
        "analysis": normalize_text(
            arguments.get("analysis"), MAX_ANALYSIS_LENGTH, "analysis", multiline=True
        ),
        "sources": select_sources(arguments.get("sources"), known_urls),
    }


def parse_arguments(raw_arguments, known_urls=None):
    """Validate the arguments of a tool call and return the clean mapping.

    ``raw_arguments`` is what the model produced: normally a JSON object string,
    but a mapping is accepted too because providers differ here. Anything that is
    not a JSON object with usable fields raises SaveError, so bad input never
    reaches the renderer or the file system.
    """
    if isinstance(raw_arguments, (bytes, bytearray)):
        try:
            raw_arguments = bytes(raw_arguments).decode("utf-8")
        except UnicodeDecodeError:
            raise SaveError("the tool arguments were not valid JSON") from None

    if isinstance(raw_arguments, str):
        text = raw_arguments.strip()
        if not text:
            raise SaveError("the tool call had no arguments")
        try:
            arguments = json.loads(text)
        except (ValueError, TypeError):
            raise SaveError("the tool arguments were not valid JSON") from None
    else:
        arguments = raw_arguments

    return normalize_arguments(arguments, known_urls)


def escape_link_text(text):
    """Escape the characters that would break a Markdown link's label."""
    return text.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


def escape_link_url(url):
    """Percent-encode the few characters that would break a Markdown link.

    The URL itself is already validated by normalize_source_url(); this only keeps
    the link target unambiguous inside ``[label](target)``.
    """
    for char, replacement in (
        (" ", "%20"),
        ("(", "%28"),
        (")", "%29"),
        ("<", "%3C"),
        (">", "%3E"),
        ('"', "%22"),
        ("`", "%60"),
    ):
        url = url.replace(char, replacement)
    return url


def render_markdown(arguments):
    """Return the Markdown text for an already validated argument mapping.

    The document has one section per field and a numbered source list. This is
    pure text work: nothing here touches the file system, so a caller can inspect
    the document before a single byte is written.
    """
    lines = [
        f"# {arguments['title']}",
        "",
        "## خلاصه",
        "",
        arguments["summary"],
        "",
        "## تحلیل",
        "",
        arguments["analysis"],
        "",
        "## منابع",
        "",
    ]

    sources = arguments["sources"]
    if sources:
        for index, source in enumerate(sources, start=1):
            label = escape_link_text(source["title"])
            lines.append(f"{index}. [{label}]({escape_link_url(source['url'])})")
    else:
        lines.append("منبعی ثبت نشده است.")

    lines.append("")
    return "\n".join(lines)


def utc_stamp():
    """Return the current UTC time as a compact, sortable label."""
    return datetime.now(timezone.utc).strftime(STAMP_FORMAT)


def sanitize_label(value, limit):
    """Return a conservative label: letters, digits and '-' only, "" if unusable."""
    if not isinstance(value, str):
        return ""
    kept = "".join(char for char in value if char.isalnum() or char == "-")
    return kept.strip("-")[:limit].strip("-")


def slugify(title):
    """Return a filesystem-safe, lower-cased slug for a title.

    Only letters and digits survive (Unicode included, so Persian titles keep
    their words) and every other run of characters becomes a single '-'. Path
    separators, dots and reserved characters can therefore never appear.
    """
    if not isinstance(title, str):
        return ""
    kept = "".join(char if char.isalnum() else " " for char in title.casefold())
    return "-".join(kept.split())[:MAX_SLUG_LENGTH].strip("-")


def make_filename(title, stamp=None, token=None):
    """Return a safe, practically unique Markdown file name for a title.

    The name is ``<utc-timestamp>-<title-slug>-<random-token>.md``. The stamp and
    the token make two saves of the same title collision-proof without consulting
    the file system, and the slug can only contain letters, digits and '-', so no
    separator, dot or drive letter can ever appear. ``stamp`` and ``token`` are
    injectable so a caller (or a test) can make the name fixed.
    """
    label = "" if stamp is None else sanitize_label(stamp, MAX_LABEL_LENGTH)
    if not label:
        label = utc_stamp()

    slug = slugify(title) or SLUG_FALLBACK

    suffix = sanitize_label(token, MAX_TOKEN_LENGTH) if token is not None else ""
    if not suffix:
        suffix = secrets.token_hex(TOKEN_BYTES)

    stem = f"{label}-{slug}-{suffix}"
    if stem.lower() in RESERVED_NAMES:
        stem = f"_{stem}"
    return stem + FILE_EXTENSION


def output_directory(directory=None):
    """Return the absolute output folder; the model can never influence it.

    Without an argument this is the fixed ``PROJECT_ROOT/search_results`` folder.
    ``directory`` exists so tests and other embedding code can redirect the
    output; the tool layer never passes anything here.
    """
    if directory is None:
        return OUTPUT_DIR
    return os.path.abspath(directory)


def validate_filename(filename):
    """Return ``filename`` when it is a plain Markdown file name, else raise.

    Only the name itself is judged -- no folder is consulted and nothing is read
    or written -- so a proposal can be checked before it is stored, and
    resolve_target() can reuse exactly the same rules before a path is built.
    """
    if not isinstance(filename, str):
        raise SaveError("the file name must be a string")

    name = filename
    if not name or name != name.strip():
        raise SaveError("the file name was empty or had surrounding whitespace")
    if name != os.path.basename(name) or name in (".", ".."):
        raise SaveError("the file name must not contain a path")
    if not name.lower().endswith(FILE_EXTENSION):
        raise SaveError(f"the file name must end with '{FILE_EXTENSION}'")
    if name.startswith("."):
        raise SaveError("the file name must not start with a dot")
    if any(char in name for char in '<>:"/\\|?*') or any(
        char < " " or char == "\x7f" for char in name
    ):
        raise SaveError("the file name contains an unsupported character")
    return name


def resolve_target(filename, directory=None):
    """Return the absolute path of ``filename`` inside the output folder.

    The name must be a plain, relative Markdown file name: no directory part, no
    ``..``, no drive, no leading dot, no reserved character, no surrounding
    whitespace. The resolved path is then checked to be a direct child of the
    output folder, so path traversal cannot escape it even if a check above ever
    misses something.
    """
    name = validate_filename(filename)

    folder = output_directory(directory)
    target = os.path.realpath(os.path.join(folder, name))
    if os.path.dirname(target) != os.path.realpath(folder):
        raise SaveError("the file name escapes the output folder")
    return target


def build_proposal(arguments, known_urls=None, stamp=None, token=None):
    """Turn tool arguments into a ready-to-save proposal. Never touches the disk.

    ``arguments`` is a mapping (or a raw JSON string) with ``title``, ``summary``,
    ``analysis`` and ``sources``. The result holds the cleaned fields, the
    Markdown ``content``, its UTF-8 size in ``bytes`` and the generated
    ``filename``. No folder is created and no file is read or written: this is
    only preparation, so a caller can inspect or reject the document first. Invalid
    arguments or a document larger than MAX_FILE_BYTES raise SaveError.
    """
    if isinstance(arguments, (str, bytes, bytearray)):
        clean = parse_arguments(arguments, known_urls)
    else:
        clean = normalize_arguments(arguments, known_urls)

    content = render_markdown(clean)
    payload = content.encode("utf-8")
    if len(payload) > MAX_FILE_BYTES:
        raise SaveError(
            f"the research file would be larger than {MAX_FILE_BYTES} bytes"
        )

    return {
        "title": clean["title"],
        "summary": clean["summary"],
        "analysis": clean["analysis"],
        "sources": clean["sources"],
        "filename": make_filename(clean["title"], stamp=stamp, token=token),
        "content": content,
        "bytes": len(payload),
    }


def copy_proposal(proposal):
    """Return a private copy of one build_proposal() result, or None.

    This is the single place that decides what a *valid* proposal is: the server
    stores a proposal only when this function accepts it, and write_proposal()
    refuses everything else. The structure checked is exactly the one
    build_proposal() returns -- every text field non-empty, a plain Markdown
    ``filename`` (validate_filename), ``sources`` a list of ``{"title", "url"}``
    mappings whose URL passes normalize_source_url(), and ``bytes`` a
    non-negative integer when it is present.

    The copy is rebuilt field by field, nested source mappings included, so it is
    completely independent of the caller's object: a later change to the original
    -- even to ``proposal["sources"][0]`` -- can never alter what is written.
    Anything that does not match the contract yields None instead of raising.
    """
    if not isinstance(proposal, dict):
        return None

    copied = {}
    for field in PROPOSAL_TEXT_FIELDS:
        value = proposal.get(field)
        if not isinstance(value, str) or not value:
            return None
        copied[field] = value

    try:
        validate_filename(copied["filename"])
    except SaveError:
        return None

    sources = proposal.get("sources")
    if not isinstance(sources, list):
        return None
    copied_sources = []
    for source in sources:
        if not isinstance(source, dict) or set(source) != {"title", "url"}:
            return None
        title = source.get("title")
        url = source.get("url")
        if not isinstance(title, str) or not title:
            return None
        if not isinstance(url, str) or normalize_source_url(url) != url:
            return None
        copied_sources.append({"title": title, "url": url})
    copied["sources"] = copied_sources

    size = proposal.get(PROPOSAL_SIZE_FIELD)
    if size is None:
        # build_proposal() always sets it; recomputing keeps the copy complete
        # when a caller only kept the text.
        size = len(copied["content"].encode("utf-8"))
    elif not isinstance(size, int) or isinstance(size, bool) or size < 0:
        return None
    copied[PROPOSAL_SIZE_FIELD] = size
    return copied


def _close_quietly(handle):
    """Close a file handle if there is one; cleanup must never raise."""
    if handle is None:
        return
    try:
        handle.close()
    except OSError:
        pass


def _remove_file(path):
    """Remove a partially written file; cleanup must never raise."""
    try:
        os.remove(path)
    except OSError:
        pass


def write_proposal(proposal, directory=None):
    """Write one proposal to the output folder. The only function that writes.

    ``proposal`` must be the mapping build_proposal() returned. The folder is
    created when missing and the file is created exclusively, so a name that
    already exists is never replaced. When the write fails after the file was
    created, the partial file is removed, so a broken save cannot leave a
    half-written document behind. Returns a small result mapping
    (``status``/``file``/``title``/``sources``/``bytes``) that contains no
    absolute path; failures raise SaveError with a path-free message.
    """
    if not isinstance(proposal, dict):
        raise SaveError("a proposal is required")

    # The whole structure is checked against the build_proposal() contract in one
    # place, copy_proposal(): not just filename and content, but title, summary,
    # analysis, the source mappings and the declared size. The private copy means
    # nothing can change under us between validation and the write.
    checked = copy_proposal(proposal)
    if checked is None:
        raise SaveError("the proposal must come from build_proposal()")

    filename = checked["filename"]
    content = checked["content"]

    payload = content.encode("utf-8")
    if len(payload) > MAX_FILE_BYTES:
        raise SaveError(f"the research file must be at most {MAX_FILE_BYTES} bytes")

    folder = output_directory(directory)
    try:
        os.makedirs(folder, exist_ok=True)
    except OSError:
        raise SaveError("the research folder could not be created") from None

    path = resolve_target(filename, directory)

    handle = None
    try:
        # "x" = create exclusively: an existing file is never overwritten.
        handle = open(path, "xb")
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    except FileExistsError:
        _close_quietly(handle)
        raise SaveError(f"a file named '{filename}' already exists") from None
    except OSError:
        _close_quietly(handle)
        _remove_file(path)
        raise SaveError("the research file could not be saved") from None
    _close_quietly(handle)

    sources = checked["sources"]
    return {
        "status": "saved",
        "file": filename,
        "title": single_line(checked["title"]),
        "sources": len(sources),
        "bytes": len(payload),
    }


def ack_content(result):
    """Return the small JSON acknowledgement sent back to the model.

    Only the ACK_FIELDS survive, so the file's path, its contents and every other
    internal value are never handed back into the conversation. The string is
    always valid JSON and never longer than MAX_ACK_LENGTH, whatever it is given.
    """
    payload = {}
    if isinstance(result, dict):
        for field in ACK_FIELDS:
            if field not in result:
                continue
            value = result[field]
            if field in ("sources", "bytes"):
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    continue
            else:
                value = single_line(value)[:MAX_ACK_FIELD_LENGTH]
            payload[field] = value

    text = json.dumps(payload, ensure_ascii=False)
    if len(text) <= MAX_ACK_LENGTH:
        return text

    minimal = {
        key: payload[key] for key in ("status", "error", "file") if key in payload
    }
    return json.dumps(minimal, ensure_ascii=False)


def error_content(message):
    """Return a short JSON error object the model can read. Paths excluded."""
    reason = single_line(message)[:MAX_ERROR_LENGTH] or "the save tool failed"
    return ack_content({"status": "error", "error": reason})
