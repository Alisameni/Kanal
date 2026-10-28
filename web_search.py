"""Optional web-search tool for the chatbot, backed by the Tavily Search API.

The chatbot only offers this tool when a search key is configured. The key is
read from the ``SEARCH_API_KEY`` environment variable *at call time* (never at
import time, never hard-coded), so the chatbot keeps working exactly as before
when the variable is missing.

Rules enforced here:

- One HTTP POST to the official Tavily endpoint, with a fixed timeout.
- At most ``MAX_RESULTS`` results, bounded title/URL/snippet lengths and a
  bounded total payload, so one tool call cannot flood the model context.
- Every failure (invalid input, network error, timeout, HTTP error, malformed
  body) becomes a small JSON error object for the model, so the tool loop always
  has something safe to send back.
- The API key only travels in the Authorization header and never appears in an
  error message, a tool payload or a log line.
- No eval/exec/subprocess: nothing the model produces is executed as code.

The module uses only the standard library, so no new dependency is required.
"""

import json
import os
import urllib.error
import urllib.request

# Official Tavily Search endpoint.
TAVILY_SEARCH_URL = "https://api.tavily.com/search"

# Name of the environment variable that enables the tool. Only the *name* is
# referenced here; the value is read at call time and never printed or logged.
SEARCH_API_KEY_ENV = "SEARCH_API_KEY"

# The one tool name the model is allowed to ask for.
TOOL_NAME = "web_search"

# Budgets for a single search. They are deliberately small: one tool call must
# not be able to spend a lot of time or model context.
REQUEST_TIMEOUT_SECONDS = 8

# A search is only worth starting while at least this much time is left; below it
# the caller's deadline has effectively passed, so no socket is opened at all.
MIN_REQUEST_SECONDS = 0.5
MAX_RESPONSE_BYTES = 200_000
MAX_QUERY_LENGTH = 300
MAX_RESULTS = 5
MAX_TITLE_LENGTH = 200
MAX_URL_LENGTH = 500
MAX_SNIPPET_LENGTH = 500
MAX_OUTPUT_LENGTH = 4000


class SearchError(Exception):
    """A search could not be performed. Messages never contain the API key."""


def get_api_key():
    """Return the configured search key, or None when it is missing/blank.

    The value is only ever handed to the HTTP layer. It is never returned to
    the caller of this module, printed, logged or put into an error message.
    """
    return (os.getenv(SEARCH_API_KEY_ENV) or "").strip() or None


def is_available():
    """True when SEARCH_API_KEY is configured, so the tool may be offered."""
    return get_api_key() is not None


def build_tool_spec():
    """Return the function-tool schema the model sees for web_search."""
    return {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": (
                "Search the public web and return a few short results. Use it "
                "when the answer depends on recent, current or verifiable facts; "
                "do not use it for small talk or general knowledge."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query, in the user's language.",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }


def build_tools():
    """Return the tool list passed to the Chat Completions API."""
    return [build_tool_spec()]


def bounded_timeout(timeout):
    """Return the HTTP timeout to use for one search request.

    ``timeout`` is the part of the caller's remaining time budget that this
    search may use, so a single search can never stretch past the turn's
    deadline. None means "use the module default" and any value is capped at
    REQUEST_TIMEOUT_SECONDS. A value below MIN_REQUEST_SECONDS raises, so a search
    with no usable time left never reaches the network.
    """
    if timeout is None:
        return REQUEST_TIMEOUT_SECONDS

    try:
        value = float(timeout)
    except (TypeError, ValueError):
        return REQUEST_TIMEOUT_SECONDS

    if value < MIN_REQUEST_SECONDS:
        raise SearchError("there was not enough time left for a search")
    return min(value, REQUEST_TIMEOUT_SECONDS)


def normalize_query(value):
    """Validate a query and return it with collapsed whitespace.

    Empty, non-text and over-long queries are rejected, so a malformed tool call
    never reaches the network.
    """
    if not isinstance(value, str):
        raise SearchError("the 'query' argument is required and must be a string")

    query = " ".join(value.split())
    if not query:
        raise SearchError("the search query was empty")
    if len(query) > MAX_QUERY_LENGTH:
        raise SearchError(
            f"the search query must be at most {MAX_QUERY_LENGTH} characters"
        )
    return query


def parse_arguments(raw_arguments):
    """Validate the arguments of a tool call and return ``{"query": str}``.

    ``raw_arguments`` is what the model produced: normally a JSON object string,
    but a mapping is accepted too because providers differ here. Anything that
    is not a JSON object with a usable ``query`` raises SearchError.
    """
    if isinstance(raw_arguments, (bytes, bytearray)):
        try:
            raw_arguments = bytes(raw_arguments).decode("utf-8")
        except UnicodeDecodeError:
            raise SearchError("the tool arguments were not valid JSON") from None

    if isinstance(raw_arguments, str):
        text = raw_arguments.strip()
        if not text:
            raise SearchError("the tool call had no arguments")
        try:
            arguments = json.loads(text)
        except (ValueError, TypeError):
            raise SearchError("the tool arguments were not valid JSON") from None
    else:
        arguments = raw_arguments

    if not isinstance(arguments, dict):
        raise SearchError("the tool arguments must be a JSON object")

    if "query" not in arguments:
        raise SearchError("the 'query' argument is required and must be a string")

    return {"query": normalize_query(arguments["query"])}


def clean_text(value, limit):
    """Return a single-line, length-bounded string. Non-text values become ""."""
    if value is None:
        return ""
    if not isinstance(value, str):
        if isinstance(value, (int, float)):
            value = str(value)
        else:
            return ""

    text = " ".join(value.split())
    if len(text) > limit:
        text = text[: max(limit - 1, 0)].rstrip() + "…"
    return text


def normalize_results(payload):
    """Turn a Tavily response into the small shape the model is given.

    Only ``title``, ``url`` and a bounded ``snippet`` survive; entries without a
    usable http(s) URL are skipped and no more than MAX_RESULTS are kept.
    """
    if not isinstance(payload, dict):
        raise SearchError("the search service returned an unexpected response")

    raw_results = payload.get("results")
    if raw_results is None:
        raw_results = []
    if not isinstance(raw_results, list):
        raise SearchError("the search service returned an unexpected response")

    results = []
    for item in raw_results:
        if len(results) >= MAX_RESULTS:
            break
        if not isinstance(item, dict):
            continue

        url = clean_text(item.get("url"), MAX_URL_LENGTH)
        if not url.startswith(("http://", "https://")):
            continue

        results.append(
            {
                "title": clean_text(item.get("title"), MAX_TITLE_LENGTH) or url,
                "url": url,
                "snippet": clean_text(item.get("content"), MAX_SNIPPET_LENGTH),
            }
        )
    return results


def to_tool_content(payload):
    """Serialize a tool payload for the model, bounded to MAX_OUTPUT_LENGTH.

    Results are dropped from the end until the JSON fits, so the model always
    receives a small, parseable object.
    """
    text = json.dumps(payload, ensure_ascii=False)
    if len(text) <= MAX_OUTPUT_LENGTH:
        return text

    results = payload.get("results")
    if not isinstance(results, list):
        # Error payloads are tiny; this is a last-resort bound.
        return json.dumps({"error": "the tool response was too large"})

    for keep in range(len(results) - 1, -1, -1):
        candidate = dict(payload, results=results[:keep], truncated=True)
        text = json.dumps(candidate, ensure_ascii=False)
        if len(text) <= MAX_OUTPUT_LENGTH:
            return text

    return json.dumps(
        {
            "query": clean_text(payload.get("query"), MAX_QUERY_LENGTH),
            "results": [],
            "truncated": True,
        },
        ensure_ascii=False,
    )


def error_content(message):
    """Return a small JSON error object the model can read. Secrets excluded."""
    reason = clean_text(message, MAX_SNIPPET_LENGTH) or "tool error"
    return to_tool_content({"error": reason})


def post_json(url, payload, api_key, timeout=REQUEST_TIMEOUT_SECONDS):
    """POST a JSON body with a bearer token and return the decoded JSON.

    Kept as its own function so tests can replace it and never open a socket.
    The key is only placed in the Authorization header: never in the body, an
    error message or a log line.

    Note: the bearer header is the documented Tavily scheme. It has not been
    verified against the live service yet (no live call is made in Phase 1); if
    Tavily rejects it, the fix is to also send ``api_key`` in the body here.
    """
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES)
    except urllib.error.HTTPError as error:
        raise SearchError(f"the search service returned HTTP {error.code}") from None
    except urllib.error.URLError as error:
        if isinstance(error.reason, TimeoutError):
            raise SearchError("the search request timed out") from None
        raise SearchError("could not reach the search service") from None
    except TimeoutError:
        raise SearchError("the search request timed out") from None
    except OSError:
        raise SearchError("could not reach the search service") from None

    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise SearchError(
            "the search service returned an unreadable response"
        ) from None
    return decoded


def search(query, timeout=None):
    """Run one Tavily search and return ``{"query": str, "results": [...]}``.

    ``timeout`` is the part of the caller's time budget this search may spend; it
    is capped at REQUEST_TIMEOUT_SECONDS by bounded_timeout(). Raises SearchError
    when the tool is disabled, the query is invalid, no usable time is left, or
    the request/response cannot be used.
    """
    api_key = get_api_key()
    if api_key is None:
        raise SearchError("web search is not configured")

    request_timeout = bounded_timeout(timeout)
    clean_query = normalize_query(query)
    payload = {
        "query": clean_query,
        "search_depth": "basic",
        "max_results": MAX_RESULTS,
        "include_answer": False,
        "include_raw_content": False,
        "include_images": False,
    }

    body = post_json(TAVILY_SEARCH_URL, payload, api_key, timeout=request_timeout)
    return {"query": clean_query, "results": normalize_results(body)}


def execute_tool(name, raw_arguments, timeout=None):
    """Run one tool call and return the JSON string sent back to the model.

    Only ``web_search`` is ever executed. An unknown tool name, a malformed
    argument payload, a disabled tool, a request with no time left and any search
    failure all become a small JSON error object, so the caller always has a safe
    tool message to append and the tool loop can never get stuck. ``timeout`` is
    passed through to search().
    """
    if name != TOOL_NAME:
        return error_content(f"unknown tool '{clean_text(name, 64)}'")

    try:
        arguments = parse_arguments(raw_arguments)
        return to_tool_content(search(arguments["query"], timeout=timeout))
    except SearchError as error:
        return error_content(str(error))
