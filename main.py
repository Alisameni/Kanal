"""A tiny chatbot backed by an OpenAI-compatible API.

Run it with:  python main.py

The chat logic is also reusable from other code (see server.py): call
chat(message, history) to get (reply, updated_history) without the terminal
loop. Configuration is read from a local .env file (see .env for the keys).

When the optional SEARCH_API_KEY variable is configured, the model may use the
web_search tool implemented in web_search.py; without it, the chatbot behaves
exactly as before and never tries to search.
"""

import os
import time

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

import web_search

load_dotenv()

# Required configuration keys, in the order we want to report them.
REQUIRED_ENV_VARS = ("API_KEY", "API_BASE_URL", "MODEL_NAME")

# The system prompt is added by build_prompt() on every request and is never
# taken from the caller's history, so a client cannot change it.
SYSTEM_PROMPT = "You are a helpful assistant. Answer in Persian."

# How many recent messages (besides the system message) are kept in memory
# and sent to the model. 10 messages is about 5 turns of conversation.
HISTORY_LIMIT = 10

EXIT_COMMANDS = {"exit", "quit"}

# Commands that reset the conversation. The system message is kept so the
# assistant's behaviour and language stay the same after clearing.
CLEAR_COMMANDS = {"clear"}

# Tool-calling budget. They only apply when SEARCH_API_KEY is configured (see
# web_search.py) and keep the extra work per message bounded: at most
# MAX_TOOL_ROUNDS rounds in which the model may ask for a tool, and at most
# MAX_TOOL_CALLS searches actually executed. There is no retry loop, and the
# loop always ends with a tool-free answer, so it cannot run forever.
#
# One turn therefore costs at most MAX_TOOL_ROUNDS + 1 model calls plus
# MAX_TOOL_CALLS searches, all inside the time budget below.
MAX_TOOL_ROUNDS = 2
MAX_TOOL_CALLS = 3

# One shared wall-clock budget for the whole tool-calling turn
# (get_reply_with_tools). The clock starts when that path is entered, so this
# covers *all* of the turn's model calls and searches together: every step only
# gets a slice of what is left, never a fresh full allowance.
#
# Why 30s: the browser aborts POST /chat after 45s (frontend/src/api/chat.ts,
# REQUEST_TIMEOUT_MS). A turn can need up to MAX_TOOL_ROUNDS + 1 model calls, so
# the budget is what keeps that worst case inside the 45s ceiling, leaving the
# remainder to the last in-flight request plus JSON and transport overhead. The
# plain path (no search key) is untouched: it keeps the client's own settings
# from build_client().
TOOL_DEADLINE_SECONDS = 30.0

# Held back from the tool rounds for the final, tool-free answer, which is what
# the user actually reads. The rounds stop while this much is still unspent.
TOOL_ANSWER_RESERVE_SECONDS = 8.0

# Longest a single model call inside the tool path may take. It is below the
# client default (build_client: 20s) so that one slow call cannot spend the whole
# turn, and so the first call still leaves room for a search and an answer.
TOOL_STEP_TIMEOUT_SECONDS = 12.0

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


def get_reply_with_tools(client, model_name, messages, now=None):
    """get_reply() plus a bounded web_search tool-calling loop.

    The model may ask for a tool in at most MAX_TOOL_ROUNDS rounds, and at most
    MAX_TOOL_CALLS searches are executed per message. Tool requests, their
    arguments and their raw results live only in the temporary list built here:
    they are never added to the history the caller keeps.

    The whole turn runs under one shared budget of TOOL_DEADLINE_SECONDS, counted
    from entry with ``now`` (a monotonic clock by default, injectable for tests).
    Before every model call and every search the time left is checked, and the
    step's own timeout is capped by what remains, so a slow provider cannot
    restart the budget on each round. A step that cannot be started any more is
    skipped; when the budget is gone altogether, ToolDeadlineExceeded is raised
    instead of returning a partial answer.

    This bounds when work *starts*; it cannot cancel a request that is already in
    flight (see bounded_client).

    A failed or disallowed tool call becomes a short JSON error object that the
    model still receives, so the loop always makes progress and ends with a
    tool-free answer. Returns the final reply text, or None when the model
    returns nothing at all.
    """
    clock = tool_clock(now)
    deadline = clock() + TOOL_DEADLINE_SECONDS

    conversation = list(messages)
    tools = web_search.build_tools()
    calls_left = MAX_TOOL_CALLS

    for _ in range(MAX_TOOL_ROUNDS):
        timeout = step_timeout(
            deadline, clock, TOOL_STEP_TIMEOUT_SECONDS, TOOL_ANSWER_RESERVE_SECONDS
        )
        if timeout is None:
            # Not enough time is left for another round: keep the rest for the
            # final, tool-free answer instead of starting work we cannot finish.
            break

        message = request_message(
            client, model_name, conversation, tools, timeout=timeout
        )
        if message is None:
            return None

        calls = collect_tool_calls(message)
        if not calls:
            return message.content

        conversation.append(build_assistant_tool_message(message, calls))
        for call_id, name, raw_arguments in calls:
            if calls_left <= 0:
                content = web_search.error_content(
                    "the search budget for this message is used up"
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
                else:
                    calls_left -= 1
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
            conversation.append(
                {"role": "tool", "tool_call_id": call_id, "content": content}
            )

    # The tool budget (rounds or time) is spent, so ask once more without tools:
    # the request then ends with a plain answer instead of another tool call.
    timeout = step_timeout(deadline, clock, TOOL_STEP_TIMEOUT_SECONDS)
    if timeout is None:
        raise ToolDeadlineExceeded(
            "the web search turn ran out of time before an answer could be made"
        )

    message = request_message(client, model_name, conversation, None, timeout=timeout)
    if message is None:
        return None
    return message.content


def chat(message, history=None, client=None, model_name=None):
    """Send one message to the model and return (reply, updated_history).

    This is the reusable entry point (used by server.py). The terminal loop in
    main() uses it too, passing the client it already built.

    The returned history holds only "user" and "assistant" messages, never the
    system message and never any internal tool-calling message. When the model
    replies with nothing, or when the request fails, the history is returned
    unchanged so a failed turn is never stored as a successful exchange.

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
    if isinstance(error, APIConnectionError):
        # This also covers APITimeoutError, which the SDK derives from it, so a
        # slow or unreachable provider keeps the message it already had.
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
