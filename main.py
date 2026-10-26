"""A tiny chatbot backed by an OpenAI-compatible API.

Run it with:  python main.py

Configuration is read from a local .env file (see .env for the keys).
"""

import os

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

load_dotenv()

# Required configuration keys, in the order we want to report them.
REQUIRED_ENV_VARS = ("API_KEY", "API_BASE_URL", "MODEL_NAME")

# The system message is always kept at history[0] and never trimmed.
SYSTEM_PROMPT = "You are a helpful assistant. Answer in Persian."

# How many recent messages (besides the system message) are kept in memory
# and sent to the model. 10 messages is about 5 turns of conversation.
HISTORY_LIMIT = 10

EXIT_COMMANDS = {"exit", "quit"}

# Commands that reset the conversation. The system message is kept so the
# assistant's behaviour and language stay the same after clearing.
CLEAR_COMMANDS = {"clear"}


def load_config():
    """Read and validate the environment variables we need.

    Returns an (api_key, base_url, model_name) tuple, or prints a clear
    error and returns None when something is missing.
    """
    values = {name: os.getenv(name) for name in REQUIRED_ENV_VARS}
    missing = [name for name in REQUIRED_ENV_VARS if not values[name]]

    if missing:
        print("Missing required environment variable(s): " + ", ".join(missing))
        print("Please add them to your .env file and try again.")
        return None

    return values["API_KEY"], values["API_BASE_URL"], values["MODEL_NAME"]


def trim_history(history):
    """Keep the system message plus the most recent HISTORY_LIMIT messages."""
    system_message, *rest = history
    return [system_message] + rest[-HISTORY_LIMIT:]


def build_client(api_key, base_url):
    """Create a configured OpenAI client."""
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        max_retries=2,
        timeout=30,
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


def describe_error(error):
    """Turn an exception into a short, friendly message."""
    if isinstance(error, AuthenticationError):
        return "Authentication failed. Check your API key."
    if isinstance(error, RateLimitError):
        return "Rate limit exceeded. Check your provider limits."
    if isinstance(error, APIConnectionError):
        return "Connection failed. Check your internet or API endpoint."
    if isinstance(error, APIStatusError):
        return f"API returned HTTP {error.status_code}."
    return f"Unexpected error ({type(error).__name__}): {error}"


def main():
    config = load_config()
    if config is None:
        return

    api_key, base_url, model_name = config
    client = build_client(api_key, base_url)

    # Conversation memory. The system message always stays at index 0.
    history = [{"role": "system", "content": SYSTEM_PROMPT}]

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
                # Reset the conversation but keep the system message.
                history = [{"role": "system", "content": SYSTEM_PROMPT}]
                print("Conversation history cleared.")
                continue

            if not user_message:
                continue

            # Build the request without touching stored history yet, so a
            # failed request never leaves a dangling user message in memory.
            user_turn = {"role": "user", "content": user_message}
            prompt = trim_history(history + [user_turn])

            try:
                bot_message = get_reply(client, model_name, prompt)
            except Exception as error:
                # describe_error() maps every exception to one short, friendly
                # message, so the user never sees a raw traceback.
                print(describe_error(error))
                continue

            if not bot_message:
                print("Bot: [empty response]")
                continue

            print(f"Bot: {bot_message}")

            # Commit the successful turn to memory and keep it bounded.
            history = trim_history(
                prompt + [{"role": "assistant", "content": bot_message}]
            )
    except KeyboardInterrupt:
        print("\nChatbot stopped.")


if __name__ == "__main__":
    main()
