"""A FastAPI server that exposes the chatbot from main.py over HTTP.

Run it with:  uvicorn server:app --reload

The chat logic itself lives in main.py, so the terminal chatbot and this API
always behave the same way. The .env values and the API key never leave the
server: they are only read inside main.get_config().
"""

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    RateLimitError,
)
from pydantic import BaseModel, Field, field_validator

import main

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


@app.post("/chat")
def read_chat(request: ChatRequest):
    """Send one message to the model and return the reply and new history.

    Errors are translated to plain HTTP status codes. Only short, generic
    messages are returned, so the API key, the .env values and tracebacks are
    never exposed. The returned history never contains the system message,
    because the server adds it itself on every request.
    """
    history = [turn.model_dump() for turn in request.history]

    try:
        reply, updated_history = main.chat(request.message, history)
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

    return {"reply": reply, "history": updated_history}

