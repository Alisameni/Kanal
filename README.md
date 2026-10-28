# Kanal — Personal AI Chatbot

Kanal is a Persian-first personal AI chatbot with a command-line interface and a local web interface. It uses an OpenAI-compatible API provider and supports optional web search through Tavily.

## Features

- Command-line chatbot
- Persian, right-to-left web interface
- FastAPI backend
- React, TypeScript, and Vite frontend
- Conversation history support
- Optional web search using Tavily
- Automated backend tests

## Requirements

- Python 3.11 or newer
- Node.js and npm
- An API key from an OpenAI-compatible provider

A Tavily API key is optional and required only for web search.

## Installation

### 1. Clone the repository

```powershell
git clone https://github.com/Alisameni/Kanal.git
cd Kanal
```

### 2. Set up the Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks script execution, allow activation scripts for the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Configure environment variables

Create your local `.env` file from the example:

```powershell
Copy-Item .env.example .env
```

Set these variables in `.env`:

| Variable | Description |
|---|---|
| `API_KEY` | Your provider API key |
| `API_BASE_URL` | Your provider's OpenAI-compatible API base URL |
| `MODEL_NAME` | The model identifier supported by your provider |
| `SEARCH_API_KEY` | Optional Tavily API key for web search |

Never commit your real `.env` file or publish API keys.

### 4. Install frontend dependencies

From the repository root:

```powershell
npm install
```

## Running Kanal

### Command-line interface

With the virtual environment activated:

```powershell
python main.py
```

### Web interface

Start the backend in the first terminal:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn server:app --reload --host 127.0.0.1 --port 8000
```

Start the frontend in a second terminal, from the repository root:

```powershell
npm run dev
```

Open the local address printed by Vite, usually `http://localhost:5173`.

The frontend uses Vite's development proxy to communicate with the local backend.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Basic API status |
| `GET` | `/health` | Health check |
| `POST` | `/chat` | Send a chat request |

The backend runs locally at `http://127.0.0.1:8000` with the commands above.

## Tests and Quality Checks

Run the Python test suite:

```powershell
python -m unittest discover -v
```

Check frontend linting:

```powershell
npm run lint
```

Build the frontend for production:

```powershell
npm run build
```

## Security Notes

- Keep `.env` private; it contains credentials.
- `.env.example` should contain placeholders only.
- `SEARCH_API_KEY` is optional. Web search requires a valid Tavily API key.
- The development server is configured to bind to the local machine.

## License

No license has been specified yet. Check the repository for licensing terms before redistributing the project.
