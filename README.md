# FinLit Chatbot (UV)

Minimal financial literacy chatbot built with:

- LangGraph orchestration
- ChromaDB RAG with cosine similarity
- OpenRouter for embeddings + chat models (fast + smart)
- LangSmith observability
- Streamlit frontend

## 1) Setup

1. Copy env template:

	- `.env.example` -> `.env`

2. Fill in keys in `.env`:

	- `OPENROUTER_API_KEY`
	- `LANGSMITH_API_KEY`
  - `TELEGRAM_BOT_TOKEN` (only if using Telegram bot)

3. Install dependencies (already UV-managed):

```bash
uv sync
```

## 2) Run app

```bash
uv run streamlit run app.py
```

## 2a) Run Telegram bot

The Telegram bot supports clickable inline follow-up buttons after each answer.

```bash
uv run finlit-telegram
```

## 2b) Run with Docker

Build and run directly:

```bash
docker build -t finlit-chatbot .
docker run --rm -p 8501:8501 --env-file .env finlit-chatbot
```

Or with Compose (persists Chroma locally):

```bash
docker compose up --build
```

Run ingestion inside Docker (with host ./data mounted to /app/data):

```bash
docker compose run --rm finlit finlit-ingest --folder /app/data/investment_txt --topic investment_education
docker compose run --rm finlit finlit-ingest --folder /app/data/money_txt --topic money_management
```

## 3) Ingest .txt files into RAG

Each `.txt` file is ingested as one chunk/document.

```bash
uv run finlit-ingest --folder data/investment_txt --topic investment_education
uv run finlit-ingest --folder data/money_txt --topic money_management
```

## Notes

- Chroma is persisted locally on disk at `RAG_CHROMA_DIR` (default `.chroma/`).
- No financial knowledge is hardcoded in code. Seed own `.txt/.pdf` corpus into Chroma.
- RAG parameters are configured via `.env` (`RAG_*` variables).
- Risk quiz remains deterministic in code.
- Two knowledge routes are implemented:
  - `investment_education`
  - `money_management`
- Structured deterministic workflows are included for:
  - risk quiz
  - budget summary
  - debt example
  - insurance checklist
