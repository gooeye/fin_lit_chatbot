from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str | None = os.getenv("OPENROUTER_API_KEY")
    openrouter_base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    openrouter_referer: str = os.getenv("OPENROUTER_REFERER", "http://localhost:8501")
    openrouter_title: str = os.getenv("OPENROUTER_TITLE", "FinLit Chatbot")
    telegram_bot_token: str | None = os.getenv("TELEGRAM_BOT_TOKEN")

    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    fast_model: str = os.getenv("FAST_MODEL", "openai/gpt-4.1-mini")
    smart_model: str = os.getenv("SMART_MODEL", "openai/gpt-4.1")

    def openrouter_headers(self) -> dict[str, str]:
        return {
            "HTTP-Referer": self.openrouter_referer,
            "X-OpenRouter-Title": self.openrouter_title,
        }
