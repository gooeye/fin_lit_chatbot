from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RagSettings:
    chroma_dir: str = os.getenv("RAG_CHROMA_DIR", ".chroma")
    similarity_space: str = os.getenv("RAG_SIMILARITY_SPACE", "cosine")
    top_k: int = int(os.getenv("RAG_TOP_K", "4"))
    candidate_multiplier: int = int(os.getenv("RAG_CANDIDATE_MULTIPLIER", "4"))
    title_boost_weight: float = float(os.getenv("RAG_TITLE_BOOST_WEIGHT", "0.35"))

    investment_collection: str = os.getenv("RAG_COLLECTION_INVESTMENT", "investment_education")
    money_management_collection: str = os.getenv("RAG_COLLECTION_MONEY", "money_management")
