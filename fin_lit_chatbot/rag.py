from __future__ import annotations

import re
from typing import Any

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langsmith import traceable

from fin_lit_chatbot.config import Settings
from fin_lit_chatbot.rag_settings import RagSettings


class RagService:
    def __init__(self, settings: Settings) -> None:
        self.rag_settings = RagSettings()
        embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            default_headers=settings.openrouter_headers(),
        )

        self._investment_vs = Chroma(
            collection_name=self.rag_settings.investment_collection,
            persist_directory=self.rag_settings.chroma_dir,
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": self.rag_settings.similarity_space},
        )
        self._money_vs = Chroma(
            collection_name=self.rag_settings.money_management_collection,
            persist_directory=self.rag_settings.chroma_dir,
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": self.rag_settings.similarity_space},
        )

    @traceable(name="retrieve_docs", run_type="retriever")
    def retrieve_docs(self, query: str, topic: str, k: int | None = None) -> list[dict[str, Any]]:
        store = self._investment_vs if topic == "investment_education" else self._money_vs
        top_k = k or self.rag_settings.top_k
        candidate_k = max(top_k, top_k * self.rag_settings.candidate_multiplier)
        docs = store.similarity_search_with_score(query, k=candidate_k)

        query_tokens = self._tokenize(query)
        reranked: list[tuple[Any, float, float, float]] = []
        for doc, distance in docs:
            title = str(doc.metadata.get("title", ""))
            title_overlap = self._title_overlap(query_tokens, title)
            # Chroma returns distance-like score for cosine: lower is better.
            retrieval_score = float(distance) - (self.rag_settings.title_boost_weight * title_overlap)
            reranked.append((doc, float(distance), title_overlap, retrieval_score))

        reranked.sort(key=lambda x: x[3])
        selected = reranked[:top_k]

        return [
            {
                # LangSmith-friendly document shape for retriever traces
                "type": "Document",
                "page_content": doc.page_content,
                "metadata": {
                    "title": doc.metadata.get("title", "untitled"),
                    "source": doc.metadata.get("source", "unknown"),
                    "topic": doc.metadata.get("topic", topic),
                    "similarity_score": float(distance),
                    "title_overlap": float(title_overlap),
                    "retrieval_score": float(retrieval_score),
                },
                # Existing app fields consumed by engine
                "title": doc.metadata.get("title", "untitled"),
                "source": doc.metadata.get("source", "unknown"),
                "topic": doc.metadata.get("topic", topic),
                "chunk_text": doc.page_content,
                "similarity_score": float(distance),
                "title_overlap": float(title_overlap),
                "retrieval_score": float(retrieval_score),
            }
            for doc, distance, title_overlap, retrieval_score in selected
        ]

    def _tokenize(self, text: str) -> set[str]:
        return {t for t in re.findall(r"[a-zA-Z0-9]+", text.lower()) if len(t) > 2}

    def _title_overlap(self, query_tokens: set[str], title: str) -> float:
        if not query_tokens:
            return 0.0
        title_tokens = self._tokenize(title)
        if not title_tokens:
            return 0.0
        overlap = query_tokens.intersection(title_tokens)
        return len(overlap) / len(query_tokens)
