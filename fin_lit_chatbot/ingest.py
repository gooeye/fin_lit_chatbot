from __future__ import annotations

import argparse
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from fin_lit_chatbot.config import Settings
from fin_lit_chatbot.rag_settings import RagSettings


def _derive_title_from_filename(path: Path) -> str:
    stem = path.stem
    slug = stem.split("_", 1)[-1] if "_" in stem else stem
    return " ".join(part for part in slug.replace("-", " ").split() if part).strip()


def _derive_source_from_text(text: str, fallback: str) -> str:
    first_line = text.splitlines()[0].strip() if text.splitlines() else ""
    if first_line.startswith("#"):
        source = first_line[1:].strip()
        return source or fallback
    return fallback


def _strip_header_line(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("#"):
        return "\n".join(lines[1:]).strip()
    return text.strip()


def _build_store(topic: str, settings: Settings, rag_settings: RagSettings) -> Chroma:
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        default_headers=settings.openrouter_headers(),
    )

    collection_name = (
        rag_settings.investment_collection
        if topic == "investment_education"
        else rag_settings.money_management_collection
    )

    return Chroma(
        collection_name=collection_name,
        persist_directory=rag_settings.chroma_dir,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": rag_settings.similarity_space},
    )


def ingest_txt_folder(folder: Path, topic: str) -> int:
    settings = Settings()
    rag_settings = RagSettings()
    store = _build_store(topic, settings, rag_settings)

    txt_files = sorted([p for p in folder.rglob("*.txt") if p.is_file()])
    if not txt_files:
        return 0

    docs: list[Document] = []
    ids: list[str] = []

    for path in txt_files:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue

        rel = path.relative_to(folder).as_posix()
        doc_id = f"{topic}:{rel}"
        title = _derive_title_from_filename(path)
        source = _derive_source_from_text(text, fallback=path.name)
        body = _strip_header_line(text)
        if not body:
            continue

        docs.append(
            Document(
                page_content=body,
                metadata={
                    "source": source,
                    "title": title,
                    "topic": topic,
                },
            )
        )
        ids.append(doc_id)

    if not docs:
        return 0

    store.add_documents(documents=docs, ids=ids)
    return len(docs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a folder of .txt files into Chroma.")
    parser.add_argument("--folder", required=True, help="Folder containing .txt files")
    parser.add_argument(
        "--topic",
        required=True,
        choices=["investment_education", "money_management"],
        help="Target topic/collection",
    )

    args = parser.parse_args()
    folder = Path(args.folder)
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"Invalid folder: {folder}")

    inserted = ingest_txt_folder(folder=folder, topic=args.topic)
    print(f"Ingested {inserted} file(s) into topic '{args.topic}'.")


if __name__ == "__main__":
    main()
