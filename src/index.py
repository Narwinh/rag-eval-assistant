"""
Phase 2: Build a FAISS vector index over the chunked AWS Lambda Developer
Guide.

Two backends are supported (see src/rag.py for the full rationale):
  --backend local  (default): Ollama's nomic-embed-text, zero API cost.
  --backend hosted: Google Gemini's embedding API, used only for the
                     free-hosted-demo deployment where a local Ollama
                     server isn't available.

Run `python src/ingest.py` first to produce data/processed/chunks.json.
"""

import argparse
import json
import time
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

CHUNKS_PATH = Path("data/processed/chunks.json")
INDEX_DIR = Path("faiss_index")
EMBEDDING_MODEL = "nomic-embed-text"
HOSTED_INDEX_DIR = Path("faiss_index_hosted")
HOSTED_EMBEDDING_MODEL = "models/gemini-embedding-001"


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _get_embeddings(backend: str, embedding_model: str):
    if backend == "hosted":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model=embedding_model)
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=embedding_model)


def build_index(chunks: list[dict], embedding_model: str = EMBEDDING_MODEL, backend: str = "local") -> FAISS:
    docs = [
        Document(
            page_content=c["text"],
            metadata={
                "chunk_id": c["chunk_id"],
                "source_doc": c["source_doc"],
                "chapter": c["chapter"],
                "subsection": c["subsection"],
                "page_range": c["page_range"],
            },
        )
        for c in chunks
    ]
    embeddings = _get_embeddings(backend, embedding_model)
    print(f"Embedding {len(docs)} chunks with '{embedding_model}' (backend={backend})...")
    start = time.time()
    vectorstore = FAISS.from_documents(docs, embeddings)
    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s ({elapsed / len(docs):.2f}s/chunk)")
    return vectorstore


def main():
    parser = argparse.ArgumentParser(description="Build a FAISS index over chunked documents.")
    parser.add_argument("--chunks", type=Path, default=CHUNKS_PATH, help="Path to chunks.json.")
    parser.add_argument("--index-dir", type=Path, default=None, help="Directory to save the FAISS index.")
    parser.add_argument("--backend", choices=["local", "hosted"], default="local")
    parser.add_argument("--embedding-model", default=None)
    args = parser.parse_args()

    index_dir = args.index_dir or (HOSTED_INDEX_DIR if args.backend == "hosted" else INDEX_DIR)
    embedding_model = args.embedding_model or (
        HOSTED_EMBEDDING_MODEL if args.backend == "hosted" else EMBEDDING_MODEL
    )

    chunks = load_chunks(args.chunks)
    vectorstore = build_index(chunks, embedding_model=embedding_model, backend=args.backend)
    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    print(f"Saved FAISS index to {index_dir}/")


if __name__ == "__main__":
    main()
