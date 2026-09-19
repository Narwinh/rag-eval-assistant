"""
Phase 2: Build a FAISS vector index over the chunked AWS Lambda Developer
Guide, using a local Ollama embedding model (zero API cost).

Run `python src/ingest.py` first to produce data/processed/chunks.json.
"""

import argparse
import json
import time
from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

CHUNKS_PATH = Path("data/processed/chunks.json")
INDEX_DIR = Path("faiss_index")
EMBEDDING_MODEL = "nomic-embed-text"


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_index(chunks: list[dict], embedding_model: str = EMBEDDING_MODEL) -> FAISS:
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
    embeddings = OllamaEmbeddings(model=embedding_model)
    print(f"Embedding {len(docs)} chunks with '{embedding_model}' (local, via Ollama)...")
    start = time.time()
    vectorstore = FAISS.from_documents(docs, embeddings)
    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s ({elapsed / len(docs):.2f}s/chunk)")
    return vectorstore


def main():
    parser = argparse.ArgumentParser(description="Build a FAISS index over chunked documents.")
    parser.add_argument("--chunks", type=Path, default=CHUNKS_PATH, help="Path to chunks.json.")
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR, help="Directory to save the FAISS index.")
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    vectorstore = build_index(chunks)
    args.index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(args.index_dir))
    print(f"Saved FAISS index to {args.index_dir}/")


if __name__ == "__main__":
    main()
