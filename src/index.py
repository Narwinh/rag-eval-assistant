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
from tenacity import retry, stop_after_attempt, wait_exponential

CHUNKS_PATH = Path("data/processed/chunks.json")
INDEX_DIR = Path("faiss_index")
EMBEDDING_MODEL = "nomic-embed-text"
HOSTED_INDEX_DIR = Path("faiss_index_hosted")
HOSTED_EMBEDDING_MODEL = "models/gemini-embedding-001"

# The Gemini free tier's embedding quota is much tighter than its generation
# quota (undocumented exact numbers - not exposed without an AI Studio
# login), so hosted-mode embedding is done in small batches with pauses and
# retries, rather than one big FAISS.from_documents() call.
HOSTED_BATCH_SIZE = 5
HOSTED_PAUSE_SECONDS = 3.0


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _get_embeddings(backend: str, embedding_model: str):
    if backend == "hosted":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model=embedding_model)
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=embedding_model)


@retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=2, min=5, max=60), reraise=True)
def _embed_batch_with_retry(embeddings, batch: list[str]) -> list[list[float]]:
    return embeddings.embed_documents(batch)


def _build_hosted_index(docs: list[Document], embeddings) -> FAISS:
    """Embed in small batches with pauses/retries to respect the Gemini free
    tier's (undocumented, apparently tight) embedding rate limit."""
    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]
    vectors: list[list[float]] = []

    for i in range(0, len(texts), HOSTED_BATCH_SIZE):
        batch = texts[i : i + HOSTED_BATCH_SIZE]
        vectors.extend(_embed_batch_with_retry(embeddings, batch))
        done = min(i + HOSTED_BATCH_SIZE, len(texts))
        print(f"  embedded {done}/{len(texts)} chunks")
        if done < len(texts):
            time.sleep(HOSTED_PAUSE_SECONDS)

    return FAISS.from_embeddings(list(zip(texts, vectors)), embeddings, metadatas=metadatas)


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
    if backend == "hosted":
        vectorstore = _build_hosted_index(docs, embeddings)
    else:
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
