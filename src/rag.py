"""
Phase 2: Retrieve + generate pipeline.

Given a question:
  1. Embed it with the same local Ollama embedding model used to build the index.
  2. Retrieve the top-k most similar chunks from FAISS.
  3. Ask a local Ollama chat model to answer using ONLY those chunks, citing
     which chunk(s) it used.

Usage (from the command line):
    python src/rag.py "What is the default timeout for a Lambda function?"
"""

import os
import sys
import re
from pathlib import Path
from dataclasses import dataclass, field

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

# --- Backend selection --------------------------------------------------
#
# "local" (default): fully self-hosted via Ollama - zero API cost, used for
#   all development, the eval harness, and `docker run` locally.
# "hosted": free-tier hosted APIs (Google Gemini, for both generation and
#   embeddings) instead of Ollama, because the free hosting platforms this
#   project's live demo runs on (e.g. Render's free web service, 512MB RAM)
#   don't have enough memory/CPU to run a local LLM server. Same RAG logic,
#   same eval harness - only the embedding/generation backend differs.
#   Requires a GOOGLE_API_KEY env var (free key from https://aistudio.google.com).
#
# Set via the RAG_BACKEND env var; the two backends use SEPARATE FAISS
# indices (faiss_index/ vs faiss_index_hosted/) since embedding spaces from
# different models are not interchangeable.
BACKEND = os.environ.get("RAG_BACKEND", "local")


def _defaults_for_backend(backend: str) -> tuple[Path, str, str]:
    """(index_dir, embedding_model, generation_model) defaults for a given backend.

    Used both for this module's top-level constants (below, for the current
    RAG_BACKEND) and inside RagPipeline.__init__ (so passing backend="hosted"
    explicitly - e.g. from run_eval.py --backend hosted - gets hosted
    defaults even if RAG_BACKEND itself is still "local")."""
    if backend == "hosted":
        return Path("faiss_index_hosted"), "models/gemini-embedding-001", "gemini-3.5-flash-lite"
    return Path("faiss_index"), "nomic-embed-text", "llama3.2:3b"


INDEX_DIR, EMBEDDING_MODEL, GENERATION_MODEL = _defaults_for_backend(BACKEND)
DEFAULT_K = 3  # winning config from the Phase 5 ablation sweep - see README results table
RERANK_FETCH_MULTIPLIER = 4  # when reranking, fetch this many x k candidates before rescoring

# Overridable so the same code works against a local `ollama serve` and
# against an Ollama instance reachable from inside a Docker container.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

PROMPT_TEMPLATE = """You are a technical assistant answering questions about AWS Lambda, \
using ONLY the numbered source excerpts below. Do not use outside knowledge.

If, and only if, NONE of the excerpts contain relevant information, say exactly "I don't have \
enough information in the provided sources to answer that." and write "Citations: none" - do not \
cite any sources in that case, since none of them helped.

Otherwise, answer the question and then, on a new line, list ONLY the sources you actually used, \
like:
Citations: [1], [3]

Sources:
{context}

Question: {question}

Answer:"""


@dataclass
class RagResult:
    question: str
    answer: str
    cited_chunk_ids: list[str] = field(default_factory=list)
    retrieved_chunks: list[dict] = field(default_factory=list)
    confidence: float = 0.0


def _format_context(docs: list[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        blocks.append(
            f"[{i}] (source: {d.metadata['source_doc']}, chunk_id: {d.metadata['chunk_id']})\n"
            f"{d.page_content}"
        )
    return "\n\n".join(blocks)


def _parse_citations(answer: str, docs: list[Document]) -> list[str]:
    """Map bracketed numbers like [1], [3] in the answer back to chunk_ids."""
    numbers = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    cited = []
    for i, d in enumerate(docs, start=1):
        if i in numbers:
            cited.append(d.metadata["chunk_id"])
    return cited


class RagPipeline:
    def __init__(
        self,
        index_dir: Path | None = None,
        embedding_model: str | None = None,
        generation_model: str | None = None,
        k: int = DEFAULT_K,
        rerank: bool = False,
        backend: str = BACKEND,
    ):
        # Resolve backend-specific defaults from the `backend` actually passed
        # in here, NOT from the module-level constants above (which reflect
        # RAG_BACKEND at import time) - otherwise RagPipeline(backend="hosted")
        # would silently keep using local (Ollama) index/model defaults.
        default_index_dir, default_embedding_model, default_generation_model = _defaults_for_backend(backend)
        index_dir = index_dir or default_index_dir
        embedding_model = embedding_model or default_embedding_model
        generation_model = generation_model or default_generation_model

        if backend == "hosted":
            # Imported lazily so `langchain-google-genai` is only required
            # when actually running in hosted mode.
            from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

            self.embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
            self.llm = ChatGoogleGenerativeAI(model=generation_model, temperature=0)
        else:
            self.embeddings = OllamaEmbeddings(model=embedding_model, base_url=OLLAMA_BASE_URL)
            self.llm = ChatOllama(model=generation_model, temperature=0, base_url=OLLAMA_BASE_URL)

        self.vectorstore = FAISS.load_local(
            str(index_dir), self.embeddings, allow_dangerous_deserialization=True
        )
        self.k = k
        self.rerank = rerank

    def retrieve(self, question: str, k: int | None = None) -> list[Document]:
        k = k or self.k
        if not self.rerank:
            return self.vectorstore.similarity_search(question, k=k)

        # Simple lexical re-ranker: over-fetch by embedding similarity, then
        # re-score with BM25 (term overlap) and keep the top k. This tends to
        # help on queries with specific technical terms/numbers that
        # embedding similarity alone can under-weight.
        candidates = self.vectorstore.similarity_search(question, k=k * RERANK_FETCH_MULTIPLIER)
        tokenized_corpus = [doc.page_content.lower().split() for doc in candidates]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(question.lower().split())
        ranked = [doc for _, doc in sorted(zip(scores, candidates), key=lambda p: p[0], reverse=True)]
        return ranked[:k]

    def ask(self, question: str, k: int | None = None) -> RagResult:
        k = k or self.k
        docs = self.retrieve(question, k=k)

        # Confidence proxy: how similar the single best-matching chunk is to
        # the question, mapped from FAISS's raw L2 distance (lower = closer)
        # into a 0-1 range via 1 / (1 + distance). This is a heuristic, not a
        # calibrated probability - it is meant to flag "nothing relevant was
        # found" cases (low score) rather than to be precise.
        top_hits = self.vectorstore.similarity_search_with_score(question, k=1)
        confidence = round(1 / (1 + top_hits[0][1]), 3) if top_hits else 0.0

        context = _format_context(docs)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        response = self.llm.invoke(prompt)
        answer_text = response.content

        cited_ids = _parse_citations(answer_text, docs)
        retrieved = [
            {
                "chunk_id": d.metadata["chunk_id"],
                "source_doc": d.metadata["source_doc"],
                "text": d.page_content,
            }
            for d in docs
        ]
        return RagResult(
            question=question,
            answer=answer_text,
            cited_chunk_ids=cited_ids,
            retrieved_chunks=retrieved,
            confidence=confidence,
        )


def main():
    if len(sys.argv) < 2:
        print('Usage: python src/rag.py "your question here"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    pipeline = RagPipeline()
    result = pipeline.ask(question)

    print(f"\nQuestion: {result.question}\n")
    print(f"Answer:\n{result.answer}\n")
    print("Retrieved chunks:")
    for c in result.retrieved_chunks:
        marker = "*" if c["chunk_id"] in result.cited_chunk_ids else " "
        print(f"  [{marker}] {c['chunk_id']} - {c['source_doc']}")


if __name__ == "__main__":
    main()
