"""
Phase 6: Serve the RAG pipeline behind a small FastAPI app.

Run locally:
    uvicorn src.serve:app --reload

Then either use the auto-generated docs at http://localhost:8000/docs,
or POST directly:
    curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
         -d '{"question": "What is the default timeout for a Lambda function?"}'
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.rag import RagPipeline

pipeline: RagPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    # Loaded once at startup: the FAISS index + local Ollama model handles
    # are reused across requests rather than reopened per-call.
    pipeline = RagPipeline()
    yield


app = FastAPI(
    title="AWS Lambda Docs Q&A",
    description="RAG assistant over the AWS Lambda Developer Guide, with citations.",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["What is the default timeout for a Lambda function?"])
    k: int = Field(4, ge=1, le=10, description="Number of chunks to retrieve.")


class Citation(BaseModel):
    chunk_id: str
    source_doc: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    result = pipeline.ask(request.question, k=request.k)
    citations = [
        Citation(chunk_id=c["chunk_id"], source_doc=c["source_doc"])
        for c in result.retrieved_chunks
        if c["chunk_id"] in result.cited_chunk_ids
    ]
    return AskResponse(answer=result.answer, citations=citations, confidence=result.confidence)
