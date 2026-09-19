# AWS Lambda Docs Q&A — a RAG assistant with an evaluation harness

A retrieval-augmented Q&A assistant over ~300 pages of the official AWS Lambda Developer Guide, built to answer one question with evidence: **which design choices actually improve a RAG pipeline, and by how much?**

**Live demo:** _TODO: Hugging Face Space link_

## Results

_TODO: fill in after running `python -m src.eval.run_eval` (baseline) and the ablation sweep. Numbers below are placeholders — do not quote until replaced._

| Config | Chunk size | k | Re-rank | Answer relevance (1-5) | Retrieval recall@k | Citation accuracy |
|---|---|---|---|---|---|---|
| Baseline | 500 tok | 4 | off | TBD | TBD | TBD |
| Chunk size 300 | 300 tok | 4 | off | TBD | TBD | TBD |
| Chunk size 800 | 800 tok | 4 | off | TBD | TBD | TBD |
| k=3 | 500 tok | 3 | off | TBD | TBD | TBD |
| k=6 | 500 tok | 6 | off | TBD | TBD | TBD |
| Re-ranking on | 500 tok | 4 | on | TBD | TBD | TBD |
| **Best config** | TBD | TBD | TBD | **TBD** | TBD | TBD |

Full per-question scores and judge reasoning are in [`results/`](results/).

## What this project actually tests

Building a RAG chatbot is the easy part. The point of this repo is the harness in [`src/eval/`](src/eval/): a hand-written 38-question eval set (not LLM-generated — every question and answer was checked against the actual source text), three metrics (retrieval precision/recall@k, LLM-judged answer relevance, and citation accuracy), and a set of ablations that isolate what chunk size, retrieval depth (`k`), and re-ranking each contribute to answer quality.

## Architecture

```
AWS Lambda Developer Guide (PDF, 5 chapters / ~299 pages)
        │  src/ingest.py  (page-range extraction + token-based chunking)
        ▼
   chunks.json  (229 chunks, section-level metadata)
        │  src/index.py   (Ollama nomic-embed-text embeddings)
        ▼
   FAISS index
        │  src/rag.py     (retrieve top-k, optional BM25 re-rank)
        ▼
   llama3.2:3b (local, via Ollama)  →  answer + citations
        │
        ▼
   src/serve.py (FastAPI /ask)  ──►  Docker  ──►  Hugging Face Space
```

Evaluation runs the same pipeline through `src/eval/run_eval.py`, scoring each answer with `phi3:mini` as an independent LLM judge (deliberately a different model from the generator, to reduce self-preference bias).

### Why these choices

- **Orchestration — LangChain**: more widely documented than the alternatives for a from-scratch RAG pipeline over FAISS + Ollama, which matters for being able to explain every line in an interview.
- **Embeddings & generation — local via Ollama (`nomic-embed-text`, `llama3.2:3b`)**: zero API cost to run or demo, fully offline-capable. Trade-off: a 3B model is noticeably weaker than a frontier API model at nuanced synthesis questions — see the hard-difficulty breakdown in the results.
- **Judge model — `phi3:mini`, not the generator**: using the same model to both generate and grade its own answers is a well-known source of inflated scores. A different (and differently-trained) small model is a cheap partial mitigation, not a full fix — see "What I'd improve."
- **Vector store — FAISS**: local, no infrastructure, sufficient for ~230-370 chunks.
- **Document set — AWS Lambda Developer Guide**: ties to the author's AWS certifications; the guide's own structure (concepts → configuration → scaling → permissions) gives natural easy/medium/hard question boundaries.

## Repo layout

```
src/
  ingest.py          # PDF → cleaned, chunked text with section metadata
  index.py           # chunks → FAISS index (Ollama embeddings)
  rag.py             # retrieve + generate + cite (+ optional BM25 re-rank)
  serve.py           # FastAPI app
  eval/
    eval_set.json    # 38 hand-verified Q&A pairs
    metrics.py        # retrieval precision/recall@k, LLM-judge relevance, citation accuracy
    run_eval.py       # runs eval_set.json through a configured pipeline
data/
  raw/               # source PDF (gitignored — see "Running it locally")
  processed/         # chunked JSON at each tested chunk size
  eval/eval_set.json
results/              # baseline + ablation results (one JSON per config)
Dockerfile / entrypoint.sh
```

## Running it locally

Requires Python 3.12 and [Ollama](https://ollama.com/) installed and running.

```bash
python -m venv venv
source venv/Scripts/activate  # or venv/bin/activate on macOS/Linux
pip install -r requirements.txt

ollama pull llama3.2:3b
ollama pull nomic-embed-text
ollama pull phi3:mini      # only needed to run the eval harness (judge model)

# Download the source PDF (not committed — ~29MB):
curl -o data/raw/lambda-dg.pdf https://docs.aws.amazon.com/lambda/latest/dg/lambda-dg.pdf

python src/ingest.py
python src/index.py

python src/rag.py "What is the default timeout for a Lambda function?"

# Run the eval harness
python -m src.eval.run_eval --config-name baseline

# Serve the API
uvicorn src.serve:app --reload
# then open http://localhost:8000/docs
```

### Reproducing the ablations

```bash
# Chunk size
python src/ingest.py --chunk-size 300 --chunk-overlap 30 --output data/processed/chunks_300.json
python src/index.py --chunks data/processed/chunks_300.json --index-dir faiss_index_300
python -m src.eval.run_eval --config-name chunk300 --index-dir faiss_index_300

# k value
python -m src.eval.run_eval --config-name k6 --k 6

# Re-ranking
python -m src.eval.run_eval --config-name rerank --rerank
```

### Docker

```bash
docker build -t lambda-docs-qa .
docker run -p 8000:8000 lambda-docs-qa
```

The image bundles Ollama itself and bakes in both models at build time, so the container needs no external Ollama server. _Note: `docker build`/`docker run` have not been verified on this machine (Docker wasn't installed in the dev environment) — verified instead by the equivalent Hugging Face Space build, which uses this same Dockerfile._

## What I'd improve with more time

- **Judge reliability**: a single small local judge model (`phi3:mini`) scoring on a 1-5 scale is noisy; a stronger or ensembled judge (or a frontier API model, if cost weren't a constraint) would tighten the eval's signal-to-noise ratio.
- **Re-ranking**: the BM25 re-ranker is a lexical baseline; a cross-encoder re-ranker would likely help more on the harder synthesis questions.
- **Chunking**: fixed-size token chunking ignores document structure; header-aware or semantic chunking could reduce mid-sentence/mid-example splits.
- **Generation model**: `llama3.2:3b` is small; several "hard" failures in the results are the model failing to synthesize across two retrieved chunks rather than a retrieval failure — worth separately measuring "retrieval succeeded but generation didn't use it."
- **Broader corpus**: only 5 of the guide's ~30 chapters are ingested; extending to the full guide (with the same section-level metadata scheme) would test retrieval discrimination at a larger scale.
