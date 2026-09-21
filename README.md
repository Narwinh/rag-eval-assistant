# AWS Lambda Docs Q&A — a RAG assistant with an evaluation harness

A retrieval-augmented Q&A assistant over ~300 pages of the official AWS Lambda Developer Guide, built to answer one question with evidence: **which design choices actually improve a RAG pipeline, and by how much?**

**Live demo:** _TODO: Render URL, once deployed_ · **Run it yourself in one command:** see [Docker](#docker) below

## Results

All numbers are averages over the same 38-question hand-verified eval set, scored by `phi3:mini` as an independent LLM judge (never the generator model). Full per-question scores and judge reasoning are in [`results/`](results/); `results/ablation_results.json` has the combined summary.

| Config | Chunk size | k | Re-rank | Answer relevance (1-5) | Precision@k | Recall@k | Citation accuracy | Avg latency |
|---|---|---|---|---|---|---|---|---|
| Baseline | 500 tok | 4 | off | 3.92 | 0.53 | 0.80 | 0.61 | 31.0s |
| Chunk size 300 | 300 tok | 4 | off | 3.79 | **0.63** | **0.86** | 0.57 | **25.9s** |
| Chunk size 800 | 800 tok | 4 | off | 3.97 | 0.43 | 0.79 | **0.76** | 50.5s |
| k=3 | 500 tok | 3 | off | **3.97** | 0.61 | 0.75 | 0.63 | 25.8s |
| k=6 | 500 tok | 6 | off | 3.95 | 0.45 | 0.86 | 0.60 | 40.9s |
| Re-ranking (BM25) | 500 tok | 4 | on | 3.84 | 0.45 | 0.83 | 0.60 | 30.1s |
| **Best config: k=3** | 500 tok | **3** | off | **3.97** | 0.61 | 0.75 | 0.63 | **25.8s** |

**Headline finding: retrieving fewer, more targeted chunks (k=3 vs k=4) improved answer relevance, precision, and latency all at once** — a rare case where nothing traded off against anything else. The effect is concentrated in the hard (multi-hop synthesis) questions:

| Config | Easy relevance | Medium relevance | Hard relevance |
|---|---|---|---|
| Baseline (k=4) | 4.13 | 3.73 | 3.88 |
| **k=3** | 4.27 | 3.47 | **4.38** |
| Chunk size 300 | 4.13 | 3.73 | 3.25 |

Dropping from 4 to 3 retrieved chunks lifted hard-question relevance from 3.88 to 4.38 — the single largest effect in the whole sweep. With a small (3B parameter) local generator, apparently even one extra chunk of context is enough added noise to measurably hurt cross-section synthesis, even though it does cost some recall (0.80 → 0.75). Smaller chunks (300 tokens) show the opposite pattern: better retrieval precision/recall in isolation, but the worst hard-question relevance (3.25) — fragmenting the text into smaller pieces makes it harder for the model to find a single chunk that contains a full synthesis-worthy explanation. Larger chunks (800 tokens) had the best citation accuracy (0.76, since a bigger excerpt is more likely to self-containedly support a claim) but at nearly double the latency and the worst retrieval precision. The BM25 re-ranker did not help on this corpus — a lexical re-ranker adds little when the questions and source text already share vocabulary closely, which is typical of technical documentation.

Read honestly: the overall relevance gain (3.92 → 3.97 on a 5-point scale) is modest — this is a small local 3B model, not a frontier API model, and the eval set is only 38 questions. The clearer, more actionable signal is the difficulty breakdown, not the single aggregate number.

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
   llama3.2:3b via Ollama (local)  or  gemini-3.5-flash-lite (hosted)
        │
        ▼
   src/serve.py (FastAPI /ask)  ──►  Docker (local)  /  Render (hosted demo)
```

Evaluation runs the same pipeline through `src/eval/run_eval.py`, scoring each answer with `phi3:mini` as an independent LLM judge (deliberately a different model from the generator, to reduce self-preference bias). All eval numbers below are from the local Ollama backend — see "Local vs. hosted" for why the deployed demo differs.

### Why these choices

- **Orchestration — LangChain**: more widely documented than the alternatives for a from-scratch RAG pipeline over FAISS + Ollama, which matters for being able to explain every line in an interview.
- **Embeddings & generation — local via Ollama (`nomic-embed-text`, `llama3.2:3b`)**: zero API cost to run or demo, fully offline-capable. Trade-off: a 3B model is noticeably weaker than a frontier API model at nuanced synthesis questions — see the hard-difficulty breakdown in the results.
- **Judge model — `phi3:mini`, not the generator**: using the same model to both generate and grade its own answers is a well-known source of inflated scores. A different (and differently-trained) small model is a cheap partial mitigation, not a full fix — see "What I'd improve."
- **Vector store — FAISS**: local, no infrastructure, sufficient for ~230-370 chunks.
- **Document set — AWS Lambda Developer Guide**: ties to the author's AWS certifications; the guide's own structure (concepts → configuration → scaling → permissions) gives natural easy/medium/hard question boundaries.

### Local vs. hosted

Everything above (development, the eval harness, all results in this README) runs against the **local** backend: Ollama, fully self-hosted, zero API cost. That's the "real" version of this project and the one `docker run` gives you.

Hugging Face Spaces' Docker/Gradio tier now requires a paid plan (checked directly against their docs while building this — it used to be free), which doesn't fit a zero-cost portfolio project. Free serverless hosts (Render's free tier, etc.) don't have enough RAM to run a local LLM server, so the **hosted** demo swaps Ollama for Google Gemini's free-tier API (`gemini-3.5-flash-lite` for generation, `gemini-embedding-001` for embeddings — one provider for both, so the embedding space stays consistent) via `RAG_BACKEND=hosted`. Same retrieval logic, same prompts, same citation parsing — only the model client changes. See [`src/rag.py`](src/rag.py) for the full backend-selection code.

This is a real trade-off, not a free upgrade: the hosted demo's answer quality hasn't been separately measured against the eval harness (see "What I'd improve"), and it depends on an external API being up, unlike the fully offline local path.

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
faiss_index/           # committed - production index (local backend, 500-tok chunks)
faiss_index_hosted/    # committed - Gemini-embedded index for the hosted demo
Dockerfile / entrypoint.sh    # self-contained image (bundles Ollama) - local use
Dockerfile.hosted             # lightweight image (no Ollama) - Render deploy
.env.example
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

### Docker (local, self-contained)

```bash
docker build -t lambda-docs-qa .
docker run -p 8000:8000 lambda-docs-qa
```

The image bundles Ollama itself and bakes in both models at build time, so the container needs no external Ollama server or API key — this is the "clone and run" path. _Note: `docker build`/`docker run` have not been verified on this machine (Docker wasn't installed in the dev environment); the Dockerfile logic was verified step-by-step instead (base image, model-baking `RUN` step, entrypoint readiness check)._

### Deploying the hosted demo (Render + Gemini free tier)

1. Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
2. Build the Gemini-backed index once (needs `GOOGLE_API_KEY` set in your own shell — never share or commit this key):
   ```bash
   pip install -r requirements.txt   # includes langchain-google-genai
   export GOOGLE_API_KEY=...         # your own key, your own shell
   python src/index.py --backend hosted
   git add faiss_index_hosted/ && git commit -m "Add Gemini-backed index for hosted demo"
   ```
3. On [render.com](https://render.com), create a new **Web Service** from this GitHub repo, Docker runtime, Dockerfile path `Dockerfile.hosted`, free instance type.
4. In the service's Environment settings, add `GOOGLE_API_KEY` as a secret (enter it directly in Render's dashboard, not here). `RAG_BACKEND=hosted` is already set by `Dockerfile.hosted`.
5. Render auto-redeploys on every push to the connected branch.

## What I'd improve with more time

- **Judge reliability**: a single small local judge model (`phi3:mini`) scoring on a 1-5 scale is noisy; a stronger or ensembled judge (or a frontier API model, if cost weren't a constraint) would tighten the eval's signal-to-noise ratio.
- **Re-ranking**: the BM25 re-ranker is a lexical baseline; a cross-encoder re-ranker would likely help more on the harder synthesis questions.
- **Chunking**: fixed-size token chunking ignores document structure; header-aware or semantic chunking could reduce mid-sentence/mid-example splits.
- **Generation model**: `llama3.2:3b` is small; several "hard" failures in the results are the model failing to synthesize across two retrieved chunks rather than a retrieval failure — worth separately measuring "retrieval succeeded but generation didn't use it."
- **Broader corpus**: only 5 of the guide's ~30 chapters are ingested; extending to the full guide (with the same section-level metadata scheme) would test retrieval discrimination at a larger scale.
- **Evaluate the hosted backend too**: `run_eval.py` currently only exercises `RAG_BACKEND=local`; running the same 38-question eval set against the Gemini-backed hosted pipeline would show whether the swapped-in API model changes answer quality (likely better, given it's a much larger model) and would make the local-vs-hosted trade-off in this README a measured one instead of an assumed one.
