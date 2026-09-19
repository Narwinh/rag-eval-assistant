"""
Phase 4/5: Run the eval set through the RAG pipeline and score it.

Usage:
    python -m src.eval.run_eval --config-name baseline
    python -m src.eval.run_eval --config-name k6 --k 6
    python -m src.eval.run_eval --config-name chunk300 --index-dir faiss_index_300
    python -m src.eval.run_eval --config-name rerank --rerank

Writes a results JSON with per-question scores and aggregate averages
(overall and broken down by difficulty).
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for `src.rag` import

from src.rag import RagPipeline, INDEX_DIR, EMBEDDING_MODEL, GENERATION_MODEL, DEFAULT_K
from src.eval.metrics import retrieval_precision_recall, answer_relevance, citation_accuracy, JUDGE_MODEL

EVAL_SET_PATH = Path("data/eval/eval_set.json")


def run_eval(pipeline: RagPipeline, eval_set: list[dict], k: int) -> dict:
    per_question = []

    for item in eval_set:
        t0 = time.time()
        result = pipeline.ask(item["question"], k=k)
        elapsed = time.time() - t0

        retrieved_sources = [c["source_doc"] for c in result.retrieved_chunks]
        retrieval_scores = retrieval_precision_recall(retrieved_sources, item["source_doc"])

        relevance = answer_relevance(item["question"], item["ideal_answer"], result.answer)

        cited_texts = [
            c["text"] for c in result.retrieved_chunks if c["chunk_id"] in result.cited_chunk_ids
        ]
        citation = citation_accuracy(result.answer, cited_texts)

        per_question.append({
            "id": item["id"],
            "question": item["question"],
            "difficulty": item["difficulty"],
            "gold_source_doc": item["source_doc"],
            "generated_answer": result.answer,
            "ideal_answer": item["ideal_answer"],
            "retrieved_sources": retrieved_sources,
            "cited_chunk_ids": result.cited_chunk_ids,
            "retrieval": retrieval_scores,
            "answer_relevance": relevance,
            "citation_accuracy": citation,
            "latency_seconds": round(elapsed, 2),
        })
        print(
            f"  {item['id']} [{item['difficulty']:6s}] "
            f"relevance={relevance['score']} "
            f"recall@k={retrieval_scores['recall_at_k']:.2f} "
            f"({elapsed:.1f}s)"
        )

    return aggregate(per_question)


def _avg(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def aggregate(per_question: list[dict]) -> dict:
    overall = {
        "n_questions": len(per_question),
        "avg_precision_at_k": _avg([q["retrieval"]["precision_at_k"] for q in per_question]),
        "avg_recall_at_k": _avg([q["retrieval"]["recall_at_k"] for q in per_question]),
        "avg_answer_relevance": _avg([q["answer_relevance"]["score"] for q in per_question]),
        "avg_citation_accuracy": _avg([q["citation_accuracy"]["accuracy"] for q in per_question]),
        "avg_latency_seconds": _avg([q["latency_seconds"] for q in per_question]),
    }

    by_difficulty = {}
    for diff in ("easy", "medium", "hard"):
        subset = [q for q in per_question if q["difficulty"] == diff]
        if not subset:
            continue
        by_difficulty[diff] = {
            "n_questions": len(subset),
            "avg_precision_at_k": _avg([q["retrieval"]["precision_at_k"] for q in subset]),
            "avg_recall_at_k": _avg([q["retrieval"]["recall_at_k"] for q in subset]),
            "avg_answer_relevance": _avg([q["answer_relevance"]["score"] for q in subset]),
            "avg_citation_accuracy": _avg([q["citation_accuracy"]["accuracy"] for q in subset]),
        }

    return {"overall": overall, "by_difficulty": by_difficulty, "per_question": per_question}


def main():
    parser = argparse.ArgumentParser(description="Run the eval set through the RAG pipeline.")
    parser.add_argument("--config-name", default="baseline", help="Label for this run, used in the output filename and stored config.")
    parser.add_argument("--eval-set", type=Path, default=EVAL_SET_PATH)
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    parser.add_argument("--generation-model", default=GENERATION_MODEL)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--rerank", action="store_true", help="Enable the BM25 re-ranking step.")
    parser.add_argument("--output", type=Path, default=None, help="Output path (default: results/<config-name>_results.json)")
    args = parser.parse_args()

    with open(args.eval_set, encoding="utf-8") as f:
        eval_set = json.load(f)

    print(f"Loaded {len(eval_set)} eval questions from {args.eval_set}")
    print(
        f"Config '{args.config_name}': index_dir={args.index_dir}, k={args.k}, "
        f"rerank={args.rerank}, generation_model={args.generation_model}, "
        f"embedding_model={args.embedding_model}, judge_model={JUDGE_MODEL}"
    )

    pipeline = RagPipeline(
        index_dir=args.index_dir,
        embedding_model=args.embedding_model,
        generation_model=args.generation_model,
        k=args.k,
        rerank=args.rerank,
    )

    scored = run_eval(pipeline, eval_set, k=args.k)
    scored["config"] = {
        "config_name": args.config_name,
        "index_dir": str(args.index_dir),
        "k": args.k,
        "rerank": args.rerank,
        "embedding_model": args.embedding_model,
        "generation_model": args.generation_model,
        "judge_model": JUDGE_MODEL,
    }

    output_path = args.output or Path("results") / f"{args.config_name}_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scored, f, indent=2, ensure_ascii=False)

    print(f"\n--- {args.config_name} aggregate ---")
    for k_, v_ in scored["overall"].items():
        print(f"  {k_}: {v_}")
    print(f"Wrote results to {output_path}")


if __name__ == "__main__":
    main()
