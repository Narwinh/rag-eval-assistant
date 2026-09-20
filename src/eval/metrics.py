"""
Phase 4: Evaluation metrics for the RAG pipeline.

Three metrics, all computed per-question and then averaged:

1. Retrieval precision/recall@k - did the retrieved chunks include the
   gold source_doc(s) for the question?
2. Answer relevance (LLM-as-judge) - a local judge model scores the
   generated answer against the question + ideal_answer on a 1-5 scale,
   with logged reasoning.
3. Citation accuracy - for each chunk the pipeline actually cited, ask the
   judge whether that chunk's text supports the generated answer.

The judge model is intentionally DIFFERENT from the generation model
(phi3:mini vs llama3.2:3b) to reduce self-preference bias: a model judging
its own answers tends to rate them more favorably than an independent judge
would. Both run locally via Ollama at zero API cost.
"""

import os
import re

from langchain_ollama import ChatOllama

JUDGE_MODEL = "phi3:mini"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

_judge_llm = None


def get_judge_llm():
    global _judge_llm
    if _judge_llm is None:
        _judge_llm = ChatOllama(model=JUDGE_MODEL, temperature=0, base_url=OLLAMA_BASE_URL)
    return _judge_llm


# --- 1. Retrieval precision/recall@k -----------------------------------

def parse_gold_sources(source_doc_field: str) -> list[str]:
    """Hard questions may list multiple required sources joined by ' ; '."""
    return [s.strip() for s in source_doc_field.split(";") if s.strip()]


def retrieval_precision_recall(retrieved_source_docs: list[str], gold_source_doc_field: str) -> dict:
    gold_sources = parse_gold_sources(gold_source_doc_field)
    k = len(retrieved_source_docs)

    relevant_retrieved = sum(1 for s in retrieved_source_docs if s in gold_sources)
    gold_found = sum(1 for g in gold_sources if g in retrieved_source_docs)

    precision = relevant_retrieved / k if k else 0.0
    recall = gold_found / len(gold_sources) if gold_sources else 0.0

    return {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "gold_sources": gold_sources,
        "retrieved_sources": retrieved_source_docs,
    }


# --- 2. Answer relevance (LLM-as-judge) ---------------------------------

RELEVANCE_PROMPT = """You are grading an AI assistant's answer to a factual question about AWS Lambda.

Question: {question}

Ideal (reference) answer: {ideal_answer}

Assistant's answer: {generated_answer}

Score the assistant's answer on a 1-5 scale for how well it matches the factual content of the \
ideal answer:
5 = fully correct and complete, matches the ideal answer's facts
4 = mostly correct, minor omission or imprecision
3 = partially correct, missing an important detail or slightly inaccurate
2 = mostly incorrect or very incomplete, but touches the right topic
1 = wrong, off-topic, or "I don't have enough information" when the ideal answer shows the info was available

Respond in EXACTLY this format:
SCORE: <a single integer 1-5>
REASONING: <one or two sentences explaining the score>"""


def _parse_score(judge_text: str) -> tuple[int | None, str]:
    score_match = re.search(r"SCORE:\s*(\d)", judge_text)
    reasoning_match = re.search(r"REASONING:\s*(.+)", judge_text, re.DOTALL)
    score = int(score_match.group(1)) if score_match else None
    reasoning = reasoning_match.group(1).strip() if reasoning_match else judge_text.strip()
    return score, reasoning


def answer_relevance(question: str, ideal_answer: str, generated_answer: str) -> dict:
    prompt = RELEVANCE_PROMPT.format(
        question=question, ideal_answer=ideal_answer, generated_answer=generated_answer
    )
    response = get_judge_llm().invoke(prompt)
    score, reasoning = _parse_score(response.content)
    if score is None:
        # Fallback: couldn't parse a score, treat as lowest score but keep raw text for inspection.
        score = 1
        reasoning = f"[unparsed judge output] {response.content.strip()}"
    return {"score": score, "reasoning": reasoning, "raw_judge_output": response.content}


# --- 3. Citation accuracy -----------------------------------------------

CITATION_PROMPT = """An AI assistant answered a question about AWS Lambda, and cited the source excerpt \
below as one of (possibly several) sources for its answer. The answer may combine facts from multiple \
sources, so the excerpt does not need to support the ENTIRE answer by itself - it only needs to \
provide relevant supporting evidence for AT LEAST PART of the answer's factual content.

Assistant's answer: {answer}

Cited source excerpt:
\"\"\"
{source_text}
\"\"\"

Does this excerpt provide real, relevant supporting evidence for at least part of the answer? Answer \
NO only if the excerpt is irrelevant, contradicts the answer, or the answer explicitly claims no \
information was available (in which case an excerpt containing real information does NOT support \
that "no information" claim). Respond in EXACTLY this format:
VERDICT: <YES or NO>
REASONING: <one sentence>"""


def citation_supported(generated_answer: str, source_text: str) -> dict:
    prompt = CITATION_PROMPT.format(answer=generated_answer, source_text=source_text)
    response = get_judge_llm().invoke(prompt)
    text = response.content
    verdict_match = re.search(r"VERDICT:\s*(YES|NO)", text, re.IGNORECASE)
    reasoning_match = re.search(r"REASONING:\s*(.+)", text, re.DOTALL)
    supported = bool(verdict_match) and verdict_match.group(1).upper() == "YES"
    reasoning = reasoning_match.group(1).strip() if reasoning_match else text.strip()
    return {"supported": supported, "reasoning": reasoning}


def citation_accuracy(generated_answer: str, cited_chunk_texts: list[str]) -> dict:
    if not cited_chunk_texts:
        return {"accuracy": None, "n_citations": 0, "per_citation": []}

    results = [citation_supported(generated_answer, text) for text in cited_chunk_texts]
    n_supported = sum(1 for r in results if r["supported"])
    return {
        "accuracy": n_supported / len(results),
        "n_citations": len(results),
        "per_citation": results,
    }
