import time
import os
from typing import (Optional, List,)

import pandas as pd

from rag_core.retrieval import RetrievalAndGenerationService


def summarise_passages(passages: List[dict]) -> dict:
    num_chunks = len(passages)

    qdrant_scores = [float(p.get("qdrant_score", 0.0)) for p in passages]
    confidences = [float(p.get("confidence", 0.0)) for p in passages]
    sources = [str(p.get("source", "unknown")) for p in passages]
    buckets = [str(p.get("conf_bucket", "")) for p in passages]
    doc_ids = [str(p.get("doc_id", "")) for p in passages]

    chunks_text = "\n\n-----\n\n".join(p.get("text", "") for p in passages)

    def safe_mean(xs):
        return float(sum(xs) / len(xs)) if xs else 0.0

    return {
        "num_chunks_used": num_chunks,
        "max_qdrant_score": max(qdrant_scores) if qdrant_scores else 0.0,
        "mean_qdrant_score": safe_mean(qdrant_scores),
        "max_confidence_passage": max(confidences) if confidences else 0.0,
        "mean_confidence_passage": safe_mean(confidences),
        "chunk_sources": " || ".join(sources),
        "chunk_buckets": " || ".join(buckets),
        "chunk_doc_ids": " || ".join(doc_ids),
        "chunks_text": chunks_text,
    }


def evaluate_csv(
    in_csv: str,
    question_col: str = "question",
    out_csv: Optional[str] = None,
):
    if out_csv is None:
        out_csv = in_csv

    if not os.path.exists(in_csv):
        raise FileNotFoundError(f"Input CSV not found: {in_csv}")

    df = pd.read_csv(in_csv)
    if question_col not in df.columns:
        raise ValueError(f"Column '{question_col}' not found in CSV.")

    service = RetrievalAndGenerationService()

    response_times = []
    overall_confidences = []
    answers = []

    num_chunks_used = []
    max_qdrant_scores = []
    mean_qdrant_scores = []
    max_conf_pass = []
    mean_conf_pass = []
    chunk_sources = []
    chunk_buckets = []
    chunk_doc_ids = []
    chunks_texts = []

    for q in df[question_col].astype(str).tolist():
        t0 = time.perf_counter()
        result = service.answer_with_confidence(q)
        dt = time.perf_counter() - t0

        response_times.append(dt)
        answers.append(result.get("answer", ""))
        overall_confidences.append(float(result.get("overall_confidence", 0.0)))

        passages = result.get("passages", []) or []
        summary = summarise_passages(passages)

        num_chunks_used.append(summary["num_chunks_used"])
        max_qdrant_scores.append(summary["max_qdrant_score"])
        mean_qdrant_scores.append(summary["mean_qdrant_score"])
        max_conf_pass.append(summary["max_confidence_passage"])
        mean_conf_pass.append(summary["mean_confidence_passage"])
        chunk_sources.append(summary["chunk_sources"])
        chunk_buckets.append(summary["chunk_buckets"])
        chunk_doc_ids.append(summary["chunk_doc_ids"])
        chunks_texts.append(summary["chunks_text"])

    df["answer"] = answers
    df["overall_confidence"] = overall_confidences
    df["response_time_sec"] = response_times

    df["num_chunks_used"] = num_chunks_used
    df["max_qdrant_score"] = max_qdrant_scores
    df["mean_qdrant_score"] = mean_qdrant_scores
    df["max_passage_confidence"] = max_conf_pass
    df["mean_passage_confidence"] = mean_conf_pass

    df["chunk_sources"] = chunk_sources
    df["chunk_buckets"] = chunk_buckets
    df["chunk_doc_ids"] = chunk_doc_ids
    df["chunks_text"] = chunks_texts

    df.to_csv(out_csv, index=False)
    print(f"Saved evaluated CSV to: {out_csv}")


if __name__ == "__main__":
    evaluate_csv("dataset/questions.csv", question_col="question")
