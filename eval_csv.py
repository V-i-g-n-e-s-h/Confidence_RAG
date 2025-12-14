import time
import os
from typing import (Optional, List,)

import pandas as pd

from rag_core.retrieval import RetrievalAndGenerationService


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

    qdrant_score = []
    conf_pass = []
    chunk_source = []
    chunks_texts = []
    questions = []
    for q in df[question_col].astype(str).tolist():
        t0 = time.perf_counter()
        result = service.answer_with_confidence(q)
        # dt = time.perf_counter() - t0

        passages = result.get("passages", []) or []

        summarys = [{
                "qdrant_score": float(p.get("qdrant_score", 0.0)),
                "confidence_passage": float(p.get("confidence", 0.0)),
                "chunk_source": p.get("source", "unknown"),
                "chunk_text": p.get("text", ""),
            } for p in passages]

        for summary in summarys:
            questions.append(q)
            qdrant_score.append(summary["qdrant_score"])
            conf_pass.append(summary["confidence_passage"])
            chunk_source.append(summary["chunk_source"])
            chunks_texts.append(summary["chunk_text"])
            # response_times.append(dt)
            answers.append(result.get("answer", ""))
            # overall_confidences.append(float(result.get("overall_confidence", 0.0)))

    # df["answer"] = answers
    # df["overall_confidence"] = overall_confidences
    # df["response_time_sec"] = response_times
    dfn  = pd.DataFrame()
    dfn["question"] = questions
    dfn["qdrant_score"] = qdrant_score
    dfn["passage_confidence"] = conf_pass

    dfn["chunk_source"] = chunk_source
    dfn["chunk_text"] = chunks_texts
    dfn["answer"] = answers

    dfn.to_csv(out_csv, index=False)
    print(f"Saved evaluated CSV to: {out_csv}")


if __name__ == "__main__":
    evaluate_csv("dataset/questions.csv", question_col="question")
