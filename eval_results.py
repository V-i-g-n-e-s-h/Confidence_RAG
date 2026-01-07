import pandas as pd
import numpy as np
import re
from pathlib import Path

import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt


BASE_DIR = Path("./dataset")

CSV_INTERMEDIATE = BASE_DIR / "question_with_answer.csv"
CSV_BASELINE = BASE_DIR / "question_with_answer_no_conf.csv"

PDFS = [
    BASE_DIR / "criminal_law_act_1997.pdf",
    BASE_DIR / "a2423.pdf",
]

OUT_COMPARISON_CSV = BASE_DIR / "per_question_context_and_correctness_comparison.csv"

P_CONTEXT_LINE = BASE_DIR / "context_length_per_question.png"
P_CORRECT_STACKED = BASE_DIR / "correctness_stacked.png"
P_TRANSITION_MATRIX = BASE_DIR / "correctness_transition_matrix.png"

df_inter = pd.read_csv(CSV_INTERMEDIATE)
df_base = pd.read_csv(CSV_BASELINE)

def normalize_q(s: str) -> str:
    s = str(s) if not pd.isna(s) else ""
    return re.sub(r"\s+", " ", s).strip()

def per_question_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["question_norm"] = df["question"].apply(normalize_q)
    df["chunk_text"] = df["chunk_text"].fillna("").astype(str)
    df["answer"] = df["answer"].fillna("").astype(str)

    g = df.groupby("question_norm", sort=False)
    out = g.agg(
        passages=("chunk_text", "size"),
        context_chars=("chunk_text", lambda x: int(sum(len(t) for t in x))),
        context_words=("chunk_text", lambda x: int(sum(len(t.split()) for t in x))),
        answer=("answer", lambda x: next((a for a in x if a.strip()), "")),
    ).reset_index().rename(columns={"question_norm": "question"})
    return out

def extract_pdf_text(pdf_path: Path):
    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages):
            txt = page.extract_text() or ""
            txt = re.sub(r"\s+", " ", txt).strip()
            if txt:
                pages.append((f"{pdf_path.name}:p{i+1}", txt))
    return pages

def chunk_text(text, chunk_size=1400, overlap=250):
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


pages_all = []
for p in PDFS:
    pages_all.extend(extract_pdf_text(p))

pdf_chunks = []
chunk_meta = []
for page_id, txt in pages_all:
    for ch in chunk_text(txt, 1400, 250):
        pdf_chunks.append(ch)
        chunk_meta.append(page_id)

vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=120000)
X = vectorizer.fit_transform(pdf_chunks)

def answer_support_score(question: str, answer: str, k=6):
    answer = (answer or "").strip()
    if not answer or answer.lower() in {"none", "null", "nan"}:
        return 0.0, None

    qv = vectorizer.transform([question])
    sims_q = cosine_similarity(qv, X).ravel()

    idx = np.argpartition(-sims_q, k-1)[:k]
    idx = idx[np.argsort(-sims_q[idx])]

    av = vectorizer.transform([answer])
    sims_a = cosine_similarity(av, X[idx]).ravel()
    best_j = int(np.argmax(sims_a))
    best_idx = int(idx[best_j])
    return float(sims_a[best_j]), chunk_meta[best_idx]

def correctness_flag(score: float, answer: str, threshold=0.12) -> bool:
    a = (answer or "").strip().lower()
    if not a or a in {"none", "null", "nan"}:
        return False
    if a == "0":
        return False
    return score >= threshold


stats_base = per_question_stats(df_base)
stats_inter = per_question_stats(df_inter)

comp = stats_base.merge(stats_inter, on="question", suffixes=("_base", "_inter"), how="outer")

comp["context_words_diff"] = comp["context_words_base"] - comp["context_words_inter"]
comp["context_chars_diff"] = comp["context_chars_base"] - comp["context_chars_inter"]
comp["passages_removed"] = comp["passages_base"] - comp["passages_inter"]

comp["pct_words_reduced"] = np.where(
    comp["context_words_base"] > 0,
    comp["context_words_diff"] / comp["context_words_base"] * 100,
    np.nan
)

comp["tokens_base_est"] = (comp["context_chars_base"] / 4).round().astype("Int64")
comp["tokens_inter_est"] = (comp["context_chars_inter"] / 4).round().astype("Int64")
comp["tokens_saved_est"] = comp["tokens_base_est"] - comp["tokens_inter_est"]

comp["support_base"], comp["evidence_base"] = zip(*[
    answer_support_score(q, a) for q, a in zip(comp["question"], comp["answer_base"])
])
comp["support_inter"], comp["evidence_inter"] = zip(*[
    answer_support_score(q, a) for q, a in zip(comp["question"], comp["answer_inter"])
])

comp["is_correct_base"] = [
    correctness_flag(s, a, threshold=0.12) for s, a in zip(comp["support_base"], comp["answer_base"])
]
comp["is_correct_inter"] = [
    correctness_flag(s, a, threshold=0.12) for s, a in zip(comp["support_inter"], comp["answer_inter"])
]

comp_out = comp[[
    "question",
    "passages_base", "passages_inter", "passages_removed",
    "context_words_base", "context_words_inter", "context_words_diff", "pct_words_reduced",
    "tokens_base_est", "tokens_inter_est", "tokens_saved_est",
    "answer_base", "answer_inter",
    "support_base", "support_inter",
    "is_correct_base", "is_correct_inter",
    "evidence_base", "evidence_inter"
]].copy()

comp_out.to_csv(OUT_COMPARISON_CSV, index=False)
print(f"Saved per-question comparison CSV: {OUT_COMPARISON_CSV}")

comp_sorted = comp.sort_values("context_words_base", ascending=False).reset_index(drop=True)

plt.figure(figsize=(12, 5))
plt.plot(comp_sorted["context_words_base"].values, label="Baseline context length (words)")
plt.plot(comp_sorted["context_words_inter"].values, label="Confidence-filtered context length (words)")
plt.title("Per-question context length (words): Baseline vs Confidence-filtered")
plt.xlabel("Question index")
plt.ylabel("Context length (words)")
plt.legend()
plt.tight_layout()
plt.savefig(P_CONTEXT_LINE, dpi=200)
plt.close()


counts = pd.DataFrame({
    "Run": ["Baseline", "Confidence-filtered"],
    "Correct": [comp["is_correct_base"].sum(), comp["is_correct_inter"].sum()],
    "Incorrect": [(~comp["is_correct_base"]).sum(), (~comp["is_correct_inter"]).sum()],
})
plt.figure(figsize=(7, 5))
plt.bar(counts["Run"], counts["Correct"], label="Correct")
plt.bar(counts["Run"], counts["Incorrect"], bottom=counts["Correct"], label="Incorrect")
plt.title("Answer correctness: Baseline vs Confidence-filtered")
plt.ylabel("Number of questions (out of 50)")
plt.legend()
plt.tight_layout()
plt.savefig(P_CORRECT_STACKED, dpi=200)
plt.close()

transition = pd.crosstab(comp["is_correct_base"], comp["is_correct_inter"])
transition.index = transition.index.map({False: "Baseline Incorrect", True: "Baseline Correct"})
transition.columns = transition.columns.map({False: "Confidence Incorrect", True: "Confidence Correct"})

plt.figure(figsize=(7, 5))
mat = transition.values
plt.imshow(mat, aspect="auto")
plt.xticks(range(mat.shape[1]), transition.columns, rotation=20, ha="right")
plt.yticks(range(mat.shape[0]), transition.index)
plt.title("Correctness transitions: Baseline vs Confidence-filtered")
plt.xlabel("Confidence-filtered")
plt.ylabel("Baseline")
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        plt.text(j, i, str(mat[i, j]), ha="center", va="center")
plt.tight_layout()
plt.savefig(P_TRANSITION_MATRIX, dpi=200)
plt.close()

print("Saved charts:")
for p in [P_CONTEXT_LINE, P_CORRECT_STACKED, P_TRANSITION_MATRIX,]:
    print(" -", p)
