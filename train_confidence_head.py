import os
import json
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import (Dataset, DataLoader,)
from sentence_transformers import SentenceTransformer

from rag_core.confidence import ConfHead
from rag_core.settings import (MODEL_DIR, DEVICE, ENCODE_BATCH_SIZE,)


DATASET_CSV_PATH = "dataset/filtered_dataset.csv"
METRICS_CSV_PATH = os.path.join(MODEL_DIR, "training_metrics.csv")
TRAIN_FRACTION = 0.8
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-3
RANDOM_SEED = 42


class QPPDataset(Dataset):
    def __init__(self, questions: List[str], passages: List[str], targets: List[float]):
        assert len(questions) == len(passages) == len(targets)
        self.questions = questions
        self.passages = passages
        self.targets = targets

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return (
            self.questions[idx],
            self.passages[idx],
            float(self.targets[idx]),
        )


def load_config(model_dir: str):
    cfg_path = os.path.join(model_dir, "config.txt")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"Missing {cfg_path}."
        )

    model_name = None
    dim = None
    with open(cfg_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("model_name="):
                model_name = line.split("=", 1)[1].strip()
            elif line.startswith("dim="):
                dim = int(line.split("=", 1)[1].strip())

    if model_name is None:
        model_name = "BAAI/bge-small-en-v1.5"
    if dim is None:
        raise ValueError("config.txt must contain 'dim=<int>'")

    return model_name, dim


def compute_classification_metrics(preds, targets, threshold=0.5):
    with torch.no_grad():
        labels = (targets >= 0.5).long()
        y_hat = (preds >= threshold).long()

        correct = (y_hat == labels).sum().item()
        total = len(labels)
        acc = correct / total if total > 0 else 0.0

        tp = ((y_hat == 1) & (labels == 1)).sum().item()
        tn = ((y_hat == 0) & (labels == 0)).sum().item()
        fp = ((y_hat == 1) & (labels == 0)).sum().item()
        fn = ((y_hat == 0) & (labels == 1)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def train_conf_head_from_csv():
    model_name, dim = load_config(MODEL_DIR)
    print(f"Using encoder: {model_name} (dim={dim})")
    print(f"Device: {DEVICE}")

    encoder = SentenceTransformer(model_name, device=str(DEVICE))
    encoder.eval()

    if not os.path.exists(DATASET_CSV_PATH):
        raise FileNotFoundError(f"Dataset CSV not found at {DATASET_CSV_PATH}")

    df = pd.read_csv(DATASET_CSV_PATH)
    df = df.dropna(subset=['question', 'passage', 'target_confidence'])

    df["target_confidence"] = df["target_confidence"].astype(float)
    df["target_confidence"] = df["target_confidence"].clip(0.0, 1.0)

    questions = df["question"].astype(str).tolist()
    passages = df["passage"].astype(str).tolist()
    targets = df["target_confidence"].tolist()

    n = len(questions)
    if n == 0:
        raise ValueError("Dataset is empty after cleaning.")

    print(f"Total examples in CSV: {n}")

    np.random.seed(RANDOM_SEED)
    indices = np.arange(n)
    np.random.shuffle(indices)

    train_size = int(TRAIN_FRACTION * n)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    def select(idx_array):
        return (
            [questions[i] for i in idx_array],
            [passages[i] for i in idx_array],
            [targets[i] for i in idx_array],
        )

    train_q, train_p, train_t = select(train_idx)
    val_q, val_p, val_t = select(val_idx)

    print(f"Train examples: {len(train_q)}")
    print(f"Val examples:   {len(val_q)}")

    train_ds = QPPDataset(train_q, train_p, train_t)
    val_ds = QPPDataset(val_q, val_p, val_t)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    conf_head = ConfHead(dim).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(conf_head.parameters(), lr=LEARNING_RATE)

    metrics_history = []

    for epoch in range(1, EPOCHS + 1):
        conf_head.train()
        total_loss = 0.0
        n_batches = 0

        for questions_b, passages_b, targets_b in train_loader:
            targets_b = targets_b.float().to(DEVICE)  # [B]

            with torch.no_grad():
                texts = list(questions_b) + list(passages_b)
                embs = encoder.encode(
                    texts,
                    batch_size=ENCODE_BATCH_SIZE,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                q_embs = torch.from_numpy(embs[: len(questions_b)]).to(DEVICE)
                p_embs = torch.from_numpy(embs[len(questions_b) :]).to(DEVICE)

            optimizer.zero_grad()
            probs = conf_head(q_embs, p_embs)
            loss = criterion(probs, targets_b)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_loss / max(1, n_batches)
        print(f"[Epoch {epoch}] Train loss: {avg_train_loss:.4f}")

        val_loss, val_metrics = evaluate_conf_head(
            encoder, conf_head, val_loader, criterion
        )
        print(
            f"\nVal loss: {val_loss:.4f} | "
            f"Acc: {val_metrics['accuracy']:.3f} | "
            f"F1: {val_metrics['f1']:.3f} | "
            f"Precision: {val_metrics['precision']:.3f} | "
            f"Recall: {val_metrics['recall']:.3f}\n"
        )
        metrics_history.append({
            "epoch": epoch,
            "train_loss": float(avg_train_loss),
            "val_loss": float(val_loss),
            "accuracy": float(val_metrics["accuracy"]),
            "precision": float(val_metrics["precision"]),
            "recall": float(val_metrics["recall"]),
            "f1": float(val_metrics["f1"]),
            "tp": int(val_metrics["tp"]),
            "tn": int(val_metrics["tn"]),
            "fp": int(val_metrics["fp"]),
            "fn": int(val_metrics["fn"]),
        })

    os.makedirs(MODEL_DIR, exist_ok=True)
    head_path = os.path.join(MODEL_DIR, "confidence_head.pt")
    torch.save(conf_head.state_dict(), head_path)
    print(f"Saved trained confidence head to: {head_path}")

    if metrics_history:
        metrics_df = pd.DataFrame(metrics_history)
        metrics_df.to_csv(METRICS_CSV_PATH, index=False)

    calib_path = os.path.join(MODEL_DIR, "calibration.json")
    fit_platt_calibration(encoder, conf_head, val_loader, calib_path)
    print(f"Saved calibration parameters to: {calib_path}")


def evaluate_conf_head(encoder, conf_head, data_loader, criterion):
    conf_head.eval()
    total_loss = 0.0
    n_batches = 0
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for questions_b, passages_b, targets_b in data_loader:
            targets_b = targets_b.float().to(DEVICE)
            texts = list(questions_b) + list(passages_b)
            embs = encoder.encode(
                texts,
                batch_size=ENCODE_BATCH_SIZE,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            q_embs = torch.from_numpy(embs[: len(questions_b)]).to(DEVICE)
            p_embs = torch.from_numpy(embs[len(questions_b) :]).to(DEVICE)

            probs = conf_head(q_embs, p_embs)
            loss = criterion(probs, targets_b)

            total_loss += loss.item()
            n_batches += 1

            all_probs.append(probs.cpu())
            all_targets.append(targets_b.cpu())

    avg_loss = total_loss / max(1, n_batches)
    all_probs = torch.cat(all_probs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics = compute_classification_metrics(all_probs, all_targets)
    return avg_loss, metrics


class PlattCalibrator(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.sigmoid(self.a * x + self.b)


def fit_platt_calibration(encoder, conf_head, data_loader, out_path: str, epochs=50, lr=1e-2):
    conf_head.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for questions_b, passages_b, targets_b in data_loader:
            labels_b = (targets_b >= 0.5).float()
            texts = list(questions_b) + list(passages_b)
            embs = encoder.encode(
                texts,
                batch_size=ENCODE_BATCH_SIZE,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            q_embs = torch.from_numpy(embs[: len(questions_b)]).to(DEVICE)
            p_embs = torch.from_numpy(embs[len(questions_b) :]).to(DEVICE)

            probs = conf_head(q_embs, p_embs).cpu()
            all_scores.append(probs)
            all_labels.append(labels_b)

    if not all_scores:
        print("No validation data for calibration; skipping.")
        return

    scores = torch.cat(all_scores, dim=0).unsqueeze(1)
    labels = torch.cat(all_labels, dim=0).unsqueeze(1)

    calib = PlattCalibrator()
    optimizer = torch.optim.Adam(calib.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = calib(scores)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

    a = calib.a.item()
    b = calib.b.item()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"a": a, "b": b}, f, indent=2)


if __name__ == "__main__":
    train_conf_head_from_csv()