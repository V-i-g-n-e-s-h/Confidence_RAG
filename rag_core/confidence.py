import os
import json
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

from .settings import (MODEL_DIR, DEVICE, ENCODE_BATCH_SIZE,)


class ConfHead(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(4 * d + 1)
        self.mlp = nn.Sequential(
            nn.Linear(4 * d + 1, 256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        abs_diff = torch.abs(q - p)
        prod = q * p
        cos = torch.sum(q * p, dim=1, keepdim=True)
        x = torch.cat([q, p, abs_diff, prod, cos], dim=1)
        x = self.bn(x)
        out = self.mlp(x)
        return self.sigmoid(out).squeeze(1)


class ConfidenceScorerService:
    def __init__(self, model_dir: str = MODEL_DIR):
        cfg_path = os.path.join(model_dir, "config.txt")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(
                f"Missing {cfg_path}."
            )

        with open(cfg_path, "r", encoding="utf-8") as f:
            txt = f.read()

        model_name = None
        dim = None
        for line in txt.splitlines():
            if line.startswith("model_name="):
                model_name = line.split("=", 1)[1].strip()
            if line.startswith("dim="):
                dim = int(line.split("=", 1)[1].strip())

        if model_name is None:
            model_name = "BAAI/bge-small-en-v1.5"
        if dim is None:
            raise ValueError("config.txt must contain 'dim=<int>'")

        self.encoder = SentenceTransformer(model_name, device=str(DEVICE))
        self.dim = dim

        self.head = ConfHead(dim).to(DEVICE)
        head_path = os.path.join(model_dir, "confidence_head.pt")
        if not os.path.exists(head_path):
            raise FileNotFoundError(
                f"Missing {head_path}."
            )
        self.head.load_state_dict(torch.load(head_path, map_location=DEVICE))
        self.head.eval()

        self._load_calibrator(model_dir)

    def embed_question(self, question: str):
        vec = self.encoder.encode(
            [question],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vec[0]

    def _load_calibrator(self, model_dir: str):
        path = os.path.join(model_dir, "calibration.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                params = json.load(f)
            self.cal_a = float(params.get("a", 1.0))
            self.cal_b = float(params.get("b", 0.0))
        else:
            self.cal_a = 1.0
            self.cal_b = 0.0

    def _calibrate(self, scores):
        s = torch.tensor(scores, dtype=torch.float32)
        p = torch.sigmoid(self.cal_a * s + self.cal_b)
        return p.tolist()

    def score_many(self, question: str, passages, calibrate: bool = True):
        if not passages:
            return []

        texts = [question] * len(passages) + passages
        embs = self.encoder.encode(
            texts,
            batch_size=ENCODE_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        q_embs = torch.from_numpy(embs[: len(passages)]).to(DEVICE)
        p_embs = torch.from_numpy(embs[len(passages) :]).to(DEVICE)

        with torch.no_grad():
            raw = self.head(q_embs, p_embs).cpu().numpy().tolist()

        if calibrate:
            return self._calibrate(raw)
        return [float(s) for s in raw]
