import os
import json
from typing import List, Optional, Literal

import requests
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

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


Backend = Literal["trained", "openai", "ollama", "hf"]


class ConfidenceScorerService:
    def __init__(
        self,
        model_dir: str = MODEL_DIR,
        backend: Optional[Backend] = None,
        openai_model: str = None,
        ollama_model: str = None,
        ollama_base_url: str = None,
        hf_model_id: str = None,
        llm_timeout_s: float = 30.0,
        llm_max_chars_per_passage: int = 1800,
    ):
        self.model_dir = model_dir
        self.backend: Backend = (backend or os.getenv("CONF_SCORER_BACKEND", "trained")).lower()
        if self.backend not in ("trained", "openai", "ollama", "hf"):
            raise ValueError(f"Unknown backend '{self.backend}'. Use trained|openai|ollama")

        self.openai_model = openai_model or os.getenv("CONF_SCORER_OPENAI_MODEL", "gpt-5.2")
        self.ollama_model = ollama_model or os.getenv("OLLAMA_MODEL", "llama3-chatqa")
        self.ollama_base_url = ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.hf_model_id = hf_model_id or os.getenv("CONF_SCORER_HF_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
        self.llm_timeout_s = llm_timeout_s
        self.llm_max_chars_per_passage = llm_max_chars_per_passage

        # Always set defaults
        self.encoder = None
        self.dim = None
        self.head = None
        self.cal_a = 1.0
        self.cal_b = 0.0

        # If trained backend: require files and load as before
        if self.backend == "trained":
            self._load_trained(model_dir)
            self._load_calibrator(model_dir)
        else:
            # LLM backends: do NOT require config/head files (so you can toggle without breaking init)
            # If you still want embeddings for other parts of your app, you can optionally keep a default encoder:
            self.encoder = SentenceTransformer("BAAI/bge-small-en-v1.5", device=str(DEVICE))
            self._init_llm_clients()

    # -------------------------
    # Existing trained loading
    # -------------------------
    def _load_trained(self, model_dir: str):
        cfg_path = os.path.join(model_dir, "config.txt")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Missing {cfg_path}.")

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
            raise FileNotFoundError(f"Missing {head_path}.")
        self.head.load_state_dict(torch.load(head_path, map_location=DEVICE))
        self.head.eval()

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

    # -------------------------
    # Public API stays same
    # -------------------------
    def embed_question(self, question: str):
        if self.encoder is None:
            raise RuntimeError("embed_question() is unavailable when backend != 'trained'.")
        vec = self.encoder.encode(
            [question],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vec[0]

    def score_many(self, question: str, passages, calibrate: bool = True):
        if not passages:
            return []

        if self.backend == "trained":
            return self._score_many_trained(question, passages, calibrate=calibrate)
        if self.backend == "openai":
            return self._score_many_openai(question, passages)
        if self.backend == "ollama":
            return self._score_many_ollama(question, passages)
        if self.backend == "hf":
            return self._score_many_hf(question, passages)

        # Should never happen due to validation in __init__
        raise RuntimeError(f"Unhandled backend: {self.backend}")

    # -------------------------
    # Trained scoring (your current logic)
    # -------------------------
    def _score_many_trained(self, question: str, passages, calibrate: bool = True):
        assert self.encoder is not None and self.head is not None

        texts = [question] * len(passages) + passages
        embs = self.encoder.encode(
            texts,
            batch_size=ENCODE_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        q_embs = torch.from_numpy(embs[: len(passages)]).to(DEVICE)
        p_embs = torch.from_numpy(embs[len(passages):]).to(DEVICE)

        with torch.no_grad():
            raw = self.head(q_embs, p_embs).cpu().numpy().tolist()

        if calibrate:
            return self._calibrate(raw)
        return [float(s) for s in raw]

    # -------------------------
    # LLM scoring
    # -------------------------
    def _init_llm_clients(self):
        self._openai_client = None
        self._hf_tokenizer = None
        self._hf_model = None
        if self.backend == "openai":
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError("Install openai: pip install openai") from e
            self._openai_client = OpenAI()

        if self.backend == "ollama":
            try:
                import requests  # type: ignore
            except ImportError as e:
                raise ImportError("Install requests: pip install requests") from e
        if self.backend == "hf":
            # Load once and keep in memory
            self._hf_tokenizer = AutoTokenizer.from_pretrained(self.hf_model_id, use_fast=True)
            self._hf_model = AutoModelForCausalLM.from_pretrained(
                self.hf_model_id,
                torch_dtype="auto",
                device_map="auto",   # GPU if available, else CPU
            )
            self._hf_model.eval()

    def _truncate(self, s: str) -> str:
        s = (s or "").strip()
        if len(s) <= self.llm_max_chars_per_passage:
            return s
        return s[: self.llm_max_chars_per_passage] + "…"

    def _build_llm_prompt(self, question: str, passages: List[str]) -> str:
        # Single call returning a JSON array of floats (0..1), same order as passages.
        parts = [
            "You are a strict relevance scorer for RAG.",
            "Given a Question and a list of Passages, output ONLY valid JSON: an array of numbers in [0, 1].",
            "The i-th number is how relevant Passage i is to answering the Question.",
            "Use 0.0 for totally irrelevant, 1.0 for directly answering.",
            "",
            f"Question: {question.strip()}",
            "",
            "Passages:",
        ]
        for i, p in enumerate(passages):
            parts.append(f"[{i}] {self._truncate(p)}")
        parts.append("")
        parts.append("Return JSON array only. Example: [0.1, 0.0, 0.85, 0.65, 0.43]")
        return "\n".join(parts)

    def _parse_json_array(self, text: str, n: int) -> List[float]:
        text = (text or "").strip()

        # Fast path
        try:
            arr = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract the first [...] block
            start = text.find("[")
            end = text.rfind("]")
            if start == -1 or end == -1 or end <= start:
                raise ValueError(f"LLM did not return JSON array. Got: {text[:200]}")
            arr = json.loads(text[start:end + 1])

        if not isinstance(arr, list) or len(arr) != n:
            raise ValueError(f"LLM JSON must be a list of length {n}, got {type(arr)} len={getattr(arr,'__len__',None)}")

        out: List[float] = []
        for x in arr:
            try:
                fx = float(x)
            except Exception:
                fx = 0.0
            # clamp
            if fx < 0.0:
                fx = 0.0
            if fx > 1.0:
                fx = 1.0
            out.append(fx)
        return out

    def _score_many_openai(self, question: str, passages) -> List[float]:
        assert self._openai_client is not None

        prompt = self._build_llm_prompt(question, list(passages))

        # Responses API call (recommended) :contentReference[oaicite:4]{index=4}
        resp = self._openai_client.responses.create(
            model=self.openai_model,
            input=prompt,
        )
        text = getattr(resp, "output_text", None) or ""
        return self._parse_json_array(text, n=len(passages))

    def _score_many_ollama(self, question: str, passages) -> List[float]:
        prompt = self._build_llm_prompt(question, list(passages))

        # Ollama generate endpoint + JSON mode :contentReference[oaicite:5]{index=5}
        url = self.ollama_base_url.rstrip("/") + "/api/chat" # os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
        r = requests.post(
            url,
            json={
                "model": self.ollama_model,
                "prompt": prompt,
                "format": "json",
                "stream": False,
            },
            timeout=self.llm_timeout_s,
        )
        r.raise_for_status()
        data = r.json()

        # Ollama typically returns the generated content in "response"
        text = data.get("response", "")
        return self._parse_json_array(text, n=len(passages))
    
    def _score_many_hf(self, question: str, passages) -> List[float]:
        assert self._hf_model is not None and self._hf_tokenizer is not None

        prompt = self._build_llm_prompt(question, list(passages))

        # Qwen Instruct works best with chat template
        messages = [
            {"role": "system", "content": "You are a strict relevance scorer for RAG."},
            {"role": "user", "content": prompt},
        ]

        tok = self._hf_tokenizer
        model = self._hf_model

        chat_text = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tok(chat_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,     # more deterministic -> better JSON
                temperature=0.0,
            )

        gen_ids = out_ids[0][inputs["input_ids"].shape[-1]:]
        text = tok.decode(gen_ids, skip_special_tokens=True).strip()

        return self._parse_json_array(text, n=len(passages))

