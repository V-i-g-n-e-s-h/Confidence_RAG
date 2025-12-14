import os
import re
import uuid
from typing import (List, Dict, Any,)

import requests
import PyPDF2
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    Distance,
    VectorParams,
    PointStruct,
)

from .settings import (
    MODEL_DIR,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION,
    QDRANT_TEXT_KEY,
    QDRANT_SOURCE_KEY,
    DOC_ID_KEY,
    FILEPATH_KEY,
    CHUNK_INDEX_KEY,
    QDRANT_TOP_K,
    OLLAMA_URL,
    OLLAMA_MODEL,
    CONTEXT_MAX_CHARS,
    CONF_THRESHOLD,
    HIGH_T,
    MED_T,
    MAX_WORDS_PER_CHUNK,
    CHUNK_WORD_OVERLAP,
    MAX_TOKENS_PER_CHUNK,
)
from .confidence import ConfidenceScorerService


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch if ch.isprintable() else " " for ch in text)
    text = re.sub(r"[^\w\s.,?!:;'\"()\-/%]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def read_file_to_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".txt", ".md", ".log", ".csv"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        return clean_text(raw)

    if ext == ".pdf":
        text = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                try:
                    txt = page.extract_text() or ""
                    text.append(txt)
                except Exception:
                    continue
        raw = "\n".join(text)
        return clean_text(raw)

    raise ValueError(f"Unsupported file type: {ext}")



def chunk_text(text: str, max_words: int = MAX_WORDS_PER_CHUNK, overlap: int = CHUNK_WORD_OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    n = len(words)

    if n == 0:
        return chunks

    while start < n:
        end = min(start + max_words, n)
        chunk_words = words[start:end]
        if len(chunk_words) > MAX_TOKENS_PER_CHUNK:
            chunk_words = chunk_words[:MAX_TOKENS_PER_CHUNK]
        chunk = " ".join(chunk_words).strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)

    return chunks


class RetrievalAndGenerationService:
    def __init__(self):
        self.conf_service = ConfidenceScorerService(MODEL_DIR)
        self.qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._ensure_collection()

    def _ensure_collection(self):
        dim = self.conf_service.dim
        collections = self.qdrant.get_collections().collections
        names = [c.name for c in collections]
        if QDRANT_COLLECTION not in names:
            self.qdrant.recreate_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def retrieve(self, question: str):
        q_vec = self.conf_service.embed_question(question)
        results = self.qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=q_vec.tolist(),
            limit=QDRANT_TOP_K,
            with_payload=True,
            with_vectors=False,
            query_filter=None,
        )
        out = []
        for r in results.points:
            payload = r.payload or {}
            text = payload.get(QDRANT_TEXT_KEY, "")
            source = payload.get(QDRANT_SOURCE_KEY, "unknown")
            out.append(
                {
                    "text": text,
                    "source": source,
                    "qdrant_score": float(r.score),
                    DOC_ID_KEY: payload.get(DOC_ID_KEY),
                }
            )
        return out

    def generate_answer(self, question: str, passages: List[Dict[str, Any]], overall_conf: float = None):
        high_ctx, med_ctx, low_ctx = [], [], []
        for p in passages:
            txt = p.get("text") or ""
            src = p.get("source") or "unknown"
            c = p.get("confidence", 0.0)
            bucket = p.get("conf_bucket", "medium")
            if not txt:
                continue
            entry = f"[source={src}; conf={c:.2f}; bucket={bucket}]\n{txt}"
            if bucket == "high":
                high_ctx.append(entry)
            elif bucket == "medium":
                med_ctx.append(entry)
            else:
                low_ctx.append(entry)

        def join(section):
            return "\n\n".join(section) if section else "(none)"

        context = (
            "### High-confidence evidence\n"
            f"{join(high_ctx)}\n\n"
            "### Medium-confidence evidence\n"
            f"{join(med_ctx)}\n\n"
            "### Low-confidence evidence\n"
            f"{join(low_ctx)}"
        )

        if len(context) > CONTEXT_MAX_CHARS:
            context = context[:CONTEXT_MAX_CHARS] + "\n\n[Context truncated]"

        prompt = (
            "You are a question answering assistant in a confidence-aware RAG system.\n"
            "You receive evidence passages with calibrated confidence scores in [0,1].\n"
            "Follow these rules:\n"
            " - Rely primarily on HIGH confidence evidence.\n"
            " - Use MEDIUM confidence evidence cautiously.\n"
            " - Only use LOW confidence evidence if nothing else is available, and state uncertainty.\n"
            " - If nothing clearly supports an answer, say you don't know.\n"
            " - IF THE ANSWER NOT AVAILABLE IN THE PROVIDED ANSWER SAY YOU DON'T KNOW.\n\n"
        )

        if overall_conf is not None:
            prompt += f"Overall system confidence estimate: {overall_conf:.2f}\n\n"

        prompt += (
            f"Question:\n{question}\n\n"
            f"Evidence:\n{context}\n\n"
            "Now provide a clear and concise answer, explicitly reflecting uncertainty when appropriate:"
        )

        payload = {
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        fail = 0
        while fail != 3:
            try:
                resp = requests.post(OLLAMA_URL, json=payload, timeout=600)
                break
            except:
                fail += 1
        try:
            resp.raise_for_status()
            data = resp.json()
        except:
            data = {}
        msg = data.get("message", {})
        answer = msg.get("content", "").strip()
        return answer

    def answer_with_confidence(self, question: str):
        retrieved = self.retrieve(question)
        passages = [r["text"] for r in retrieved]
        if not passages:
            return {
                "answer": "I could not retrieve any relevant context for this question.",
                "overall_confidence": 0.0,
                "passages": [],
            }

        conf_scores = self.conf_service.score_many(question, passages, calibrate=True)
        for r, s in zip(retrieved, conf_scores):
            r["confidence"] = s

        filtered = [r for r in retrieved if r["confidence"] >= CONF_THRESHOLD]

        if not filtered:
            return {
                "answer": "The system is not confident enough in any retrieved evidence to answer.",
                "overall_confidence": max(conf_scores),
                "passages": retrieved,
            }

        filtered.sort(key=lambda x: x["confidence"], reverse=True)

        for r in filtered:
            c = r["confidence"]
            if c >= HIGH_T:
                r["conf_bucket"] = "high"
            elif c >= MED_T:
                r["conf_bucket"] = "medium"
            else:
                r["conf_bucket"] = "low"

        overall_conf = max(r["confidence"] for r in filtered)
        
        answer = self.generate_answer(question, filtered, overall_conf)

        return {
            "answer": answer,
            "overall_confidence": overall_conf,
            "passages": filtered,
        }


    def index_document(self, filepath: str):
        """Read file, chunk, embed, and upsert into Qdrant."""
        text = read_file_to_text(filepath)
        chunks = chunk_text(text)
        if not chunks:
            raise ValueError(f"No text extracted from {filepath}")

        doc_uuid = uuid.uuid4()
        doc_id = str(doc_uuid)
        filename = os.path.basename(filepath)

        vectors = self.conf_service.encoder.encode(
            chunks,
            batch_size=256,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        points = []
        for idx, (chunk, vec) in enumerate(zip(chunks, vectors)):
            payload = {
                QDRANT_TEXT_KEY: chunk,
                QDRANT_SOURCE_KEY: filename,
                DOC_ID_KEY: doc_id,
                FILEPATH_KEY: os.path.abspath(filepath),
                CHUNK_INDEX_KEY: idx,
            }
            point_id = str(uuid.uuid4())
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vec.tolist(),
                    payload=payload,
                )
            )

        self.qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)

        return {
            "doc_id": doc_id,
            "name": filename,
            "path": os.path.abspath(filepath),
            "chunks": len(chunks),
        }

    def list_documents(self):
        """Return unique documents based on doc_id."""
        docs = {}
        offset = None
        while True:
            res, offset = self.qdrant.scroll(
                collection_name=QDRANT_COLLECTION,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for pt in res:
                payload = pt.payload or {}
                doc_id = payload.get(DOC_ID_KEY)
                if not doc_id:
                    continue
                entry = docs.get(doc_id)
                if entry is None:
                    entry = {
                        "doc_id": doc_id,
                        "name": payload.get(QDRANT_SOURCE_KEY, "unknown"),
                        "path": payload.get(FILEPATH_KEY, ""),
                        "chunks": 0,
                    }
                    docs[doc_id] = entry
                entry["chunks"] += 1
            if offset is None:
                break
        return list(docs.values())

    def delete_document(self, doc_id: str):
        flt = Filter(
            must=[
                FieldCondition(
                    key=DOC_ID_KEY,
                    match=MatchValue(value=doc_id),
                )
            ]
        )
        self.qdrant.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=flt,
        )

    def get_document_text(self, doc_id: str) -> str:
        flt = Filter(
            must=[
                FieldCondition(
                    key=DOC_ID_KEY,
                    match=MatchValue(value=doc_id),
                )
            ]
        )
        chunks = []
        offset = None
        while True:
            res, offset = self.qdrant.scroll(
                collection_name=QDRANT_COLLECTION,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
                scroll_filter=flt,
            )
            for pt in res:
                payload = pt.payload or {}
                idx = payload.get(CHUNK_INDEX_KEY, 0)
                text = payload.get(QDRANT_TEXT_KEY, "")
                chunks.append((idx, text))
            if offset is None:
                break
        chunks.sort(key=lambda x: x[0])
        return "\n\n".join(t for _, t in chunks)
