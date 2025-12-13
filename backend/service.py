from typing import List, Optional

from fastapi import (FastAPI, HTTPException,)
from pydantic import BaseModel

from rag_core import RetrievalAndGenerationService

app = FastAPI(title="Confidence-aware RAG Backend")

service = RetrievalAndGenerationService()


class QARequest(BaseModel):
    question: str


class Passage(BaseModel):
    text: str
    source: str
    qdrant_score: float
    confidence: float
    conf_bucket: Optional[str] = None
    doc_id: Optional[str] = None


class QAResponse(BaseModel):
    answer: str
    overall_confidence: float
    passages: List[Passage]


class IndexRequest(BaseModel):
    filepaths: List[str]


class DocSummary(BaseModel):
    doc_id: str
    name: str
    path: str
    chunks: int


class DeleteDocRequest(BaseModel):
    doc_id: str


@app.post("/qa", response_model=QAResponse)
def qa(req: QARequest):
    result = service.answer_with_confidence(req.question)
    passages = []
    for p in result.get("passages", []):
        passages.append(
            Passage(
                text=p.get("text", ""),
                source=p.get("source", "unknown"),
                qdrant_score=float(p.get("qdrant_score", 0.0)),
                confidence=float(p.get("confidence", 0.0)),
                conf_bucket=p.get("conf_bucket"),
                doc_id=p.get("doc_id"),
            )
        )
    return QAResponse(
        answer=result.get("answer", ""),
        overall_confidence=float(result.get("overall_confidence", 0.0)),
        passages=passages,
    )


@app.get("/documents", response_model=List[DocSummary])
def list_docs():
    docs = service.list_documents()
    out = []
    for d in docs:
        out.append(
            DocSummary(
                doc_id=d.get("doc_id", ""),
                name=d.get("name", ""),
                path=d.get("path", ""),
                chunks=int(d.get("chunks", 0)),
            )
        )
    return out


@app.post("/documents/index", response_model=List[DocSummary])
def index_docs(req: IndexRequest):
    if not req.filepaths:
        raise HTTPException(status_code=400, detail="No filepaths provided.")
    for path in req.filepaths:
        service.index_document(path)
    docs = service.list_documents()
    out = []
    for d in docs:
        out.append(
            DocSummary(
                doc_id=d.get("doc_id", ""),
                name=d.get("name", ""),
                path=d.get("path", ""),
                chunks=int(d.get("chunks", 0)),
            )
        )
    return out


@app.post("/documents/delete", response_model=List[DocSummary])
def delete_doc(req: DeleteDocRequest):
    service.delete_document(req.doc_id)
    docs = service.list_documents()
    out = []
    for d in docs:
        out.append(
            DocSummary(
                doc_id=d.get("doc_id", ""),
                name=d.get("name", ""),
                path=d.get("path", ""),
                chunks=int(d.get("chunks", 0)),
            )
        )
    return out

@app.get("/documents/{doc_id}/text")
def get_doc_text(doc_id: str):
    text = service.get_document_text(doc_id)
    return {"doc_id": doc_id, "text": text}
