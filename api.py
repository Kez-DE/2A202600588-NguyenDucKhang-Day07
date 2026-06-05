from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from chat import OllamaChatLLM
from main import SAMPLE_FILES, load_documents_from_files
from src import (
    Document,
    EmbeddingStore,
    OllamaEmbedder,
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_MODEL,
    OLLAMA_EMBEDDING_MODEL,
    RecursiveChunker,
)


DEFAULT_DOC_DIR = "data/python_official_docs"
DEFAULT_METADATA_PATH = "data/python_official_docs_metadata.json"
DEFAULT_API_LOG_PATH = "logs/api_chat_history.jsonl"


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)
    metadata_filter: dict[str, Any] | None = None


class ChatResponse(BaseModel):
    answer: str
    question: str
    top_k: int
    sources: list[dict[str, Any]]
    embedding_backend: str
    chat_backend: str
    metadata_filter: dict[str, Any]


@dataclass
class ApiState:
    store: EmbeddingStore
    llm: OllamaChatLLM
    documents: list[dict[str, Any]]
    embedding_backend: str
    chat_backend: str
    chunk_count: int
    log_path: str


_STATE: ApiState | None = None


load_dotenv(override=False)
app = FastAPI(
    title="RAG Knowledge Base Agent API",
    description="FastAPI backend for a Lovable UI that chats with a local Ollama RAG agent.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("API_CORS_ORIGIN", "*")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _metadata_by_file(metadata_path: Path) -> dict[str, dict[str, Any]]:
    records = _read_json(metadata_path) or []
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        file_path = record.get("file_path")
        if file_path:
            result[str(Path(file_path))] = dict(record)
            result[Path(file_path).name] = dict(record)
    return result


def _load_api_source_documents() -> tuple[list[Document], list[dict[str, Any]]]:
    doc_dir = Path(os.getenv("API_DOC_DIR", DEFAULT_DOC_DIR))
    metadata_path = Path(os.getenv("API_METADATA_PATH", DEFAULT_METADATA_PATH))

    if doc_dir.exists() and doc_dir.is_dir():
        metadata_lookup = _metadata_by_file(metadata_path)
        documents: list[Document] = []
        summaries: list[dict[str, Any]] = []

        for path in sorted(doc_dir.glob("*")):
            if path.suffix.lower() not in {".md", ".txt"}:
                continue

            metadata = metadata_lookup.get(str(path)) or metadata_lookup.get(path.name) or {}
            doc_id = metadata.get("doc_id") or path.stem
            content = path.read_text(encoding="utf-8")
            metadata = {
                **metadata,
                "doc_id": doc_id,
                "source": str(path),
                "extension": path.suffix.lower(),
            }
            documents.append(Document(id=doc_id, content=content, metadata=metadata))
            summaries.append(
                {
                    "doc_id": doc_id,
                    "title": metadata.get("doc_title") or metadata.get("detected_title") or path.stem,
                    "topic": metadata.get("topic"),
                    "difficulty": metadata.get("difficulty"),
                    "source": str(path),
                    "source_url": metadata.get("source_url"),
                    "character_count": len(content),
                }
            )

        if documents:
            return documents, summaries

    fallback_docs = load_documents_from_files(SAMPLE_FILES)
    summaries = [
        {
            "doc_id": doc.id,
            "title": doc.id,
            "topic": None,
            "difficulty": None,
            "source": doc.metadata.get("source"),
            "source_url": None,
            "character_count": len(doc.content),
        }
        for doc in fallback_docs
    ]
    return fallback_docs, summaries


def build_state() -> ApiState:
    base_url = os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL)
    embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", OLLAMA_EMBEDDING_MODEL)
    chat_model = os.getenv("OLLAMA_CHAT_MODEL", OLLAMA_CHAT_MODEL)
    chunk_size = int(os.getenv("API_CHUNK_SIZE", "700"))
    log_path = os.getenv("API_LOG_PATH", DEFAULT_API_LOG_PATH)

    source_docs, document_summaries = _load_api_source_documents()
    chunker = RecursiveChunker(chunk_size=chunk_size)
    chunks: list[Document] = []
    for source_doc in source_docs:
        for index, chunk in enumerate(chunker.chunk(source_doc.content), start=1):
            metadata = dict(source_doc.metadata)
            metadata.update({"doc_id": source_doc.id, "chunk_index": index})
            chunks.append(
                Document(
                    id=f"{source_doc.id}_chunk_{index}",
                    content=chunk,
                    metadata=metadata,
                )
            )

    embedder = OllamaEmbedder(model_name=embedding_model, base_url=base_url)
    store = EmbeddingStore(collection_name="api_store", embedding_fn=embedder)
    store.add_documents(chunks)

    llm = OllamaChatLLM(model_name=chat_model, base_url=base_url)
    return ApiState(
        store=store,
        llm=llm,
        documents=document_summaries,
        embedding_backend=embedder._backend_name,
        chat_backend=llm._backend_name,
        chunk_count=store.get_collection_size(),
        log_path=log_path,
    )


def get_state() -> ApiState:
    global _STATE
    if _STATE is None:
        _STATE = build_state()
    return _STATE


def _source_from_result(result: dict[str, Any]) -> dict[str, Any]:
    metadata = result.get("metadata", {})
    return {
        "score": result.get("score"),
        "doc_id": metadata.get("doc_id") or result.get("doc_id"),
        "title": metadata.get("doc_title") or metadata.get("detected_title"),
        "topic": metadata.get("topic"),
        "difficulty": metadata.get("difficulty"),
        "source": metadata.get("source"),
        "source_url": metadata.get("source_url"),
        "chunk_index": metadata.get("chunk_index"),
        "preview": result.get("content", "")[:320],
    }


def append_api_log(record: dict[str, Any]) -> None:
    state = get_state()
    path = Path(state.log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_history(limit: int = 50) -> list[dict[str, Any]]:
    path = Path(os.getenv("API_LOG_PATH", DEFAULT_API_LOG_PATH))
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines[-limit:] if line.strip()]


def answer_question(payload: ChatRequest) -> ChatResponse:
    state = get_state()
    metadata_filter = payload.metadata_filter or {}
    if metadata_filter:
        results = state.store.search_with_filter(
            payload.question,
            top_k=payload.top_k,
            metadata_filter=metadata_filter,
        )
    else:
        results = state.store.search(payload.question, top_k=payload.top_k)

    context = "\n\n".join(
        (
            f"[{index}] source={result['metadata'].get('source')} "
            f"doc_id={result['metadata'].get('doc_id')} "
            f"chunk={result['metadata'].get('chunk_index')}\n"
            f"{result['content']}"
        )
        for index, result in enumerate(results, start=1)
    )
    prompt = (
        "Use the retrieved context to answer the question in Vietnamese. "
        "Answer only from the context. If the context is insufficient, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {payload.question}\n"
        "Answer:"
    )
    answer = state.llm(prompt)
    response = ChatResponse(
        answer=answer,
        question=payload.question,
        top_k=payload.top_k,
        sources=[_source_from_result(result) for result in results],
        embedding_backend=state.embedding_backend,
        chat_backend=state.chat_backend,
        metadata_filter=metadata_filter,
    )
    append_api_log(response.model_dump())
    return response


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "rag-agent-api",
        "embedding_model": os.getenv("OLLAMA_EMBEDDING_MODEL", OLLAMA_EMBEDDING_MODEL),
        "chat_model": os.getenv("OLLAMA_CHAT_MODEL", OLLAMA_CHAT_MODEL),
        "rag_loaded": _STATE is not None,
    }


@app.get("/status")
def status() -> dict[str, Any]:
    state = get_state()
    return {
        "ok": True,
        "service": "rag-agent-api",
        "embedding_backend": state.embedding_backend,
        "chat_backend": state.chat_backend,
        "document_count": len(state.documents),
        "chunk_count": state.chunk_count,
    }


@app.get("/documents")
def documents() -> dict[str, Any]:
    state = get_state()
    return {"documents": state.documents}


@app.get("/history")
def history(limit: int = Query(default=50, ge=1, le=200)) -> dict[str, Any]:
    return {"history": read_history(limit=limit)}


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="question is required")
    payload.question = payload.question.strip()
    return answer_question(payload)


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("api:app", host=host, port=port, reload=False)
