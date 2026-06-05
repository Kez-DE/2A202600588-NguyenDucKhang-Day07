from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from threading import Lock
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
    FixedSizeChunker,
    OllamaEmbedder,
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_MODEL,
    OLLAMA_EMBEDDING_MODEL,
    RecursiveChunker,
    SentenceChunker,
)


DEFAULT_DOC_DIR = "data/python_official_docs"
DEFAULT_METADATA_PATH = "data/python_official_docs_metadata.json"
DEFAULT_API_LOG_PATH = "logs/api_chat_history.jsonl"
DEFAULT_CHUNKING_STRATEGY = "recursive"
CHUNKING_STRATEGY_ORDER = ["recursive", "fixed_size", "sentence", "header_aware"]
SUPPORTED_CHUNKING_STRATEGIES = set(CHUNKING_STRATEGY_ORDER)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)
    chunking_strategy: str = DEFAULT_CHUNKING_STRATEGY
    metadata_filter: dict[str, Any] | None = None


class ChatResponse(BaseModel):
    answer: str
    question: str
    top_k: int
    chunking_strategy: str
    sources: list[dict[str, Any]]
    embedding_backend: str
    chat_backend: str
    metadata_filter: dict[str, Any]


@dataclass
class ApiState:
    source_docs: list[Document]
    stores_by_strategy: dict[str, EmbeddingStore]
    strategy_stats: dict[str, dict[str, Any]]
    embedder: OllamaEmbedder
    llm: OllamaChatLLM
    documents: list[dict[str, Any]]
    embedding_backend: str
    chat_backend: str
    chunk_size: int
    log_path: str


_STATE: ApiState | None = None
_INDEX_LOCK = Lock()


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


class HeaderAwareChunker:
    """Keep Markdown sections together, then recursively split oversized sections."""

    def __init__(self, chunk_size: int = 700) -> None:
        self.chunk_size = chunk_size
        self.fallback = RecursiveChunker(chunk_size=chunk_size)

    def chunk(self, text: str) -> list[str]:
        sections = re.split(r"(?=^#{1,3}\s)", text.strip(), flags=re.MULTILINE)
        chunks: list[str] = []
        for section in sections:
            section = section.strip()
            if not section:
                continue
            if len(section) <= self.chunk_size:
                chunks.append(section)
            else:
                chunks.extend(self.fallback.chunk(section))
        return chunks


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


def _build_chunker(strategy: str, chunk_size: int):
    if strategy == "fixed_size":
        return FixedSizeChunker(chunk_size=chunk_size, overlap=50)
    if strategy == "sentence":
        return SentenceChunker(max_sentences_per_chunk=3)
    if strategy == "recursive":
        return RecursiveChunker(chunk_size=chunk_size)
    if strategy == "header_aware":
        return HeaderAwareChunker(chunk_size=chunk_size)
    raise ValueError(f"Unsupported chunking strategy: {strategy}")


def _build_chunk_documents(source_docs: list[Document], strategy: str, chunker) -> list[Document]:
    chunks: list[Document] = []
    for source_doc in source_docs:
        for index, chunk in enumerate(chunker.chunk(source_doc.content), start=1):
            metadata = dict(source_doc.metadata)
            metadata.update(
                {
                    "doc_id": source_doc.id,
                    "chunk_index": index,
                    "chunking_strategy": strategy,
                }
            )
            chunks.append(
                Document(
                    id=f"{source_doc.id}_{strategy}_chunk_{index}",
                    content=chunk,
                    metadata=metadata,
                )
            )
    return chunks


def _summarize_chunks(chunks: list[Document]) -> dict[str, Any]:
    lengths = [len(chunk.content) for chunk in chunks]
    return {
        "chunk_count": len(chunks),
        "avg_length": round(mean(lengths), 1) if lengths else 0.0,
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
    }


def _strategy_label(strategy: str) -> str:
    labels = {
        "fixed_size": "Fixed Size",
        "sentence": "Sentence",
        "recursive": "Recursive",
        "header_aware": "Header Aware",
    }
    return labels[strategy]


def _strategy_description(strategy: str) -> str:
    descriptions = {
        "fixed_size": "Splits text into fixed 700-character chunks with 50-character overlap.",
        "sentence": "Groups text by natural sentence boundaries, 3 sentences per chunk.",
        "recursive": "Splits by paragraph, line, sentence, word, then characters.",
        "header_aware": "Keeps Markdown heading sections together, then recursively splits oversized sections.",
    }
    return descriptions[strategy]


def _strategy_comparison(strategy: str) -> dict[str, Any]:
    comparisons = {
        "recursive": {
            "how_it_splits": "Tries large natural boundaries first: paragraph, line, sentence, word, then character fallback.",
            "strengths": [
                "Balances chunk length and semantic coherence.",
                "Works well for Markdown documentation with paragraphs and sections.",
                "Avoids cutting text too aggressively when natural separators exist.",
            ],
            "weaknesses": [
                "Can still detach headings from details if the source formatting is inconsistent.",
                "Chunk sizes are less predictable than fixed-size chunking.",
            ],
            "best_for": "General documentation, tutorials, mixed prose/code content.",
            "risk": "Medium-size chunks may include extra context that slightly dilutes very specific queries.",
        },
        "fixed_size": {
            "how_it_splits": "Cuts text by character window size, using a fixed overlap between adjacent chunks.",
            "strengths": [
                "Predictable chunk length.",
                "Simple to reason about and fast to compute.",
                "Overlap helps preserve some boundary context.",
            ],
            "weaknesses": [
                "Can cut in the middle of sentences, code blocks, or explanations.",
                "Does not understand document structure.",
            ],
            "best_for": "Plain text with few structural markers, or baseline comparisons.",
            "risk": "Relevant information may be split across chunk boundaries.",
        },
        "sentence": {
            "how_it_splits": "Detects sentence boundaries and groups a fixed number of sentences per chunk.",
            "strengths": [
                "Chunks are readable and usually preserve complete sentences.",
                "Good for FAQ-style content and short explanatory paragraphs.",
            ],
            "weaknesses": [
                "Chunk length can be very uneven when sentences are long.",
                "May separate a section heading from the sentences below it.",
                "Can create too many small chunks for technical docs.",
            ],
            "best_for": "Short prose, FAQ answers, policy text, and simple explanations.",
            "risk": "Small chunks may lose surrounding context needed for grounded answers.",
        },
        "header_aware": {
            "how_it_splits": "Splits Markdown by headings first, then recursively splits oversized sections.",
            "strengths": [
                "Keeps section title and section content closer together.",
                "Very useful for official docs with clear Markdown headings.",
                "Source traceability is easier because chunks map to sections.",
            ],
            "weaknesses": [
                "Depends on good heading structure.",
                "Can create many chunks if headings are frequent.",
                "Less useful for plain text without headings.",
            ],
            "best_for": "Markdown docs, official documentation, manuals, and sectioned tutorials.",
            "risk": "Poorly formatted headings can produce awkward chunks.",
        },
    }
    return comparisons[strategy]


def build_state() -> ApiState:
    base_url = os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL)
    embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", OLLAMA_EMBEDDING_MODEL)
    chat_model = os.getenv("OLLAMA_CHAT_MODEL", OLLAMA_CHAT_MODEL)
    chunk_size = int(os.getenv("API_CHUNK_SIZE", "700"))
    log_path = os.getenv("API_LOG_PATH", DEFAULT_API_LOG_PATH)

    source_docs, document_summaries = _load_api_source_documents()
    embedder = OllamaEmbedder(model_name=embedding_model, base_url=base_url)

    stores_by_strategy: dict[str, EmbeddingStore] = {}
    strategy_stats: dict[str, dict[str, Any]] = {}
    for strategy in CHUNKING_STRATEGY_ORDER:
        chunker = _build_chunker(strategy, chunk_size=chunk_size)
        chunks = _build_chunk_documents(source_docs, strategy, chunker)
        strategy_stats[strategy] = {
            "id": strategy,
            "label": _strategy_label(strategy),
            "description": _strategy_description(strategy),
            **_strategy_comparison(strategy),
            **_summarize_chunks(chunks),
        }

    llm = OllamaChatLLM(model_name=chat_model, base_url=base_url)
    return ApiState(
        source_docs=source_docs,
        stores_by_strategy=stores_by_strategy,
        strategy_stats=strategy_stats,
        embedder=embedder,
        llm=llm,
        documents=document_summaries,
        embedding_backend=embedder._backend_name,
        chat_backend=llm._backend_name,
        chunk_size=chunk_size,
        log_path=log_path,
    )


def get_state() -> ApiState:
    global _STATE
    if _STATE is None:
        _STATE = build_state()
    return _STATE


def get_strategy_store(state: ApiState, strategy: str) -> EmbeddingStore:
    if strategy in state.stores_by_strategy:
        return state.stores_by_strategy[strategy]

    with _INDEX_LOCK:
        if strategy in state.stores_by_strategy:
            return state.stores_by_strategy[strategy]

        chunker = _build_chunker(strategy, chunk_size=state.chunk_size)
        chunks = _build_chunk_documents(state.source_docs, strategy, chunker)
        store = EmbeddingStore(collection_name=f"api_store_{strategy}", embedding_fn=state.embedder)
        store.add_documents(chunks)
        state.stores_by_strategy[strategy] = store
        return store


def public_strategy_stats(state: ApiState) -> list[dict[str, Any]]:
    return [
        {
            **state.strategy_stats[strategy],
            "index_built": strategy in state.stores_by_strategy,
        }
        for strategy in CHUNKING_STRATEGY_ORDER
    ]


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
        "chunking_strategy": metadata.get("chunking_strategy"),
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
    strategy = payload.chunking_strategy.strip().lower()
    if strategy not in SUPPORTED_CHUNKING_STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported chunking_strategy. Use one of: {sorted(SUPPORTED_CHUNKING_STRATEGIES)}",
        )

    store = get_strategy_store(state, strategy)
    metadata_filter = payload.metadata_filter or {}
    if metadata_filter:
        results = store.search_with_filter(
            payload.question,
            top_k=payload.top_k,
            metadata_filter=metadata_filter,
        )
    else:
        results = store.search(payload.question, top_k=payload.top_k)

    context = "\n\n".join(
        (
            f"[{index}] source={result['metadata'].get('source')} "
            f"doc_id={result['metadata'].get('doc_id')} "
            f"strategy={result['metadata'].get('chunking_strategy')} "
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
        chunking_strategy=strategy,
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
        "default_strategy": DEFAULT_CHUNKING_STRATEGY,
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
        "default_strategy": DEFAULT_CHUNKING_STRATEGY,
        "chunk_count": state.strategy_stats[DEFAULT_CHUNKING_STRATEGY]["chunk_count"],
        "strategies": public_strategy_stats(state),
    }


@app.get("/strategies")
def strategies() -> dict[str, Any]:
    state = get_state()
    return {
        "default_strategy": DEFAULT_CHUNKING_STRATEGY,
        "strategies": public_strategy_stats(state),
    }


@app.get("/strategy-comparison")
def strategy_comparison() -> dict[str, Any]:
    return {
        "default_strategy": DEFAULT_CHUNKING_STRATEGY,
        "strategies": [
            {
                "id": strategy,
                "label": _strategy_label(strategy),
                "description": _strategy_description(strategy),
                **_strategy_comparison(strategy),
            }
            for strategy in CHUNKING_STRATEGY_ORDER
        ],
        "summary": (
            "Fixed Size is predictable but can cut semantic boundaries. "
            "Sentence chunking keeps readable sentences but may lose context. "
            "Recursive chunking balances structure and chunk size. "
            "Header Aware chunking is strongest when Markdown headings are reliable."
        ),
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
