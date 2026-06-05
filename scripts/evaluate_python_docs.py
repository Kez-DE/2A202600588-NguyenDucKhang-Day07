from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv

from main import SAMPLE_FILES, load_documents_from_files
from src import (
    Document,
    EmbeddingStore,
    FixedSizeChunker,
    OllamaEmbedder,
    OLLAMA_BASE_URL,
    OLLAMA_EMBEDDING_MODEL,
    RecursiveChunker,
    SentenceChunker,
)

OUTPUT_PATH = Path("logs/python_docs_retrieval_eval.json")

BENCHMARKS = [
    {
        "id": 1,
        "query": "Python list comprehension khác gì so với for loop khi tạo list mới?",
        "gold_topic": "data_structures",
        "gold_doc_id": "python_data_structures",
        "gold_answer": "List comprehension tạo list mới bằng cú pháp ngắn gọn gồm expression và for clause; for loop làm cùng việc nhưng dài hơn và cần append thủ công.",
    },
    {
        "id": 2,
        "query": "Python module import hoạt động như thế nào, và Module Search Path ảnh hưởng gì?",
        "gold_topic": "modules",
        "gold_doc_id": "python_modules",
        "gold_answer": "import nạp definitions từ module; Module Search Path quyết định Python tìm module ở thư mục script, PYTHONPATH và các thư mục cài đặt chuẩn theo thứ tự nào.",
    },
    {
        "id": 3,
        "query": "try-except-finally / exception handling trong Python xử lý lỗi ra sao?",
        "gold_topic": "errors",
        "gold_doc_id": "python_errors_exceptions",
        "gold_answer": "try chạy code có thể lỗi, except bắt exception phù hợp, finally chạy cleanup dù có lỗi hay không; raise dùng để phát sinh hoặc phát lại exception.",
    },
    {
        "id": 4,
        "query": "Class, instance, attribute, method trong Python OOP khác nhau thế nào?",
        "gold_topic": "oop",
        "gold_doc_id": "python_classes",
        "gold_answer": "Class định nghĩa kiểu đối tượng; instance là object tạo từ class; attribute là dữ liệu gắn với object/class; method là function thuộc class và thường thao tác trên instance.",
    },
    {
        "id": 5,
        "query": "Virtual environment và pip giúp quản lý dependency conflict như thế nào?",
        "gold_topic": "environment",
        "gold_doc_id": "python_venv",
        "gold_answer": "Virtual environment tách package theo từng project; pip cài, nâng cấp và quản lý package trong môi trường đó, giúp hai project dùng version dependency khác nhau.",
    },
]


class HeaderAwareChunker:
    """Simple custom chunker: keep Markdown sections together, then recursively split oversized sections."""

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


def build_chunk_documents(source_docs: list[Document], chunker) -> list[Document]:
    chunk_docs: list[Document] = []
    for source_doc in source_docs:
        chunks = chunker.chunk(source_doc.content)
        for index, chunk in enumerate(chunks, start=1):
            metadata = dict(source_doc.metadata)
            metadata.update({"doc_id": source_doc.id, "chunk_index": index})
            chunk_docs.append(Document(id=f"{source_doc.id}_chunk_{index}", content=chunk, metadata=metadata))
    return chunk_docs


def summarize_chunks(chunk_docs: list[Document]) -> dict:
    lengths = [len(doc.content) for doc in chunk_docs]
    return {
        "chunk_count": len(chunk_docs),
        "avg_length": round(mean(lengths), 1) if lengths else 0,
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
    }


def evaluate_strategy(name: str, chunker, source_docs: list[Document], embedder) -> dict:
    chunk_docs = build_chunk_documents(source_docs, chunker)
    store = EmbeddingStore(collection_name=f"eval_{name}", embedding_fn=embedder)
    store.add_documents(chunk_docs)

    queries = []
    top3_relevant = 0
    top1_relevant = 0
    for item in BENCHMARKS:
        results = store.search(item["query"], top_k=3)
        compact = []
        for result in results:
            metadata = result["metadata"]
            compact.append(
                {
                    "doc_id": metadata.get("doc_id"),
                    "topic": metadata.get("topic"),
                    "chunk_index": metadata.get("chunk_index"),
                    "score": round(float(result["score"]), 4),
                    "preview": result["content"][:180].replace("\n", " "),
                }
            )
        is_top1 = bool(compact and compact[0]["doc_id"] == item["gold_doc_id"])
        is_top3 = any(row["doc_id"] == item["gold_doc_id"] for row in compact)
        top1_relevant += int(is_top1)
        top3_relevant += int(is_top3)
        queries.append({**item, "top1_relevant": is_top1, "top3_relevant": is_top3, "top3": compact})

    return {
        "strategy": name,
        "chunk_summary": summarize_chunks(chunk_docs),
        "top1_relevant": top1_relevant,
        "top3_relevant": top3_relevant,
        "queries": queries,
    }


def main() -> int:
    load_dotenv(override=False)
    source_docs = load_documents_from_files(SAMPLE_FILES)
    embedder = OllamaEmbedder(
        model_name=os.getenv("OLLAMA_EMBEDDING_MODEL", OLLAMA_EMBEDDING_MODEL),
        base_url=os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL),
    )
    embedder("embedding smoke test")

    strategies = [
        ("FixedSizeChunker(chunk_size=700, overlap=50)", FixedSizeChunker(chunk_size=700, overlap=50)),
        ("SentenceChunker(max_sentences_per_chunk=3)", SentenceChunker(max_sentences_per_chunk=3)),
        ("RecursiveChunker(chunk_size=700) -- my strategy", RecursiveChunker(chunk_size=700)),
        ("HeaderAwareChunker(chunk_size=700)", HeaderAwareChunker(chunk_size=700)),
    ]
    results = [evaluate_strategy(name, chunker, source_docs, embedder) for name, chunker in strategies]
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "embedding_backend": getattr(embedder, "_backend_name", "unknown"),
        "source_documents": [
            {
                "doc_id": doc.metadata.get("doc_id", doc.id),
                "title": doc.metadata.get("doc_title"),
                "topic": doc.metadata.get("topic"),
                "difficulty": doc.metadata.get("difficulty"),
                "source_url": doc.metadata.get("source_url"),
                "chars": len(doc.content),
            }
            for doc in source_docs
        ],
        "benchmarks": BENCHMARKS,
        "results": results,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {OUTPUT_PATH}")
    print(f"Embedding backend: {output['embedding_backend']}")
    print(f"Source docs: {len(source_docs)}")
    for result in results:
        summary = result["chunk_summary"]
        print(
            f"{result['strategy']} | chunks={summary['chunk_count']} | avg={summary['avg_length']} | "
            f"top1={result['top1_relevant']}/5 | top3={result['top3_relevant']}/5"
        )
        for q in result["queries"]:
            top = q["top3"][0] if q["top3"] else {}
            print(
                f"  Q{q['id']}: top1={top.get('doc_id')} score={top.get('score')} "
                f"top3_relevant={q['top3_relevant']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
