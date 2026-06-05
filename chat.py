from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib import request

from dotenv import load_dotenv

from main import SAMPLE_FILES, load_documents_from_files
from src import (
    Document,
    EmbeddingStore,
    KnowledgeBaseAgent,
    OllamaEmbedder,
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_MODEL,
    OLLAMA_EMBEDDING_MODEL,
    RecursiveChunker,
)


DEFAULT_CHAT_LOG_PATH = "logs/chat_history.jsonl"


class OllamaChatLLM:
    """Small Ollama chat wrapper compatible with KnowledgeBaseAgent."""

    def __init__(
        self,
        model_name: str = OLLAMA_CHAT_MODEL,
        base_url: str = OLLAMA_BASE_URL,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self._backend_name = f"ollama-chat:{model_name}"

    def __call__(self, prompt: str) -> str:
        payload = json.dumps(
            {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a concise Vietnamese knowledge-base assistant. "
                            "Answer only from the supplied context. "
                            "If the context is insufficient, say you do not know from the knowledge base."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "think": False,
                "options": {"temperature": 0.2, "num_predict": 256},
            }
        ).encode("utf-8")
        http_request = request.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(http_request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
        message = data["message"]
        return (message.get("content") or message.get("thinking") or "").strip()


def append_chat_log(question: str, answer: str, top_k: int, log_path: str | None = None) -> None:
    path = Path(log_path or os.getenv("CHAT_LOG_PATH", DEFAULT_CHAT_LOG_PATH))
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "answer": answer,
        "top_k": top_k,
        "embedding_model": os.getenv("OLLAMA_EMBEDDING_MODEL", OLLAMA_EMBEDDING_MODEL),
        "chat_model": os.getenv("OLLAMA_CHAT_MODEL", OLLAMA_CHAT_MODEL),
    }
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_agent() -> KnowledgeBaseAgent:
    load_dotenv(override=False)

    base_url = os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL)
    embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", OLLAMA_EMBEDDING_MODEL)
    chat_model = os.getenv("OLLAMA_CHAT_MODEL", OLLAMA_CHAT_MODEL)

    source_docs = load_documents_from_files(SAMPLE_FILES)
    if not source_docs:
        raise RuntimeError("No documents were loaded from SAMPLE_FILES.")

    chunker = RecursiveChunker(chunk_size=700)
    docs: list[Document] = []
    for source_doc in source_docs:
        for index, chunk in enumerate(chunker.chunk(source_doc.content), start=1):
            metadata = dict(source_doc.metadata)
            metadata.update({"doc_id": source_doc.id, "chunk_index": index})
            docs.append(
                Document(
                    id=f"{source_doc.id}_chunk_{index}",
                    content=chunk,
                    metadata=metadata,
                )
            )

    embedder = OllamaEmbedder(model_name=embedding_model, base_url=base_url)
    store = EmbeddingStore(collection_name="ollama_chat_store", embedding_fn=embedder)
    store.add_documents(docs)

    llm = OllamaChatLLM(model_name=chat_model, base_url=base_url)
    print(f"Embedding backend: {embedder._backend_name}")
    print(f"Chat backend: {llm._backend_name}")
    print(f"Loaded documents: {store.get_collection_size()}")
    return KnowledgeBaseAgent(store=store, llm_fn=llm)


def main() -> int:
    agent = build_agent()
    first_question = " ".join(sys.argv[1:]).strip()
    top_k = 3
    log_path = os.getenv("CHAT_LOG_PATH", DEFAULT_CHAT_LOG_PATH)
    print(f"Chat log: {log_path}")

    if first_question:
        answer = agent.answer(first_question, top_k=top_k)
        print(answer)
        append_chat_log(first_question, answer, top_k=top_k, log_path=log_path)
        return 0

    print("\nChat with the knowledge-base agent. Type 'exit' to quit.")
    while True:
        question = input("\nBạn: ").strip()
        if question.lower() in {"exit", "quit"}:
            return 0
        if not question:
            continue
        print("\nAgent:")
        answer = agent.answer(question, top_k=top_k)
        print(answer)
        append_chat_log(question, answer, top_k=top_k, log_path=log_path)


if __name__ == "__main__":
    raise SystemExit(main())
