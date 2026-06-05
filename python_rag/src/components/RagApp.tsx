import { useEffect, useMemo, useRef, useState } from "react";
import {
  Send,
  RefreshCw,
  FileText,
  History as HistoryIcon,
  ExternalLink,
  MessageSquare,
  Trash2,
  AlertTriangle,
  CheckCircle2,
  Loader2,
  Database,
  Cpu,
  Filter,
} from "lucide-react";

const API_BASE = import.meta.env.VITE_RAG_API_BASE_URL ?? "http://127.0.0.1:8000";

type Source = {
  score: number;
  doc_id: string;
  title: string;
  topic: string;
  difficulty: string;
  source: string;
  source_url?: string | null;
  chunk_index: number;
  preview: string;
};

type ChatResponse = {
  answer: string;
  question: string;
  top_k: number;
  sources: Source[];
  embedding_backend?: string;
  chat_backend?: string;
  metadata_filter?: Record<string, unknown>;
};

type ChatMsg = {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  error?: boolean;
};

type DocItem = {
  doc_id?: string;
  title?: string;
  topic?: string;
  difficulty?: string;
  char_count?: number;
  character_count?: number;
  chunk_count?: number;
  source_url?: string | null;
  source?: string;
  [k: string]: unknown;
};

type StatusInfo = {
  ok?: boolean;
  document_count?: number;
  chunk_count?: number;
  embedding_backend?: string;
  chat_backend?: string;
  ollama_ready?: boolean;
  [k: string]: unknown;
};

type Tab = "chat" | "documents" | "history";

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return (await res.json()) as T;
}

export default function RagApp() {
  const [tab, setTab] = useState<Tab>("chat");
  const [health, setHealth] = useState<"checking" | "ok" | "down">("checking");
  const [status, setStatus] = useState<StatusInfo | null>(null);
  const [statusWarning, setStatusWarning] = useState<string | null>(null);
  const [documents, setDocuments] = useState<DocItem[]>([]);
  const [history, setHistory] = useState<any[]>([]);
  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [latestSources, setLatestSources] = useState<Source[]>([]);
  const [input, setInput] = useState("");
  const [topK, setTopK] = useState(3);
  const [topicFilter, setTopicFilter] = useState<string>("");
  const [sending, setSending] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  const topics = useMemo(() => {
    const s = new Set<string>();
    for (const d of documents) if (d.topic) s.add(d.topic);
    return Array.from(s).sort();
  }, [documents]);

  // Boot: /health → /status, /documents
  useEffect(() => {
    (async () => {
      try {
        await api("/health");
        setHealth("ok");
      } catch {
        setHealth("down");
        return;
      }
      try {
        const s = await api<StatusInfo>("/status");
        setStatus(s);
      } catch (e: any) {
        setStatusWarning("Không thể kết nối Ollama hoặc lấy trạng thái backend.");
      }
      try {
        const d = await api<any>("/documents");
        const list: DocItem[] = Array.isArray(d) ? d : d?.documents ?? [];
        setDocuments(list);
      } catch {
        /* ignore */
      }
    })();
  }, []);

  // Load history when tab opens
  useEffect(() => {
    if (tab !== "history") return;
    (async () => {
      try {
        const h = await api<any>("/history?limit=50");
        const list = Array.isArray(h) ? h : h?.history ?? h?.items ?? [];
        setHistory(list);
      } catch {
        setHistory([]);
      }
    })();
  }, [tab]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [messages, sending]);

  async function send() {
    const q = input.trim();
    if (!q || sending) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", content: q }]);
    setSending(true);
    try {
      const body: any = { question: q, top_k: topK, metadata_filter: {} };
      if (topicFilter) body.metadata_filter.topic = topicFilter;
      const res = await api<ChatResponse>("/chat", {
        method: "POST",
        body: JSON.stringify(body),
      });
      setMessages((m) => [
        ...m,
        { role: "assistant", content: res.answer, sources: res.sources },
      ]);
      setLatestSources(res.sources || []);
    } catch (e: any) {
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: `Lỗi gọi agent: ${e?.message ?? "unknown"}`,
          error: true,
        },
      ]);
    } finally {
      setSending(false);
    }
  }

  function clearChat() {
    setMessages([]);
    setLatestSources([]);
  }

  return (
    <div className="h-screen w-screen flex flex-col bg-neutral-50 text-neutral-900">
      {/* Top bar */}
      <header className="h-14 border-b bg-white flex items-center px-4 gap-4 shrink-0">
        <div className="flex items-center gap-2 font-semibold">
          <Database className="w-5 h-5 text-neutral-700" />
          Python Docs RAG Agent
        </div>
        <StatusDot health={health} />
        <div className="ml-auto flex items-center gap-4 text-xs text-neutral-600">
          <Stat icon={<FileText className="w-3.5 h-3.5" />} label="Docs" value={status?.document_count ?? documents.length ?? "—"} />
          <Stat icon={<Database className="w-3.5 h-3.5" />} label="Chunks" value={status?.chunk_count ?? "—"} />
          <Stat icon={<Cpu className="w-3.5 h-3.5" />} label="Embed" value={status?.embedding_backend ?? "—"} mono />
          <Stat icon={<MessageSquare className="w-3.5 h-3.5" />} label="Chat" value={status?.chat_backend ?? "—"} mono />
        </div>
      </header>

      {statusWarning && (
        <div className="bg-amber-50 border-b border-amber-200 text-amber-900 text-xs px-4 py-2 flex items-center gap-2">
          <AlertTriangle className="w-4 h-4" /> {statusWarning}
        </div>
      )}

      {/* Tabs */}
      <nav className="h-10 border-b bg-white flex items-center px-2 gap-1 shrink-0">
        <TabBtn active={tab === "chat"} onClick={() => setTab("chat")} icon={<MessageSquare className="w-4 h-4" />}>
          Trò chuyện
        </TabBtn>
        <TabBtn active={tab === "documents"} onClick={() => setTab("documents")} icon={<FileText className="w-4 h-4" />}>
          Tài liệu
        </TabBtn>
        <TabBtn active={tab === "history"} onClick={() => setTab("history")} icon={<HistoryIcon className="w-4 h-4" />}>
          Lịch sử
        </TabBtn>
      </nav>

      <div className="flex-1 min-h-0 flex">
        {tab === "chat" && (
          <>
            {/* Chat main */}
            <main className="flex-1 min-w-0 flex flex-col">
              <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
                {messages.length === 0 && (
                  <div className="text-sm text-neutral-500 max-w-2xl">
                    Đặt câu hỏi về Python. Agent sẽ truy xuất ngữ cảnh từ tài liệu chính thức và trả lời kèm nguồn.
                  </div>
                )}
                {messages.map((m, i) => (
                  <MessageRow key={i} m={m} />
                ))}
                {sending && (
                  <div className="flex items-center gap-2 text-sm text-neutral-500">
                    <Loader2 className="w-4 h-4 animate-spin" /> Đang gọi agent...
                  </div>
                )}
              </div>

              {/* Controls + input */}
              <div className="border-t bg-white p-3 space-y-2">
                <div className="flex items-center gap-3 text-xs text-neutral-600">
                  <label className="flex items-center gap-1.5">
                    top_k
                    <select
                      className="border rounded px-1.5 py-0.5 bg-white"
                      value={topK}
                      onChange={(e) => setTopK(Number(e.target.value))}
                    >
                      {Array.from({ length: 10 }, (_, i) => i + 1).map((n) => (
                        <option key={n} value={n}>{n}</option>
                      ))}
                    </select>
                  </label>
                  <label className="flex items-center gap-1.5">
                    <Filter className="w-3.5 h-3.5" /> Topic
                    <select
                      className="border rounded px-1.5 py-0.5 bg-white max-w-[180px]"
                      value={topicFilter}
                      onChange={(e) => setTopicFilter(e.target.value)}
                    >
                      <option value="">(tất cả)</option>
                      {topics.map((t) => (
                        <option key={t} value={t}>{t}</option>
                      ))}
                    </select>
                  </label>
                  <button
                    onClick={clearChat}
                    className="ml-auto inline-flex items-center gap-1 px-2 py-1 rounded border hover:bg-neutral-50"
                  >
                    <Trash2 className="w-3.5 h-3.5" /> Xoá hội thoại
                  </button>
                </div>
                <div className="flex items-end gap-2">
                  <textarea
                    rows={2}
                    placeholder="Câu hỏi..."
                    className="flex-1 border rounded px-3 py-2 text-sm resize-none focus:outline-none focus:ring-1 focus:ring-neutral-400"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        send();
                      }
                    }}
                  />
                  <button
                    onClick={send}
                    disabled={sending || !input.trim()}
                    className="inline-flex items-center gap-1.5 px-3 py-2 rounded bg-neutral-900 text-white text-sm disabled:opacity-50"
                  >
                    <Send className="w-4 h-4" /> Gửi
                  </button>
                </div>
              </div>
            </main>

            {/* Right sidebar: sources */}
            <aside className="w-[340px] border-l bg-white shrink-0 flex flex-col">
              <div className="px-4 py-3 border-b text-sm font-medium flex items-center gap-2">
                <FileText className="w-4 h-4" /> Nguồn truy xuất
              </div>
              <div className="flex-1 overflow-y-auto p-3 space-y-2">
                {latestSources.length === 0 && (
                  <div className="text-xs text-neutral-500">Chưa có nguồn. Gửi câu hỏi để xem nguồn được truy xuất.</div>
                )}
                {latestSources.map((s, i) => (
                  <SourceCard key={i} s={s} />
                ))}
              </div>
            </aside>
          </>
        )}

        {tab === "documents" && (
          <main className="flex-1 overflow-y-auto p-4">
            <div className="text-sm text-neutral-600 mb-3">{documents.length} tài liệu</div>
            <div className="grid gap-2">
              {documents.map((d, i) => (
                <div key={i} className="border rounded bg-white p-3 text-sm">
                  <div className="flex items-center justify-between gap-2">
                    <div className="font-medium">{d.title ?? d.doc_id}</div>
                    {d.source_url && (
                      <a href={d.source_url} target="_blank" rel="noreferrer" className="text-xs text-blue-600 inline-flex items-center gap-1 hover:underline">
                        Mở <ExternalLink className="w-3 h-3" />
                      </a>
                    )}
                  </div>
                  <div className="mt-1 text-xs text-neutral-600 flex flex-wrap gap-x-3 gap-y-1">
                    {d.topic && <span>topic: <span className="font-mono">{d.topic}</span></span>}
                    {d.difficulty && <span>difficulty: <span className="font-mono">{d.difficulty}</span></span>}
                    {typeof (d.character_count ?? d.char_count) === "number" && (
                      <span>chars: <span className="font-mono">{d.character_count ?? d.char_count}</span></span>
                    )}
                    {typeof d.chunk_count === "number" && <span>chunks: <span className="font-mono">{d.chunk_count}</span></span>}
                  </div>
                  {d.source && <div className="mt-1 text-xs text-neutral-500 font-mono truncate">{d.source}</div>}
                </div>
              ))}
              {documents.length === 0 && <div className="text-sm text-neutral-500">Chưa có tài liệu.</div>}
            </div>
          </main>
        )}

        {tab === "history" && (
          <main className="flex-1 overflow-y-auto p-4">
            <div className="text-sm text-neutral-600 mb-3">{history.length} bản ghi</div>
            <div className="grid gap-2">
              {history.map((h, i) => (
                <div key={i} className="border rounded bg-white p-3 text-sm">
                  <div className="font-medium">{h.question ?? h.q ?? "(no question)"}</div>
                  {(h.answer ?? h.a) && (
                    <div className="mt-1 text-xs text-neutral-700 whitespace-pre-wrap line-clamp-4">{h.answer ?? h.a}</div>
                  )}
                  <div className="mt-1 text-xs text-neutral-500 font-mono">
                    {h.created_at ?? h.timestamp ?? ""}
                  </div>
                </div>
              ))}
              {history.length === 0 && <div className="text-sm text-neutral-500">Chưa có lịch sử.</div>}
            </div>
          </main>
        )}
      </div>
    </div>
  );
}

function StatusDot({ health }: { health: "checking" | "ok" | "down" }) {
  if (health === "checking")
    return (
      <span className="inline-flex items-center gap-1.5 text-xs text-neutral-500">
        <Loader2 className="w-3.5 h-3.5 animate-spin" /> kiểm tra...
      </span>
    );
  if (health === "ok")
    return (
      <span className="inline-flex items-center gap-1.5 text-xs text-emerald-700">
        <CheckCircle2 className="w-3.5 h-3.5" /> backend online
      </span>
    );
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-red-700">
      <AlertTriangle className="w-3.5 h-3.5" /> backend offline
    </span>
  );
}

function Stat({
  icon, label, value, mono,
}: { icon: React.ReactNode; label: string; value: React.ReactNode; mono?: boolean }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      {icon}
      <span className="text-neutral-500">{label}:</span>
      <span className={mono ? "font-mono" : ""}>{String(value)}</span>
    </span>
  );
}

function TabBtn({
  active, onClick, icon, children,
}: { active: boolean; onClick: () => void; icon: React.ReactNode; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded text-sm ${
        active ? "bg-neutral-900 text-white" : "text-neutral-700 hover:bg-neutral-100"
      }`}
    >
      {icon}
      {children}
    </button>
  );
}

function MessageRow({ m }: { m: ChatMsg }) {
  if (m.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] bg-neutral-900 text-white text-sm rounded px-3 py-2 whitespace-pre-wrap">
          {m.content}
        </div>
      </div>
    );
  }
  return (
    <div className="flex justify-start">
      <div
        className={`max-w-[85%] text-sm rounded px-3 py-2 whitespace-pre-wrap border ${
          m.error ? "bg-red-50 border-red-200 text-red-800" : "bg-white border-neutral-200"
        }`}
      >
        {m.content}
      </div>
    </div>
  );
}

function SourceCard({ s }: { s: Source }) {
  return (
    <div className="border rounded p-2.5 text-xs bg-white">
      <div className="flex items-center justify-between gap-2">
        <div className="font-medium text-sm truncate">{s.title}</div>
        <span className="font-mono text-[11px] text-neutral-500 shrink-0">{s.score?.toFixed(2)}</span>
      </div>
      <div className="mt-1 text-[11px] text-neutral-600 flex flex-wrap gap-x-2">
        <span>topic: <span className="font-mono">{s.topic}</span></span>
        <span>chunk: <span className="font-mono">#{s.chunk_index}</span></span>
        {s.difficulty && <span className="font-mono">{s.difficulty}</span>}
      </div>
      {s.preview && (
        <div className="mt-1.5 text-[12px] text-neutral-700 line-clamp-4">{s.preview}</div>
      )}
      {s.source_url && (
        <a
          href={s.source_url}
          target="_blank"
          rel="noreferrer"
          className="mt-1.5 inline-flex items-center gap-1 text-blue-600 hover:underline"
        >
          Mở nguồn <ExternalLink className="w-3 h-3" />
        </a>
      )}
    </div>
  );
}
