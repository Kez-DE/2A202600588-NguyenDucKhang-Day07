import { useEffect, useMemo, useRef, useState } from "react";
import {
  Send,
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
  Sparkles,
  Code2,
  Zap,
  Layers,
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
      } catch {
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

  const suggestions = [
    "List comprehension khác for loop?",
    "async/await hoạt động thế nào?",
    "Type hints có lợi gì?",
    "Decorators dùng để làm gì?",
  ];

  return (
    <div className="h-screen w-screen flex flex-col text-foreground overflow-hidden">
      {/* Top bar */}
      <header className="h-16 border-b border-sky-soft/60 bg-white/70 backdrop-blur-xl flex items-center px-5 gap-4 shrink-0 relative z-20">
        <div className="flex items-center gap-3">
          <div
            className="w-10 h-10 rounded-xl grid place-items-center text-white shadow-lg"
            style={{ background: "var(--gradient-hero)", boxShadow: "var(--shadow-sky)" }}
          >
            <Sparkles className="w-5 h-5" />
          </div>
          <div className="leading-tight">
            <div className="font-display font-semibold tracking-tight text-base">
              <span className="text-sky-600">Py</span>Docs
              <span className="text-rose-400">.AI</span>
            </div>
            <div className="text-[11px] text-muted-foreground font-mono">
              rag.agent / v1.0
            </div>
          </div>
        </div>

        <StatusDot health={health} />

        <div className="ml-auto hidden md:flex items-center gap-2">
          <ChipStat
            icon={<FileText className="w-3.5 h-3.5" />}
            label="docs"
            value={status?.document_count ?? documents.length ?? "—"}
            tone="sky"
          />
          <ChipStat
            icon={<Layers className="w-3.5 h-3.5" />}
            label="chunks"
            value={status?.chunk_count ?? "—"}
            tone="peach"
          />
          <ChipStat
            icon={<Cpu className="w-3.5 h-3.5" />}
            label="embed"
            value={status?.embedding_backend ?? "—"}
            tone="sky"
            mono
          />
          <ChipStat
            icon={<Zap className="w-3.5 h-3.5" />}
            label="chat"
            value={status?.chat_backend ?? "—"}
            tone="peach"
            mono
          />
        </div>
      </header>

      {statusWarning && (
        <div className="bg-gradient-to-r from-amber-50 to-rose-50 border-b border-amber-200/70 text-amber-900 text-xs px-5 py-2 flex items-center gap-2">
          <AlertTriangle className="w-4 h-4" /> {statusWarning}
        </div>
      )}

      {/* Tabs */}
      <nav className="h-12 border-b border-sky-soft/60 bg-white/60 backdrop-blur-md flex items-center px-3 gap-1 shrink-0 relative z-10">
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
              <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-6 space-y-4">
                {messages.length === 0 && (
                  <div className="max-w-2xl mx-auto w-full">
                    <div
                      className="rounded-3xl p-8 border border-white/60 shadow-xl relative overflow-hidden"
                      style={{ background: "var(--gradient-mesh), white" }}
                    >
                      <div className="flex items-center gap-2 text-sky-600 text-xs font-mono font-semibold uppercase tracking-widest">
                        <Code2 className="w-4 h-4" /> python.knowledge_base
                      </div>
                      <h1 className="mt-3 font-display text-3xl font-bold tracking-tight">
                        Hỏi gì về <span className="text-sky-600">Python</span>?
                      </h1>
                      <p className="mt-2 text-sm text-muted-foreground max-w-md">
                        Agent truy xuất ngữ cảnh từ tài liệu Python chính thức và trả lời kèm nguồn rõ ràng.
                      </p>
                      <div className="mt-5 grid grid-cols-1 sm:grid-cols-2 gap-2">
                        {suggestions.map((s) => (
                          <button
                            key={s}
                            onClick={() => setInput(s)}
                            className="text-left text-xs px-3 py-2.5 rounded-xl bg-white/80 hover:bg-white border border-sky-soft/80 hover:border-sky-bright transition-all hover:-translate-y-0.5 hover:shadow-md group"
                          >
                            <span className="text-sky-500 font-mono mr-1.5 group-hover:text-rose-400 transition-colors">›</span>
                            {s}
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
                {messages.map((m, i) => (
                  <MessageRow key={i} m={m} />
                ))}
                {sending && (
                  <div className="flex items-center gap-2 text-sm text-sky-600">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span className="font-mono text-xs">agent.thinking...</span>
                  </div>
                )}
              </div>

              {/* Controls + input */}
              <div className="border-t border-sky-soft/60 bg-white/70 backdrop-blur-xl p-4 space-y-3">
                <div className="flex items-center gap-2 text-xs flex-wrap">
                  <ControlPill label="top_k">
                    <select
                      className="bg-transparent outline-none font-mono font-semibold text-sky-700"
                      value={topK}
                      onChange={(e) => setTopK(Number(e.target.value))}
                    >
                      {Array.from({ length: 10 }, (_, i) => i + 1).map((n) => (
                        <option key={n} value={n}>{n}</option>
                      ))}
                    </select>
                  </ControlPill>
                  <ControlPill label="topic" icon={<Filter className="w-3 h-3" />}>
                    <select
                      className="bg-transparent outline-none font-mono font-semibold text-sky-700 max-w-[160px]"
                      value={topicFilter}
                      onChange={(e) => setTopicFilter(e.target.value)}
                    >
                      <option value="">all</option>
                      {topics.map((t) => (
                        <option key={t} value={t}>{t}</option>
                      ))}
                    </select>
                  </ControlPill>
                  <button
                    onClick={clearChat}
                    className="ml-auto inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-rose-50 hover:bg-rose-100 text-rose-600 border border-rose-100 transition-colors"
                  >
                    <Trash2 className="w-3.5 h-3.5" /> Xoá
                  </button>
                </div>
                <div
                  className="flex items-end gap-2 p-2 rounded-2xl border-2 border-sky-soft/80 bg-white shadow-sm focus-within:border-sky-bright focus-within:shadow-lg transition-all"
                  style={{ boxShadow: input ? "var(--shadow-sky)" : undefined }}
                >
                  <textarea
                    rows={2}
                    placeholder="Hỏi gì đó về Python..."
                    className="flex-1 bg-transparent px-3 py-2 text-sm resize-none focus:outline-none placeholder:text-sky-400/60"
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
                    className="inline-flex items-center gap-1.5 px-4 py-2.5 rounded-xl text-white text-sm font-medium disabled:opacity-40 disabled:cursor-not-allowed transition-transform hover:-translate-y-0.5 active:translate-y-0"
                    style={{
                      background: "var(--gradient-hero)",
                      boxShadow: "var(--shadow-sky)",
                    }}
                  >
                    <Send className="w-4 h-4" /> Gửi
                  </button>
                </div>
              </div>
            </main>

            {/* Right sidebar: sources */}
            <aside className="w-[360px] border-l border-sky-soft/60 bg-white/60 backdrop-blur-md shrink-0 flex flex-col">
              <div className="px-4 py-3.5 border-b border-sky-soft/60 text-sm font-semibold font-display flex items-center gap-2">
                <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-sky-bright to-peach-bright grid place-items-center text-white">
                  <FileText className="w-3.5 h-3.5" />
                </div>
                Nguồn truy xuất
                {latestSources.length > 0 && (
                  <span className="ml-auto text-xs font-mono px-2 py-0.5 rounded-full bg-sky-soft text-sky-700">
                    {latestSources.length}
                  </span>
                )}
              </div>
              <div className="flex-1 overflow-y-auto p-3 space-y-2.5">
                {latestSources.length === 0 && (
                  <div className="text-xs text-muted-foreground p-4 text-center border border-dashed border-sky-soft rounded-xl bg-white/40">
                    Chưa có nguồn. Gửi câu hỏi để xem nguồn được agent truy xuất.
                  </div>
                )}
                {latestSources.map((s, i) => (
                  <SourceCard key={i} s={s} />
                ))}
              </div>
            </aside>
          </>
        )}

        {tab === "documents" && (
          <main className="flex-1 overflow-y-auto p-6">
            <div className="flex items-center gap-2 text-sm text-muted-foreground mb-4">
              <FileText className="w-4 h-4 text-sky-500" />
              <span className="font-mono font-semibold text-sky-700">{documents.length}</span> tài liệu
            </div>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {documents.map((d, i) => (
                <div
                  key={i}
                  className="group border border-sky-soft/80 rounded-2xl bg-white/80 hover:bg-white p-4 text-sm hover:shadow-lg hover:-translate-y-0.5 transition-all hover:border-sky-bright"
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="font-display font-semibold text-foreground leading-snug">{d.title ?? d.doc_id}</div>
                    {d.source_url && (
                      <a
                        href={d.source_url}
                        target="_blank"
                        rel="noreferrer"
                        className="shrink-0 w-7 h-7 rounded-lg bg-sky-soft text-sky-700 grid place-items-center hover:bg-sky-bright hover:text-white transition-colors"
                      >
                        <ExternalLink className="w-3.5 h-3.5" />
                      </a>
                    )}
                  </div>
                  <div className="mt-3 flex flex-wrap gap-1.5">
                    {d.topic && <Tag tone="sky">{d.topic}</Tag>}
                    {d.difficulty && <Tag tone="peach">{d.difficulty}</Tag>}
                  </div>
                  <div className="mt-3 flex gap-4 text-[11px] text-muted-foreground font-mono">
                    {typeof (d.character_count ?? d.char_count) === "number" && (
                      <span>{d.character_count ?? d.char_count} chars</span>
                    )}
                    {typeof d.chunk_count === "number" && <span>{d.chunk_count} chunks</span>}
                  </div>
                </div>
              ))}
              {documents.length === 0 && (
                <div className="col-span-full text-sm text-muted-foreground">Chưa có tài liệu.</div>
              )}
            </div>
          </main>
        )}

        {tab === "history" && (
          <main className="flex-1 overflow-y-auto p-6">
            <div className="flex items-center gap-2 text-sm text-muted-foreground mb-4">
              <HistoryIcon className="w-4 h-4 text-sky-500" />
              <span className="font-mono font-semibold text-sky-700">{history.length}</span> bản ghi
            </div>
            <div className="grid gap-3 max-w-3xl">
              {history.map((h, i) => (
                <div
                  key={i}
                  className="border border-sky-soft/80 rounded-2xl bg-white/80 p-4 text-sm hover:shadow-md transition-shadow"
                >
                  <div className="flex items-start gap-3">
                    <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-sky-soft to-peach-soft grid place-items-center text-sky-700 shrink-0">
                      <MessageSquare className="w-4 h-4" />
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="font-display font-semibold">{h.question ?? h.q ?? "(no question)"}</div>
                      {(h.answer ?? h.a) && (
                        <div className="mt-1.5 text-xs text-muted-foreground whitespace-pre-wrap line-clamp-3">
                          {h.answer ?? h.a}
                        </div>
                      )}
                      <div className="mt-2 text-[11px] text-sky-600/80 font-mono">
                        {h.created_at ?? h.timestamp ?? ""}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
              {history.length === 0 && (
                <div className="text-sm text-muted-foreground">Chưa có lịch sử.</div>
              )}
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
      <span className="inline-flex items-center gap-1.5 text-[11px] font-mono px-2.5 py-1 rounded-full bg-sky-soft text-sky-700">
        <Loader2 className="w-3 h-3 animate-spin" /> checking
      </span>
    );
  if (health === "ok")
    return (
      <span className="inline-flex items-center gap-1.5 text-[11px] font-mono px-2.5 py-1 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">
        <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
        online
      </span>
    );
  return (
    <span className="inline-flex items-center gap-1.5 text-[11px] font-mono px-2.5 py-1 rounded-full bg-rose-50 text-rose-700 border border-rose-200">
      <AlertTriangle className="w-3 h-3" /> offline
    </span>
  );
}

function ChipStat({
  icon, label, value, tone, mono,
}: { icon: React.ReactNode; label: string; value: React.ReactNode; tone: "sky" | "peach"; mono?: boolean }) {
  const bg = tone === "sky" ? "bg-sky-soft/70 text-sky-700" : "bg-rose-50 text-rose-600";
  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] ${bg}`}>
      {icon}
      <span className="opacity-70">{label}</span>
      <span className={`font-semibold ${mono ? "font-mono" : ""}`}>{String(value)}</span>
    </span>
  );
}

function ControlPill({
  label, icon, children,
}: { label: string; icon?: React.ReactNode; children: React.ReactNode }) {
  return (
    <label className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-sky-soft/60 border border-sky-soft text-sky-700">
      {icon}
      <span className="font-mono text-[11px] opacity-70">{label}</span>
      {children}
    </label>
  );
}

function TabBtn({
  active, onClick, icon, children,
}: { active: boolean; onClick: () => void; icon: React.ReactNode; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      className={`inline-flex items-center gap-1.5 px-4 py-1.5 rounded-full text-sm font-medium transition-all ${
        active
          ? "text-white shadow-md"
          : "text-muted-foreground hover:bg-sky-soft/60 hover:text-sky-700"
      }`}
      style={active ? { background: "var(--gradient-hero)" } : undefined}
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
        <div
          className="max-w-[80%] text-white text-sm rounded-2xl rounded-br-md px-4 py-2.5 whitespace-pre-wrap shadow-md"
          style={{ background: "var(--gradient-hero)" }}
        >
          {m.content}
        </div>
      </div>
    );
  }
  return (
    <div className="flex justify-start gap-2.5">
      <div
        className="w-8 h-8 rounded-full grid place-items-center text-white shrink-0 shadow-sm"
        style={{ background: "var(--gradient-hero)" }}
      >
        <Sparkles className="w-4 h-4" />
      </div>
      <div
        className={`max-w-[80%] text-sm rounded-2xl rounded-tl-md px-4 py-2.5 whitespace-pre-wrap border ${
          m.error
            ? "bg-rose-50 border-rose-200 text-rose-800"
            : "bg-white/90 backdrop-blur border-sky-soft"
        }`}
      >
        {m.content}
      </div>
    </div>
  );
}

function SourceCard({ s }: { s: Source }) {
  return (
    <div className="border border-sky-soft/80 rounded-xl p-3 text-xs bg-white/90 hover:bg-white hover:shadow-md hover:border-sky-bright/70 transition-all group">
      <div className="flex items-start justify-between gap-2">
        <div className="font-display font-semibold text-sm leading-snug">{s.title}</div>
        <span
          className="font-mono text-[10px] shrink-0 px-1.5 py-0.5 rounded-full text-white"
          style={{ background: "var(--gradient-hero)" }}
        >
          {s.score?.toFixed(2)}
        </span>
      </div>
      <div className="mt-2 flex flex-wrap gap-1">
        <Tag tone="sky">{s.topic}</Tag>
        <Tag tone="muted">#{s.chunk_index}</Tag>
        {s.difficulty && <Tag tone="peach">{s.difficulty}</Tag>}
      </div>
      {s.preview && (
        <div className="mt-2 text-[12px] text-muted-foreground line-clamp-4 leading-relaxed">
          {s.preview}
        </div>
      )}
      {s.source_url && (
        <a
          href={s.source_url}
          target="_blank"
          rel="noreferrer"
          className="mt-2 inline-flex items-center gap-1 text-sky-600 hover:text-rose-500 transition-colors font-medium"
        >
          Mở nguồn <ExternalLink className="w-3 h-3" />
        </a>
      )}
    </div>
  );
}

function Tag({ tone, children }: { tone: "sky" | "peach" | "muted"; children: React.ReactNode }) {
  const cls =
    tone === "sky"
      ? "bg-sky-soft text-sky-700"
      : tone === "peach"
      ? "bg-rose-50 text-rose-600"
      : "bg-muted text-muted-foreground";
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-mono ${cls}`}>
      {children}
    </span>
  );
}
