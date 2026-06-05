import { useEffect, useMemo, useRef, useState } from "react";
import {
  Send,
  Sparkles,
  Wallet,
  ArrowLeftRight,
  Phone,
  Zap,
  Receipt,
  Gift,
  ShieldCheck,
  MoreHorizontal,
  AlertCircle,
  ArrowLeft,
  User,
  Check,
} from "lucide-react";

const TRANSFER_KEYWORDS = ["chuyển tiền", "chuyen tien", "transfer", "chuyển", "chuyen"];
const isTransferIntent = (s: string) =>
  TRANSFER_KEYWORDS.some((k) => s.toLowerCase().includes(k));

const SUGGESTED_AMOUNTS = [50000, 100000, 200000, 500000, 1000000, 2000000];
const formatVnd = (n: number) =>
  n.toLocaleString("vi-VN") + "₫";

// Parse "4m", "500k", "1tr", "2 triệu", "2.000.000", "2000000"
function parseAmount(text: string): number {
  const s = text.toLowerCase().replace(/\s+/g, " ");
  // unit-suffixed: number + k / m / tr / triệu / ngàn / nghìn
  const m1 = s.match(/(\d+(?:[.,]\d+)?)\s*(k|m|tr|triệu|trieu|ngàn|ngan|nghìn|nghin)\b/);
  if (m1) {
    const n = parseFloat(m1[1].replace(",", "."));
    const u = m1[2];
    const mult =
      u === "k" || u.startsWith("ng") ? 1_000 :
      u === "m" || u === "tr" || u.startsWith("tri") ? 1_000_000 : 1;
    return Math.round(n * mult);
  }
  // grouped digits like 2.000.000 or 2,000,000 or plain 50000
  const m2 = s.match(/(\d[\d.,]{2,})/);
  if (m2) {
    const n = Number(m2[1].replace(/[.,]/g, ""));
    if (n >= 1000) return n;
  }
  return 0;
}

// Mock "danh bạ" của user (đã lưu)
const CONTACTS: Record<string, string> = {
  "0901234567": "Nguyễn Văn An",
  "0912345678": "Trần Thị Bình",
  "0987654321": "Lê Minh Châu",
  "0978123456": "Phạm Quốc Dũng",
  "0934567890": "Hoàng Thu Hà",
};

// Mock "database" toàn bộ người dùng MoMo trong hệ thống
const MOMO_USERS: Record<string, string> = {
  "0905111222": "Đặng Hoài Linh",
  "0966333444": "Vũ Khánh My",
  "0888555666": "Bùi Anh Tuấn",
  "0823777888": "Ngô Bảo Ngọc",
  "0772999000": "Cao Thành Đạt",
  "0356121314": "Đỗ Mai Phương",
  "0399151617": "Tạ Quang Huy",
  "0707181920": "Lý Hồng Nhung",
  "0902223344": "Nguyễn Hữu Phước",
  "0913445566": "Trần Quang Minh",
  "0988776655": "Lê Thị Hồng",
  "0977665544": "Phạm Văn Thắng",
  "0935112233": "Hoàng Anh Khoa",
  "0944556677": "Vũ Thị Lan",
  "0922334455": "Đào Quốc Bảo",
  "0967788990": "Trịnh Thu Trang",
  "0828112233": "Mai Đức Anh",
  "0793445566": "Phan Thị Yến",
  "0376778899": "Nguyễn Tuấn Kiệt",
  "0388990011": "Lương Thị Hằng",
  "0865443322": "Hồ Văn Sơn",
  "0796112334": "Đỗ Thanh Tùng",
  "0708994455": "Bùi Thị Hạnh",
  "0772334455": "Nguyễn Quỳnh Anh",
  "0918226677": "Tô Minh Quân",
  "0987001234": "Châu Bích Ngân",
  "0935667788": "Lâm Hữu Nghĩa",
  "0944112299": "Phùng Thị Diệu",
  "0926553311": "Đặng Văn Lộc",
  "0903556677": "Nguyễn Thị Kim Oanh",
  "0917889900": "Trần Minh Hiếu",
  "0964223344": "Vũ Phương Thảo",
  "0853223311": "Lê Hoàng Long",
  "0789445566": "Nguyễn Thị Mỹ Duyên",
  "0367445566": "Trương Văn Hùng",
  "0386778800": "Phạm Thị Thu Hằng",
  "0859223377": "Đoàn Quốc Việt",
  "0795332244": "Nguyễn Hoàng Nam",
  "0703887799": "Lê Thị Bích Vân",
  "0901002003": "Phan Minh Tâm",
  "0912004005": "Nguyễn Văn Trí",
  "0986007008": "Đặng Thu Hiền",
  "0976010011": "Trần Đăng Khoa",
  "0938013014": "Hồ Thị Thanh Mai",
  "0947016017": "Bùi Quang Vinh",
  "0928019020": "Đỗ Thị Ngọc Diệp",
  "0968022023": "Vũ Đình Phúc",
  "0824025026": "Nguyễn Lan Chi",
  "0792028029": "Phạm Hữu Lộc",
  "0378031032": "Lê Thị Tuyết Nhi",
  "0389034035": "Hoàng Văn Cường",
  "0866037038": "Trần Thị Diễm My",
  "0795040041": "Nguyễn Đức Thịnh",
  "0709043044": "Lý Thị Hồng Nhung",
  "0903046047": "Phan Quốc Anh",
  "0913049050": "Vương Thị Kiều Trinh",
  "0987052053": "Châu Văn Tiến",
  "0978055056": "Nguyễn Thị Phương Linh",
  "0934058059": "Đinh Hoàng Phi",
};

export type ContactLookup = {
  name: string;
  source: "contact" | "momo";
} | null;

// Extract a Vietnamese phone number (starts with 0, 9-11 digits)
function parsePhone(text: string): string {
  const cleaned = text.replace(/[\s.\-()]/g, "");
  const m = cleaned.match(/0\d{8,10}/);
  return m ? m[0] : "";
}

function lookupContact(phone: string): ContactLookup {
  if (!phone) return null;
  if (CONTACTS[phone]) return { name: CONTACTS[phone], source: "contact" };
  if (MOMO_USERS[phone]) return { name: MOMO_USERS[phone], source: "momo" };
  return null;
}

type Person = { phone: string; name: string; source: "contact" | "momo" };

const ALL_PEOPLE: Person[] = [
  ...Object.entries(CONTACTS).map(
    ([phone, name]) => ({ phone, name, source: "contact" as const }),
  ),
  ...Object.entries(MOMO_USERS).map(
    ([phone, name]) => ({ phone, name, source: "momo" as const }),
  ),
];

function normalize(s: string): string {
  return s
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/đ/g, "d");
}

// Find people whose name appears in the text (full name or last word, whole-word).
function findByName(text: string): Person[] {
  const t = normalize(text);
  return ALL_PEOPLE.filter((p) => {
    const n = normalize(p.name);
    if (t.includes(n)) return true;
    const last = n.split(" ").pop() ?? "";
    if (last.length < 2) return false;
    return new RegExp(`\\b${last}\\b`).test(t);
  });
}

const LOOKUP_HINT = /\b(ai|la ai|sdt|so dien thoai|so dt|ten|thong tin|cua ai|nguoi nao)\b/;
function isLookupQuery(text: string): boolean {
  const t = normalize(text);
  return LOOKUP_HINT.test(t) || t.trim().split(/\s+/).length <= 5;
}




type TransferAction = {
  phone: string;
  name: string;
  source: "contact" | "momo";
  amount: number;
};

type Message = {
  id: string;
  role: "user" | "bot";
  content: string;
  time: string;
  suggestions?: string[];
  transfer?: TransferAction;
};

const API_URL = "http://127.0.0.1:8000/api/chat";
const SESSION_KEY = "momo_chat_session_id";

const QUICK_ACTIONS = [
  { icon: ArrowLeftRight, label: "Chuyển tiền" },
  { icon: Phone, label: "Nạp điện thoại" },
  { icon: Zap, label: "Hoá đơn điện" },
  { icon: Receipt, label: "Lịch sử GD" },
  { icon: Gift, label: "Ưu đãi" },
  { icon: ShieldCheck, label: "Bảo mật" },
];

const now = () =>
  new Date().toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit" });

const INITIAL: Message[] = [
  {
    id: "m0",
    role: "bot",
    time: now(),
    content:
      "Xin chào! Mình là Mimi 🤖 — trợ lý ảo của Ví MoMo. Mình có thể giúp bạn chuyển tiền, thanh toán hoá đơn, kiểm tra số dư và nhiều hơn nữa.",
    suggestions: ["Kiểm tra số dư", "Chuyển tiền cho bạn", "Thanh toán hoá đơn điện"],
  },
];

export default function MomoChat() {
  const [messages, setMessages] = useState<Message[]>(INITIAL);
  const [input, setInput] = useState("");
  const [typing, setTyping] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string>("");
  const [showTransfer, setShowTransfer] = useState(false);
  const [prefill, setPrefill] = useState<{
    amount: number;
    recipient: string;
    phone: string;
  }>({ amount: 0, recipient: "", phone: "" });
  const [lastPerson, setLastPerson] = useState<Person | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const saved =
      typeof window !== "undefined" ? localStorage.getItem(SESSION_KEY) : null;
    if (saved) setSessionId(saved);
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, typing]);

  const pushBot = (content: string, extra?: Partial<Message>) =>
    setMessages((m) => [
      ...m,
      { id: crypto.randomUUID(), role: "bot", content, time: now(), ...extra },
    ]);

  const pushUser = (content: string) =>
    setMessages((m) => [
      ...m,
      { id: crypto.randomUUID(), role: "user", content, time: now() },
    ]);

  const openTransfer = (amount = 0, recipient = "", phone = "") => {
    setPrefill({ amount, recipient, phone });
    setShowTransfer(true);
  };

  const offerTransfer = (person: Person, amount: number) => {
    setLastPerson(person);
    pushBot(
      amount
        ? `Bạn muốn chuyển ${formatVnd(amount)} cho người này phải không? Nhấn vào nút bên dưới để tiếp tục thanh toán.`
        : `Bạn muốn chuyển tiền cho người này phải không? Nhấn vào nút bên dưới để tiếp tục.`,
      { transfer: { ...person, amount } },
    );
  };

  const send = async (text: string) => {
    const value = text.trim();
    if (!value || typing) return;

    const phoneInMsg = parsePhone(value);
    const nameMatches = findByName(value);

    // ----- TRANSFER INTENT -----
    if (isTransferIntent(value)) {
      pushUser(value);
      setInput("");
      const withoutPhone = phoneInMsg ? value.replace(phoneInMsg, " ") : value;
      const amount = parseAmount(withoutPhone);

      // 1) phone provided
      if (phoneInMsg) {
        const found = lookupContact(phoneInMsg);
        if (!found) {
          pushBot(
            `❌ Số điện thoại ${phoneInMsg} không tồn tại trong hệ thống MoMo. Bạn vui lòng kiểm tra lại nhé.`,
          );
          return;
        }
        offerTransfer({ phone: phoneInMsg, ...found }, amount);
        return;
      }

      // 2) name provided
      if (nameMatches.length === 1) {
        offerTransfer(nameMatches[0], amount);
        return;
      }
      if (nameMatches.length > 1) {
        pushBot(
          `Mình tìm thấy ${nameMatches.length} người trùng tên: ${nameMatches
            .map((p) => `${p.name} (${p.phone})`)
            .join(", ")}. Bạn cho mình biết SĐT cụ thể nhé.`,
        );
        return;
      }

      // 3) fall back to last referenced person in conversation
      if (lastPerson) {
        offerTransfer(lastPerson, amount);
        return;
      }

      pushBot(
        amount
          ? `Mình đã ghi nhận số tiền ${formatVnd(amount)}. Bạn muốn chuyển cho ai? (nhập SĐT hoặc tên)`
          : "Mình mở màn hình chuyển tiền giúp bạn. Nhập số điện thoại và số tiền nhé.",
      );
      openTransfer(amount, "", "");
      return;
    }

    // ----- LOOKUP: phone in message -----
    if (phoneInMsg) {
      pushUser(value);
      setInput("");
      const found = lookupContact(phoneInMsg);
      if (!found) {
        pushBot(
          `❌ Số ${phoneInMsg} không tồn tại trong hệ thống MoMo.`,
        );
        return;
      }
      const person: Person = { phone: phoneInMsg, ...found };
      setLastPerson(person);
      const src = found.source === "contact" ? "danh bạ của bạn" : "người dùng MoMo";
      pushBot(
        `Số ${phoneInMsg} là của ${found.name} (${src}). Bạn muốn chuyển tiền cho ${found.name} không?`,
        { transfer: { ...person, amount: 0 } },
      );
      return;
    }

    // ----- LOOKUP: name in message -----
    if (nameMatches.length > 0 && isLookupQuery(value)) {
      pushUser(value);
      setInput("");
      if (nameMatches.length === 1) {
        const p = nameMatches[0];
        setLastPerson(p);
        const src = p.source === "contact" ? "danh bạ của bạn" : "người dùng MoMo";
        pushBot(
          `${p.name} có số điện thoại ${p.phone} (${src}). Bạn muốn chuyển tiền cho ${p.name} không?`,
          { transfer: { ...p, amount: 0 } },
        );
      } else {
        pushBot(
          `Mình tìm thấy ${nameMatches.length} người:\n${nameMatches
            .map((p) => `• ${p.name} — ${p.phone}`)
            .join("\n")}`,
        );
      }
      return;
    }



    setError(null);
    pushUser(value);
    setInput("");
    setTyping(true);

    try {
      const sid = sessionId || crypto.randomUUID();
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: value, session_id: sid }),
      });

      if (!res.ok) {
        throw new Error(`Lỗi máy chủ (${res.status})`);
      }

      const data = (await res.json()) as { reply: string; session_id: string };
      setSessionId(data.session_id);
      localStorage.setItem(SESSION_KEY, data.session_id);
      pushBot(data.reply);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Không thể kết nối đến máy chủ. Vui lòng thử lại.",
      );
    } finally {
      setTyping(false);
    }
  };

  const handleAction = (text: string) => {
    if (isTransferIntent(text)) {
      openTransfer(0, "", "");
      return;
    }
    send(text);
  };

  const handleTransferConfirm = (amount: number, recipient: string, phone: string) => {
    setShowTransfer(false);
    setPrefill({ amount: 0, recipient: "", phone: "" });
    pushBot(
      `✅ Chuyển tiền thành công!\nSố tiền: ${formatVnd(amount)}\nNgười nhận: ${recipient}${phone ? ` (${phone})` : ""}\nMã GD: ${Math.random().toString(36).slice(2, 10).toUpperCase()}`,
    );
  };

  const handleTransferCancel = () => {
    setShowTransfer(false);
    setPrefill({ amount: 0, recipient: "", phone: "" });
    pushBot("Bạn đã huỷ giao dịch chuyển tiền. Mình vẫn ở đây nếu cần hỗ trợ nhé!");
  };

  return (
    <div className="min-h-screen w-full" style={{ background: "var(--gradient-soft)" }}>
      <div className="relative mx-auto flex min-h-screen max-w-md flex-col bg-background shadow-2xl md:my-6 md:min-h-[92vh] md:max-w-lg md:rounded-3xl md:overflow-hidden">

        {/* Header */}
        <header
          className="relative px-5 pt-6 pb-5 text-primary-foreground"
          style={{ background: "var(--gradient-momo)" }}
        >
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-white/15 backdrop-blur-sm">
                <Wallet className="h-6 w-6" />
              </div>
              <span className="absolute -bottom-0.5 -right-0.5 h-3.5 w-3.5 rounded-full border-2 border-[oklch(0.55_0.25_340)] bg-green-400" />
            </div>
            <div className="flex-1">
              <h1 className="text-base font-semibold leading-tight">
                Mimi — Trợ lý MoMo
              </h1>
              <p className="text-xs text-white/80 flex items-center gap-1">
                <Sparkles className="h-3 w-3" /> Đang hoạt động · phản hồi 24/7
              </p>
            </div>
            <button className="rounded-full bg-white/15 p-2 backdrop-blur-sm transition hover:bg-white/25">
              <MoreHorizontal className="h-5 w-5" />
            </button>
          </div>

          {/* Quick actions */}
          <div className="mt-5 grid grid-cols-6 gap-1.5">
            {QUICK_ACTIONS.map(({ icon: Icon, label }) => (
              <button
                key={label}
                onClick={() => handleAction(label)}
                className="group flex flex-col items-center gap-1 rounded-xl bg-white/10 px-1 py-2 text-[10px] font-medium text-white/90 backdrop-blur-sm transition hover:bg-white/20"
              >
                <Icon className="h-4 w-4" />
                <span className="leading-tight text-center">{label}</span>
              </button>
            ))}
          </div>
        </header>

        {/* Messages */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-5 space-y-4">
          {messages.map((m) => (
            <MessageBubble
              key={m.id}
              message={m}
              onSuggestion={handleAction}
              onTransfer={(t) => openTransfer(t.amount, t.name, t.phone)}
            />
          ))}
          {typing && <TypingIndicator />}
        </div>

        {/* Error banner */}
        {error && (
          <div className="mx-4 mb-2 flex items-center gap-2 rounded-xl bg-destructive/10 px-3 py-2.5 text-xs text-destructive">
            <AlertCircle className="h-4 w-4 shrink-0" />
            <span className="flex-1">{error}</span>
            <button
              onClick={() => setError(null)}
              className="shrink-0 rounded-md px-2 py-1 font-medium hover:bg-destructive/20"
            >
              Đóng
            </button>
          </div>
        )}

        {/* Composer */}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            send(input);
          }}
          className="border-t border-border bg-card px-4 py-3"
        >
          <div className="flex items-center gap-2 rounded-2xl bg-muted px-4 py-2.5 focus-within:ring-2 focus-within:ring-ring transition">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Nhập tin nhắn cho Mimi..."
              disabled={typing}
              className="flex-1 bg-transparent text-sm text-foreground placeholder:text-muted-foreground focus:outline-none disabled:opacity-50"
            />
            <button
              type="submit"
              disabled={!input.trim() || typing}
              className="flex h-9 w-9 items-center justify-center rounded-full text-primary-foreground transition disabled:opacity-40"
              style={{
                background: "var(--gradient-momo)",
                boxShadow: "var(--shadow-momo)",
              }}
              aria-label="Gửi"
            >
              <Send className="h-4 w-4" />
            </button>
          </div>
          <p className="mt-2 text-center text-[10px] text-muted-foreground">
            Mimi có thể mắc lỗi. Vui lòng kiểm tra thông tin giao dịch quan trọng.
          </p>
        </form>

        {showTransfer && (
          <TransferScreen
            initialAmount={prefill.amount}
            initialRecipient={prefill.recipient}
            initialPhone={prefill.phone}
            onClose={handleTransferCancel}
            onConfirm={handleTransferConfirm}
          />
        )}
      </div>
    </div>
  );
}

function TransferScreen({
  onClose,
  onConfirm,
  initialAmount = 0,
  initialRecipient = "",
  initialPhone = "",
}: {
  onClose: () => void;
  onConfirm: (amount: number, recipient: string, phone: string) => void;
  initialAmount?: number;
  initialRecipient?: string;
  initialPhone?: string;
}) {
  const [phone, setPhone] = useState(initialPhone);
  const [raw, setRaw] = useState(initialAmount ? String(initialAmount) : "");
  const amount = useMemo(() => Number(raw.replace(/\D/g, "")) || 0, [raw]);
  const display = amount ? amount.toLocaleString("vi-VN") : "";

  const resolvedName = useMemo(() => lookupContact(phone.trim()), [phone]);
  const phoneEntered = phone.trim().length >= 9;
  const phoneInvalid = phoneEntered && !resolvedName;
  const recipientName = resolvedName?.name ?? initialRecipient;
  const sourceLabel =
    resolvedName?.source === "contact"
      ? "Trong danh bạ của bạn"
      : resolvedName?.source === "momo"
        ? "Người dùng MoMo đã xác thực"
        : "";
  const canSubmit = amount >= 10000 && !!resolvedName;

  return (
    <div className="absolute inset-0 z-30 flex flex-col bg-background animate-in slide-in-from-right duration-300 md:rounded-3xl md:overflow-hidden">
      <header
        className="flex items-center gap-3 px-4 pt-6 pb-5 text-primary-foreground"
        style={{ background: "var(--gradient-momo)" }}
      >
        <button
          onClick={onClose}
          className="rounded-full bg-white/15 p-2 backdrop-blur-sm transition hover:bg-white/25"
          aria-label="Quay lại"
        >
          <ArrowLeft className="h-5 w-5" />
        </button>
        <div className="flex-1">
          <h2 className="text-base font-semibold leading-tight">Chuyển tiền</h2>
          <p className="text-xs text-white/80">Số dư khả dụng: 2.480.000₫</p>
        </div>
        <ArrowLeftRight className="h-5 w-5" />
      </header>

      <div className="flex-1 overflow-y-auto px-5 py-6 space-y-6">
        {/* Recipient */}
        <div>
          <label className="text-xs font-medium text-muted-foreground">
            Số điện thoại người nhận
          </label>
          <div
            className={`mt-2 flex items-center gap-3 rounded-2xl bg-muted px-4 py-3 ring-1 transition ${
              phoneInvalid
                ? "ring-destructive"
                : resolvedName
                  ? "ring-primary/40"
                  : "ring-transparent focus-within:ring-ring"
            }`}
          >
            <Phone className="h-5 w-5 text-muted-foreground" />
            <input
              value={phone}
              onChange={(e) => setPhone(e.target.value.replace(/[^\d]/g, ""))}
              inputMode="tel"
              placeholder="Ví dụ: 0901234567"
              className="flex-1 bg-transparent text-sm text-foreground placeholder:text-muted-foreground focus:outline-none"
            />
          </div>
          {resolvedName && (
            <div className="mt-2 flex items-center gap-2 rounded-xl bg-primary/10 px-3 py-2">
              <div
                className="flex h-7 w-7 items-center justify-center rounded-full text-primary-foreground text-xs font-semibold"
                style={{ background: "var(--gradient-momo)" }}
              >
                {recipientName.charAt(0)}
              </div>
              <div className="flex-1">
                <p className="text-sm font-semibold text-foreground leading-tight">
                  {recipientName}
                </p>
                <p className="text-[11px] text-muted-foreground">{sourceLabel}</p>
              </div>
              <Check className="h-4 w-4 text-primary" />
            </div>
          )}
          {phoneInvalid && (
            <p className="mt-2 flex items-center gap-1 text-[11px] text-destructive">
              <AlertCircle className="h-3 w-3" />
              Số {phone} không tồn tại trong danh bạ MoMo
            </p>
          )}
        </div>


        {/* Amount */}
        <div>
          <label className="text-xs font-medium text-muted-foreground">
            Số tiền
          </label>
          <div className="mt-2 rounded-2xl border-2 border-primary/20 bg-card px-4 py-5 text-center focus-within:border-primary transition">
            <div className="flex items-baseline justify-center gap-1">
              <input
                inputMode="numeric"
                value={display}
                onChange={(e) => setRaw(e.target.value)}
                placeholder="0"
                className="w-full bg-transparent text-center text-3xl font-bold tracking-tight text-foreground placeholder:text-muted-foreground/40 focus:outline-none"
              />
              <span className="text-xl font-semibold text-primary">₫</span>
            </div>
            {amount > 0 && amount < 10000 && (
              <p className="mt-2 text-[11px] text-destructive">
                Số tiền tối thiểu là 10.000₫
              </p>
            )}
          </div>

          {/* Suggested amounts */}
          <div className="mt-3">
            <p className="text-[11px] text-muted-foreground mb-2">Gợi ý số tiền</p>
            <div className="grid grid-cols-3 gap-2">
              {SUGGESTED_AMOUNTS.map((v) => {
                const active = amount === v;
                return (
                  <button
                    key={v}
                    type="button"
                    onClick={() => setRaw(String(v))}
                    className={`rounded-xl px-2 py-2.5 text-xs font-semibold transition border ${
                      active
                        ? "border-primary text-primary-foreground"
                        : "border-border bg-card text-foreground hover:border-primary/50"
                    }`}
                    style={
                      active
                        ? { background: "var(--gradient-momo)" }
                        : undefined
                    }
                  >
                    {formatVnd(v)}
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Note */}
        <div>
          <label className="text-xs font-medium text-muted-foreground">
            Lời nhắn (tuỳ chọn)
          </label>
          <div className="mt-2 rounded-2xl bg-muted px-4 py-3 focus-within:ring-2 focus-within:ring-ring">
            <input
              placeholder="Ví dụ: Trả tiền cà phê ☕"
              className="w-full bg-transparent text-sm text-foreground placeholder:text-muted-foreground focus:outline-none"
            />
          </div>
        </div>
      </div>

      {/* Confirm */}
      <div className="border-t border-border bg-card px-4 py-4">
        <button
          disabled={!canSubmit}
          onClick={() => onConfirm(amount, recipientName, phone.trim())}
          className="flex w-full items-center justify-center gap-2 rounded-2xl py-3.5 text-sm font-semibold text-primary-foreground transition disabled:opacity-40"
          style={{
            background: "var(--gradient-momo)",
            boxShadow: "var(--shadow-momo)",
          }}
        >
          <Check className="h-4 w-4" />
          {amount > 0 ? `Chuyển ${formatVnd(amount)}` : "Xác nhận chuyển tiền"}
        </button>
      </div>
    </div>
  );
}


function MessageBubble({
  message,
  onSuggestion,
  onTransfer,
}: {
  message: Message;
  onSuggestion: (s: string) => void;
  onTransfer: (t: TransferAction) => void;
}) {
  const isUser = message.role === "user";
  return (
    <div
      className={`flex ${isUser ? "justify-end" : "justify-start"} animate-in fade-in slide-in-from-bottom-2 duration-300`}
    >
      <div className={`flex max-w-[85%] gap-2 ${isUser ? "flex-row-reverse" : ""}`}>
        {!isUser && (
          <div
            className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-primary-foreground"
            style={{ background: "var(--gradient-momo)" }}
          >
            <Sparkles className="h-4 w-4" />
          </div>
        )}
        <div className="space-y-1.5">
          <div
            className={`px-4 py-2.5 text-sm leading-relaxed whitespace-pre-line ${
              isUser
                ? "rounded-2xl rounded-br-md text-primary-foreground"
                : "rounded-2xl rounded-bl-md bg-bot-bubble text-bot-bubble-foreground"
            }`}
            style={
              isUser
                ? { background: "var(--gradient-momo)", boxShadow: "var(--shadow-bubble)" }
                : { boxShadow: "var(--shadow-bubble)" }
            }
          >
            {message.content}
          </div>
          {message.transfer && (
            <button
              onClick={() => onTransfer(message.transfer!)}
              className="flex w-full items-center gap-3 rounded-2xl border border-primary/30 bg-card px-3 py-2.5 text-left transition hover:border-primary hover:shadow-md"
              style={{ boxShadow: "var(--shadow-bubble)" }}
            >
              <div
                className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full text-primary-foreground text-sm font-bold"
                style={{ background: "var(--gradient-momo)" }}
              >
                {message.transfer.name.charAt(0)}
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-semibold text-foreground truncate">
                  {message.transfer.name}
                </p>
                <p className="text-[11px] text-muted-foreground">
                  {message.transfer.phone} ·{" "}
                  {message.transfer.source === "contact"
                    ? "Danh bạ"
                    : "Người dùng MoMo"}
                </p>
              </div>
              <span
                className="shrink-0 rounded-full px-3 py-1 text-[11px] font-semibold text-primary-foreground"
                style={{ background: "var(--gradient-momo)" }}
              >
                Chuyển tiền
              </span>
            </button>
          )}
          {message.suggestions && (
            <div className="flex flex-wrap gap-1.5">
              {message.suggestions.map((s) => (
                <button
                  key={s}
                  onClick={() => onSuggestion(s)}
                  className="rounded-full border border-primary/30 bg-accent px-3 py-1 text-xs font-medium text-accent-foreground transition hover:bg-primary hover:text-primary-foreground"
                >
                  {s}
                </button>
              ))}
            </div>
          )}
          <p className={`text-[10px] text-muted-foreground ${isUser ? "text-right" : ""}`}>
            {message.time}
          </p>
        </div>
      </div>
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="flex justify-start">
      <div className="flex items-center gap-2">
        <div
          className="flex h-8 w-8 items-center justify-center rounded-full text-primary-foreground"
          style={{ background: "var(--gradient-momo)" }}
        >
          <Sparkles className="h-4 w-4" />
        </div>
        <div
          className="flex items-center gap-1 rounded-2xl rounded-bl-md bg-bot-bubble px-4 py-3"
          style={{ boxShadow: "var(--shadow-bubble)" }}
        >
          <span
            className="h-2 w-2 animate-bounce rounded-full bg-primary"
            style={{ animationDelay: "0ms" }}
          />
          <span
            className="h-2 w-2 animate-bounce rounded-full bg-primary"
            style={{ animationDelay: "150ms" }}
          />
          <span
            className="h-2 w-2 animate-bounce rounded-full bg-primary"
            style={{ animationDelay: "300ms" }}
          />
        </div>
      </div>
    </div>
  );
}
