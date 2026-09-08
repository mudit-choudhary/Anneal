import { useEffect, useRef, useState } from "react";
import { api, streamQuery } from "../api";
import type { Message, Source } from "../types";
import Markdown from "./Markdown";
import Sources from "./Sources";

type Props = {
  chatId: string | null;
  filenames: string[] | null;
  backend: string;
  onChatStarted: (id: string) => void;
  onChatsChanged: () => void;
};

export default function Chat({ chatId, filenames, backend, onChatStarted, onChatsChanged }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [web, setWeb] = useState(false);
  const [busy, setBusy] = useState(false);
  const [highlight, setHighlight] = useState<{ idx: number; n: number } | null>(null);
  const [embedded, setEmbedded] = useState(false);
  const [notice, setNotice] = useState("");
  const bottom = useRef<HTMLDivElement>(null);
  const abort = useRef<AbortController | null>(null);

  // load a saved chat when selected
  useEffect(() => {
    if (!chatId) {
      setMessages([]);
      setEmbedded(false);
      return;
    }
    api
      .chat(chatId)
      .then((c) => {
        setMessages(c.messages.map((m) => ({ role: m.role, content: m.content, sources: m.sources ?? undefined })));
        setEmbedded(c.embedded);
      })
      .catch(() => setMessages([]));
  }, [chatId]);

  useEffect(() => bottom.current?.scrollIntoView({ behavior: "smooth" }), [messages]);

  async function ask() {
    const q = input.trim();
    if (!q || busy) return;
    setInput("");
    setBusy(true);
    setNotice("");
    setMessages((m) => [...m, { role: "user", content: q }, { role: "assistant", content: "", streaming: true }]);
    abort.current = new AbortController();
    const update = (fn: (a: Message) => Message) =>
      setMessages((m) => {
        const copy = [...m];
        copy[copy.length - 1] = fn(copy[copy.length - 1]);
        return copy;
      });
    try {
      await streamQuery(
        { query: q, filenames, web, chat_id: chatId },
        (ev) => {
          if (ev.type === "chat" && ev.chat_id !== chatId) onChatStarted(ev.chat_id);
          else if (ev.type === "sources") update((a) => ({ ...a, sources: ev.sources }));
          else if (ev.type === "warning") update((a) => ({ ...a, warnings: [...(a.warnings ?? []), ev.message] }));
          else if (ev.type === "delta") update((a) => ({ ...a, content: a.content + ev.text }));
          else if (ev.type === "error") update((a) => ({ ...a, error: ev.message, streaming: false }));
          else if (ev.type === "done") update((a) => ({ ...a, streaming: false }));
        },
        abort.current.signal,
      );
    } catch (e) {
      update((a) => ({ ...a, error: String((e as Error).message ?? e), streaming: false }));
    } finally {
      setBusy(false);
      setEmbedded(false);
      onChatsChanged();
    }
  }

  async function remember() {
    if (!chatId) return;
    try {
      const r = await api.embedChat(chatId);
      setEmbedded(true);
      setNotice(`Saved ${r.embedded} question/answer pair(s) to memory — future questions can retrieve them.`);
      onChatsChanged();
    } catch (e) {
      setNotice(`Could not save to memory: ${(e as Error).message}`);
    }
  }

  return (
    <div className="chat">
      <div className="messages">
        {messages.length === 0 && (
          <div className="msg bot">
            <div className="meta">assistant</div>
            <div className="bubble">
              Ask a question about your embedded research papers. Answers cite their sources — click a citation like{" "}
              <a className="cite">[1]</a> to see the excerpt it came from. Toggle <b>Web</b> to also fetch web pages for the
              question; Mermaid diagrams in answers render inline.
            </div>
          </div>
        )}
        {messages.map((m, i) =>
          m.role === "user" ? (
            <div key={i} className="msg user">
              <div className="meta">you</div>
              <div className="bubble">{m.content}</div>
            </div>
          ) : (
            <div key={i} className="msg bot">
              <div className="meta">assistant · {backend}</div>
              <div className="bubble">
                {m.warnings?.map((w, j) => (
                  <div key={j} className="warning">⚠ {w}</div>
                ))}
                <Markdown text={m.content} onCite={(n) => setHighlight({ idx: i, n })} />
                {m.streaming && <span className="cursor" />}
                {m.error && <div className="error">⚠ {m.error}</div>}
                {m.sources && <Sources sources={m.sources as Source[]} highlight={highlight?.idx === i ? highlight.n : null} />}
              </div>
            </div>
          ),
        )}
        <div ref={bottom} />
      </div>

      <div className="composer">
        <div className="box">
          <textarea
            rows={1}
            value={input}
            placeholder="Ask about your papers…  (Enter to send, Shift+Enter for newline)"
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                ask();
              }
            }}
          />
          <label className="toggle" title="Also fetch web pages and hand them to the model (not embedded)">
            <input type="checkbox" checked={web} onChange={(e) => setWeb(e.target.checked)} /> Web
          </label>
          {busy ? (
            <button className="send" onClick={() => abort.current?.abort()}>Stop</button>
          ) : (
            <button className="send" onClick={ask} disabled={!input.trim()}>Send</button>
          )}
        </div>
        <div className="note">
          {filenames ? `Searching ${filenames.length} selected paper(s). ` : ""}
          {chatId && messages.some((m) => m.role === "assistant" && !m.streaming && !m.error) && (
            <button className="small" onClick={remember} disabled={embedded}>
              {embedded ? "In memory ✓" : "Save chat to memory"}
            </button>
          )}
          {notice && <span> {notice}</span>}
        </div>
      </div>
    </div>
  );
}
