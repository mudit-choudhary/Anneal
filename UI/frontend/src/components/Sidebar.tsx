import { useState } from "react";
import { api } from "../api";
import type { ChatSummary } from "../types";

const prettify = (f: string) => f.replace(/\.txt$/, "").replace(/_/g, " ");

type Props = {
  papers: string[];
  selected: Set<string>;
  setSelected: (s: Set<string>) => void;
  onRefreshPapers: () => void;
  chats: ChatSummary[];
  chatId: string | null;
  onSelectChat: (id: string) => void;
  onNewChat: () => void;
  onChatsChanged: () => void;
};

export default function Sidebar(p: Props) {
  const [q, setQ] = useState("");
  const shown = p.papers.filter((x) => x.toLowerCase().includes(q.toLowerCase()));

  const toggle = (name: string) => {
    const next = new Set(p.selected);
    next.has(name) ? next.delete(name) : next.add(name);
    p.setSelected(next);
  };

  return (
    <aside className="sidebar">
      <section className="chats">
        <div className="row">
          <h2>Chats</h2>
          <button className="small" onClick={p.onNewChat}>+ New</button>
        </div>
        <div className="list">
          {p.chats.length === 0 && <span className="hint">No saved chats yet — every conversation is saved automatically.</span>}
          {p.chats.map((c) => (
            <div key={c.id} className={`chat-item ${c.id === p.chatId ? "active" : ""}`} onClick={() => p.onSelectChat(c.id)}>
              <span className="title" title={c.title}>{c.title}</span>
              <span className="meta">
                {c.n_messages} msg{c.embedded ? " · in memory" : ""}
              </span>
              <button
                className="x"
                title="Delete chat"
                onClick={async (e) => {
                  e.stopPropagation();
                  if (!confirm(`Delete "${c.title}"?`)) return;
                  await api.deleteChat(c.id);
                  if (c.id === p.chatId) p.onNewChat();
                  p.onChatsChanged();
                }}
              >
                ×
              </button>
            </div>
          ))}
        </div>
      </section>

      <section className="papers">
        <div className="row">
          <h2>Filter papers</h2>
          <span className="hint">{p.selected.size ? `${p.selected.size} selected` : "all"}</span>
        </div>
        <input type="search" placeholder="Search papers…" value={q} onChange={(e) => setQ(e.target.value)} />
        <div className="row">
          <button className="small" onClick={() => p.setSelected(new Set())}>Clear</button>
          <button className="small" onClick={p.onRefreshPapers}>Refresh</button>
        </div>
        <div className="list">
          {shown.length === 0 && <span className="hint">No embedded papers found.</span>}
          {shown.map((name) => (
            <label key={name}>
              <input type="checkbox" checked={p.selected.has(name)} onChange={() => toggle(name)} />
              {prettify(name)}
            </label>
          ))}
        </div>
        <span className="hint">Nothing selected = search all papers.</span>
      </section>
    </aside>
  );
}
