import { useState } from "react";
import { api } from "../api";
import { dayLabel, timeLabel } from "../format";
import type { ChatSummary } from "../types";

/** Left rail: saved conversations, newest first, grouped by day. */
export default function ChatList({
  chats, chatId, onSelectChat, onNewChat, onChatsChanged,
}: {
  chats: ChatSummary[];
  chatId: string | null;
  onSelectChat: (id: string) => void;
  onNewChat: () => void;
  onChatsChanged: () => void;
}) {
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState("");

  async function commitRename(id: string) {
    const title = draft.trim();
    setEditing(null);
    if (title) {
      await api.renameChat(id, title);
      onChatsChanged();
    }
  }

  // group by the day of the last message, preserving the newest-first order
  const groups: { day: string; items: ChatSummary[] }[] = [];
  for (const c of chats) {
    const day = dayLabel(c.updated_at);
    const last = groups[groups.length - 1];
    if (last && last.day === day) last.items.push(c);
    else groups.push({ day, items: [c] });
  }

  return (
    <aside className="rail rail-left">
      <div className="row">
        <h2>Chats</h2>
        <span className="hint">{chats.length}</span>
        <button className="small" onClick={onNewChat}>+ New</button>
      </div>

      <div className="list">
        {chats.length === 0 && (
          <span className="hint">No chats yet — every conversation is saved automatically.</span>
        )}
        {groups.map((g) => (
          <div key={g.day} className="group">
            <div className="group-head">{g.day}</div>
            {g.items.map((c) => (
              <div key={c.id} className={`chat-item ${c.id === chatId ? "active" : ""}`}
                   onClick={() => editing !== c.id && onSelectChat(c.id)}>
                {editing === c.id ? (
                  <input
                    className="rename" autoFocus value={draft}
                    onChange={(e) => setDraft(e.target.value)}
                    onBlur={() => commitRename(c.id)}
                    onClick={(e) => e.stopPropagation()}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") commitRename(c.id);
                      if (e.key === "Escape") setEditing(null);
                    }}
                  />
                ) : (
                  <span className="title" title={c.title}>{c.title}</span>
                )}
                <span className="meta">
                  {timeLabel(c.updated_at)} · {c.n_messages} msg{c.embedded ? " · in memory" : ""}
                </span>
                <span className="actions">
                  <button
                    className="x" title="Rename"
                    onClick={(e) => { e.stopPropagation(); setDraft(c.title); setEditing(c.id); }}
                  >✎</button>
                  <button
                    className="x" title="Delete chat"
                    onClick={async (e) => {
                      e.stopPropagation();
                      if (!confirm(`Delete "${c.title}"?`)) return;
                      await api.deleteChat(c.id);
                      if (c.id === chatId) onNewChat();
                      onChatsChanged();
                    }}
                  >×</button>
                </span>
              </div>
            ))}
          </div>
        ))}
      </div>

      <span className="hint">Start a new chat when you change subject — follow-ups reuse the
        previous turns for context.</span>
    </aside>
  );
}
