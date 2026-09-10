import { useState } from "react";
import { api } from "../api";
import { dayLabel, timeLabel } from "../format";
import type { ChatSummary } from "../types";

/** Left rail: saved conversations, newest first, grouped by day.
 *  "Select" turns the list into checkboxes so a batch can go in one action —
 *  test chats and abandoned questions pile up faster than one-by-one deleting
 *  clears them. */
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
  const [picking, setPicking] = useState(false);
  const [picked, setPicked] = useState<Set<string>>(new Set());
  const [busy, setBusy] = useState(false);

  async function commitRename(id: string) {
    const title = draft.trim();
    setEditing(null);
    if (title) {
      await api.renameChat(id, title);
      onChatsChanged();
    }
  }

  function toggle(id: string) {
    setPicked((cur) => {
      const next = new Set(cur);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  }

  /** Select a whole day at once, or clear it if it is already fully selected. */
  function toggleDay(items: ChatSummary[]) {
    const all = items.every((c) => picked.has(c.id));
    setPicked((cur) => {
      const next = new Set(cur);
      for (const c of items) all ? next.delete(c.id) : next.add(c.id);
      return next;
    });
  }

  function stopPicking() {
    setPicking(false);
    setPicked(new Set());
  }

  async function deletePicked() {
    const ids = [...picked];
    if (!ids.length) return;
    if (!confirm(`Delete ${ids.length} chat${ids.length > 1 ? "s" : ""}? This cannot be undone.`)) return;
    setBusy(true);
    try {
      const r = await api.deleteChats({ ids });
      if (r.note) alert(r.note);
      if (chatId && picked.has(chatId)) onNewChat();
      stopPicking();
      onChatsChanged();
    } catch (e) {
      alert(`Could not delete: ${(e as Error).message}`);
    } finally {
      setBusy(false);
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
      {picking ? (
        <div className="row">
          <h2>{picked.size} selected</h2>
          <button className="small" disabled={busy || picked.size === 0} onClick={deletePicked}>
            {busy ? "Deleting…" : "Delete"}
          </button>
          <button className="small" disabled={busy}
                  onClick={() => setPicked(new Set(chats.map((c) => c.id)))}>All</button>
          <button className="small" disabled={busy} onClick={stopPicking}>Cancel</button>
        </div>
      ) : (
        <div className="row">
          <h2>Chats</h2>
          <span className="hint">{chats.length}</span>
          {chats.length > 1 && (
            <button className="small" title="Pick several to delete at once"
                    onClick={() => setPicking(true)}>Select</button>
          )}
          <button className="small" onClick={onNewChat}>+ New</button>
        </div>
      )}

      <div className="list">
        {chats.length === 0 && (
          <span className="hint">No chats yet — every conversation is saved automatically.</span>
        )}
        {groups.map((g) => (
          <div key={g.day} className="group">
            <div className="group-head">
              <span>{g.day}</span>
              {picking && (
                <button className="x" title={`Select everything under ${g.day}`}
                        onClick={() => toggleDay(g.items)}>
                  {g.items.every((c) => picked.has(c.id)) ? "none" : "all"}
                </button>
              )}
            </div>
            {g.items.map((c) => (
              <div key={c.id}
                   className={`chat-item ${c.id === chatId && !picking ? "active" : ""} ${picking && picked.has(c.id) ? "picked" : ""}`}
                   onClick={() => {
                     if (picking) toggle(c.id);
                     else if (editing !== c.id) onSelectChat(c.id);
                   }}>
                {picking && (
                  <input type="checkbox" className="pick" checked={picked.has(c.id)}
                         onChange={() => toggle(c.id)} onClick={(e) => e.stopPropagation()} />
                )}
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
                {!picking && (
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
                )}
              </div>
            ))}
          </div>
        ))}
      </div>

      <span className="hint">
        {picking
          ? "Tap a chat to select it. Deleting also removes it from searchable memory."
          : "Start a new chat when you change subject — follow-ups reuse the previous turns for context."}
      </span>
    </aside>
  );
}
