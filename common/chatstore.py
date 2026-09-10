"""Saved conversations (SQLite, data/app.db). Owned by the UI service.

Tables
    chats    id, title, created_at, updated_at, embedded (0/1)
    messages id, chat_id, role ('user'|'assistant'), content, sources (JSON), created_at
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from common.paths import APP_DB


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ChatStore:
    def __init__(self, db_path=APP_DB):
        self.db_path = str(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True) if hasattr(db_path, "parent") else None
        with self._conn() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS chats (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    embedded INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                    role TEXT NOT NULL, content TEXT NOT NULL, sources TEXT,
                    created_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, id);
            """)
            # Added after the table shipped: how long the answer took, so the
            # timing survives a reload. CREATE TABLE IF NOT EXISTS never alters
            # an existing file, so add it explicitly.
            columns = {r[1] for r in c.execute("PRAGMA table_info(messages)")}
            if "duration_ms" not in columns:
                c.execute("ALTER TABLE messages ADD COLUMN duration_ms INTEGER")

    def _conn(self):
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys = ON")
        return c

    # -- chats ------------------------------------------------------------
    def create(self, title: str) -> Dict[str, Any]:
        chat_id, ts = uuid.uuid4().hex[:12], _now()
        with self._conn() as c:
            c.execute("INSERT INTO chats VALUES (?, ?, ?, ?, 0)", (chat_id, title[:120], ts, ts))
        return {"id": chat_id, "title": title[:120], "created_at": ts, "updated_at": ts, "embedded": False}

    def list(self) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("""SELECT ch.*, COUNT(m.id) AS n_messages FROM chats ch
                                LEFT JOIN messages m ON m.chat_id = ch.id
                                GROUP BY ch.id ORDER BY ch.updated_at DESC""").fetchall()
        return [dict(r, embedded=bool(r["embedded"])) for r in rows]

    def get(self, chat_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            ch = c.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()
            if not ch:
                return None
            msgs = c.execute("SELECT * FROM messages WHERE chat_id = ? ORDER BY id", (chat_id,)).fetchall()
        return {**dict(ch), "embedded": bool(ch["embedded"]),
                "messages": [{**dict(m), "sources": json.loads(m["sources"]) if m["sources"] else None}
                             for m in msgs]}

    def rename(self, chat_id: str, title: str) -> bool:
        with self._conn() as c:
            n = c.execute("UPDATE chats SET title = ?, updated_at = ? WHERE id = ?",
                          (title[:120], _now(), chat_id)).rowcount
        return n > 0

    def delete(self, chat_id: str) -> bool:
        with self._conn() as c:
            n = c.execute("DELETE FROM chats WHERE id = ?", (chat_id,)).rowcount
        return n > 0

    def mark_embedded(self, chat_id: str, embedded: bool = True):
        with self._conn() as c:
            c.execute("UPDATE chats SET embedded = ? WHERE id = ?", (1 if embedded else 0, chat_id))

    # -- messages ---------------------------------------------------------
    def add_message(self, chat_id: str, role: str, content: str, sources=None,
                    duration_ms: Optional[int] = None) -> int:
        ts = _now()
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO messages (chat_id, role, content, sources, created_at, duration_ms) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (chat_id, role, content, json.dumps(sources) if sources else None, ts, duration_ms))
            c.execute("UPDATE chats SET updated_at = ?, embedded = 0 WHERE id = ?", (ts, chat_id))
            return cur.lastrowid

    def qa_pairs(self, chat_id: str) -> List[Dict[str, str]]:
        """Consecutive (user, assistant) pairs — what gets embedded."""
        chat = self.get(chat_id)
        if not chat:
            return []
        pairs, pending = [], None
        for m in chat["messages"]:
            if m["role"] == "user":
                pending = m["content"]
            elif m["role"] == "assistant" and pending is not None:
                pairs.append({"question": pending, "answer": m["content"]})
                pending = None
        return pairs
