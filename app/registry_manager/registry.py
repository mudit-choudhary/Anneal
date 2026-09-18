"""SQLite-backed paper registry: one row per paper, keyed by the PDF stem.

Status chain: downloaded → parsed → processed → embedded, or error.
Timestamps are stamped per stage so throughput/ETA can be derived.
"""

import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.paths import REGISTRY_DB  # noqa: E402

STATUSES = ("downloaded", "parsed", "processed", "embedded", "error")
STAGE_AT = {"parsed": "parsed_at", "processed": "processed_at", "embedded": "embedded_at"}
COLUMNS = ("filename", "domain", "status", "error_count", "last_error", "published_at",
           "downloaded_at", "parsed_at", "processed_at", "embedded_at")


def _ts(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class FileRegistry:
    def __init__(self, db_path=None):
        self.db_path = str(db_path or REGISTRY_DB)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        # CREATE TABLE IF NOT EXISTS never alters an existing file: schema
        # changes require deleting the DB (scripts/reset_ingestion.py does).
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS file_status_table (
                    filename      TEXT PRIMARY KEY,
                    domain        TEXT,
                    status        TEXT CHECK(status IN ('downloaded', 'parsed', 'processed', 'embedded', 'error')),
                    error_count   INTEGER DEFAULT 0,
                    last_error    TEXT,
                    published_at  TIMESTAMP,
                    downloaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    parsed_at     TIMESTAMP,
                    processed_at  TIMESTAMP,
                    embedded_at   TIMESTAMP
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_domain_date ON file_status_table (domain, published_at);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON file_status_table (status);")

    # -- writes -----------------------------------------------------------
    def update_status(self, filename: str, status: str, domain: Optional[str] = None,
                      published_at=None, error_msg: Optional[str] = None,
                      force: bool = False) -> bool:
        if status not in STATUSES:
            return False
        if isinstance(published_at, datetime):
            published_at = published_at.isoformat()
        try:
            with self._conn() as conn:
                if status == "downloaded":
                    # Idempotent registration; `force` resets an existing row
                    # (used to retry errors / re-run a paper from scratch).
                    conn.execute("""INSERT OR IGNORE INTO file_status_table
                                    (filename, domain, status, published_at, error_count)
                                    VALUES (?, ?, 'downloaded', ?, 0)""",
                                 (filename, domain, published_at))
                    if force:
                        conn.execute("""UPDATE file_status_table SET status='downloaded', error_count=0,
                                        last_error=NULL, parsed_at=NULL, processed_at=NULL, embedded_at=NULL
                                        WHERE filename = ?""", (filename,))
                elif status == "error":
                    conn.execute("""INSERT OR IGNORE INTO file_status_table (filename, domain, status, error_count)
                                    VALUES (?, ?, 'error', 0)""", (filename, domain))
                    conn.execute("""UPDATE file_status_table
                                    SET status='error', error_count=error_count + 1, last_error=?
                                    WHERE filename = ?""", (error_msg or "unknown error", filename))
                else:
                    n = conn.execute(
                        f"UPDATE file_status_table SET status=?, {STAGE_AT[status]}=CURRENT_TIMESTAMP, "
                        f"error_count=0, last_error=NULL WHERE filename = ?",
                        (status, filename)).rowcount
                    if n == 0:
                        return False
                conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"DB error: {e}")
            return False

    # -- reads ------------------------------------------------------------
    def get_paper(self, filename: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM file_status_table WHERE filename = ?", (filename,)).fetchone()
        return dict(row) if row else None

    def get_status(self, filename: str) -> Optional[str]:
        paper = self.get_paper(filename)
        return paper["status"] if paper else None

    def list_papers(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            if status:
                rows = conn.execute("SELECT * FROM file_status_table WHERE status = ? ORDER BY downloaded_at",
                                    (status,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM file_status_table ORDER BY downloaded_at").fetchall()
        return [dict(r) for r in rows]

    def get_last_domain_date(self, domain: str) -> Optional[str]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT MAX(published_at) FROM file_status_table WHERE domain = ? AND status != 'error'",
                (domain,)).fetchone()
        return row[0] if row and row[0] else None

    def stats(self) -> Dict[str, Any]:
        """Counts per status, per-stage timings, and recent errors.

        Stage timings are **medians, not means**. The gap from `downloaded_at`
        to `parsed_at` is mostly *queue wait* when papers are registered in
        bulk — one sample can be 28 hours while the median is 44 seconds — and
        a mean turns that into a nonsense "average parse time".
        """
        papers = self.list_papers()
        counts = {s: 0 for s in STATUSES}
        stage_samples = {"parse": [], "process": [], "embed": [], "total": []}
        embedded_at = []
        for p in papers:
            counts[p["status"]] = counts.get(p["status"], 0) + 1
            d, pa, pr, em = (_ts(p[c]) for c in ("downloaded_at", "parsed_at", "processed_at", "embedded_at"))
            if d and pa:
                stage_samples["parse"].append((pa - d).total_seconds())
            if pa and pr:
                stage_samples["process"].append((pr - pa).total_seconds())
            if pr and em:
                stage_samples["embed"].append((em - pr).total_seconds())
            if d and em:
                stage_samples["total"].append((em - d).total_seconds())
            if em:
                embedded_at.append(em.isoformat())

        def median(values):
            if not values:
                return None
            ordered = sorted(values)
            mid = len(ordered) // 2
            return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2

        errors = [{"filename": p["filename"], "error_count": p["error_count"], "last_error": p["last_error"]}
                  for p in papers if p["status"] == "error"]
        return {"counts": counts, "total": len(papers),
                "stage_seconds": {k: median(v) for k, v in stage_samples.items()},
                "recent_embedded_at": sorted(embedded_at)[-40:], "errors": errors}
