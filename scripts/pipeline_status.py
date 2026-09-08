"""One-screen view of pipeline progress: which services are running, how
many papers are in each registry status, what's on disk, and what the
embedding service reports as embedded.

    python scripts/pipeline_status.py
    watch -n 30 python scripts/pipeline_status.py     # live, during an overnight run
"""

import os
import sqlite3
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_DB = REPO_ROOT / "registry_manager" / "rag_registry.db"
PIDS = REPO_ROOT / "run" / "pids"
DIRS = {
    "raw_pdfs": REPO_ROOT / "data" / "raw_pdfs",
    "parsed": REPO_ROOT / "data" / "parsed",
    "processed": REPO_ROOT / "data" / "processed",
}
STATUSES = ["downloaded", "parsed", "processed", "embedded", "error"]


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main():
    print("services (run/pids):")
    if PIDS.exists() and any(PIDS.iterdir()):
        for pidfile in sorted(PIDS.iterdir()):
            pid = int(pidfile.read_text().strip() or 0)
            print(f"  {pidfile.stem:<10} {'running' if pid and alive(pid) else 'STOPPED':<8} pid {pid}")
    else:
        print("  none started via fresh_start.sh")

    print("\nregistry:")
    if REGISTRY_DB.exists():
        with sqlite3.connect(f"file:{REGISTRY_DB}?mode=ro", uri=True) as conn:
            rows = dict(conn.execute(
                "SELECT status, COUNT(*) FROM file_status_table GROUP BY status").fetchall())
            total = sum(rows.values())
            for s in STATUSES:
                bar = "█" * int(40 * rows.get(s, 0) / total) if total else ""
                print(f"  {s:<11} {rows.get(s, 0):>4}  {bar}")
            errors = conn.execute(
                "SELECT filename, last_error FROM file_status_table WHERE status='error' LIMIT 5").fetchall()
            for name, err in errors:
                print(f"    ! {name[:50]}: {(err or '')[:60]}")
    else:
        print("  no DB yet (registry never started)")

    print("\nfiles on disk:")
    for name, d in DIRS.items():
        n = len([p for p in d.glob("*") if p.is_file()]) if d.exists() else 0
        print(f"  {name:<10} {n:>4}")

    print("\nembedding service:")
    try:
        files = requests.get("http://127.0.0.1:4001/v1/papers", timeout=5).json()["papers"]
        print(f"  {len(files)} papers in the vector store")
    except requests.RequestException:
        print("  not reachable on :4001")


if __name__ == "__main__":
    main()
