"""Block until every registered paper has reached a terminal status
(embedded or error), then exit 0 — so a shutdown can be chained after an
unattended run:

    python scripts/wait_for_ingestion.py && scripts/stop_services.sh && systemctl poweroff

Exit codes: 0 all done · 1 a pipeline service died · 2 no progress for
--stall minutes (default 30). Chain with `;` instead of `&&` if you want the
machine to power off even on failure (logs stay in run/logs/).

    python scripts/wait_for_ingestion.py --once   # print status and exit
"""

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_DB = APP_ROOT / "registry_manager" / "rag_registry.db"
PIDS = APP_ROOT / "run" / "pids"
REQUIRED = ("registry", "parse", "embedding")
TERMINAL = ("embedded", "error")


def counts():
    if not REGISTRY_DB.exists():
        return {}
    with sqlite3.connect(f"file:{REGISTRY_DB}?mode=ro", uri=True) as conn:
        return dict(conn.execute(
            "SELECT status, COUNT(*) FROM file_status_table GROUP BY status").fetchall())


def dead_services():
    dead = []
    for name in REQUIRED:
        pidfile = PIDS / f"{name}.pid"
        try:
            os.kill(int(pidfile.read_text().strip()), 0)
        except (OSError, ValueError, FileNotFoundError):
            dead.append(name)
    return dead


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interval", type=int, default=30, help="seconds between checks")
    ap.add_argument("--stall", type=int, default=30, help="minutes without progress before giving up")
    ap.add_argument("--once", action="store_true", help="print status and exit")
    args = ap.parse_args()

    last_done, last_change = -1, time.time()
    while True:
        c = counts()
        total = sum(c.values())
        done = sum(c.get(s, 0) for s in TERMINAL)
        pending = total - done
        stamp = time.strftime("%H:%M:%S")
        print(f"[{stamp}] embedded {c.get('embedded', 0)}  error {c.get('error', 0)}  "
              f"pending {pending}  (of {total})", flush=True)

        if args.once:
            return 0
        if total and pending == 0:
            print("all papers reached a terminal status")
            return 0
        if dead := dead_services():
            print(f"service(s) not running: {', '.join(dead)} — see run/logs/")
            return 1
        if done != last_done:
            last_done, last_change = done, time.time()
        elif time.time() - last_change > args.stall * 60:
            print(f"no progress for {args.stall} minutes; giving up")
            return 2
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
