"""Status of the running retrieval evaluation.

    python evals/scripts/eval_status.py

Reads the checkpoint file and the run log. Safe to run at any time; it only
reads. Shows which of the 12 cells are done, what is in progress, an ETA from
the observed per-cell time, and the headline numbers so far.
"""

import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
RESULTS = REPO / "evals" / "Reports" / "rag_results.json"
LOG = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/rageval2.log")

PARSERS = ["oss_docling", "oss_pymupdf4llm", "current"]
CHUNKERS = ["fixed_token", "recursive_char", "semantic", "grain_growth"]
TOTAL = len(PARSERS) * len(CHUNKERS)


def running():
    try:
        out = subprocess.run(["pgrep", "-f", "scripts/run_rag_eval"],
                             capture_output=True, text=True).stdout.strip()
        return bool(out)
    except Exception:                                            # noqa: BLE001
        return False


def main():
    alive = running()
    done, cells = {}, {}
    if RESULTS.exists():
        cells = json.loads(RESULTS.read_text()).get("cells", {})
        done = {k: v for k, v in cells.items() if v.get("complete")}

    print(f"\n  status   {'RUNNING' if alive else 'NOT RUNNING'}")
    print(f"  cells    {len(done)} of {TOTAL} complete")

    if done:
        avg = sum(v["seconds"] for v in done.values()) / len(done) / 60
        left = (TOTAL - len(done)) * avg
        eta = time.strftime("%a %H:%M", time.localtime(time.time() + left * 60))
        print(f"  pace     {avg:.0f} min/cell")
        if alive:
            print(f"  eta      ~{left/60:.1f} h  (about {eta})")

    # what the log says about the cell in flight
    if LOG.exists():
        text = LOG.read_text(errors="ignore").splitlines()
        cur = next((l[3:] for l in reversed(text) if l.startswith("== ")), None)
        prog = next((l.strip() for l in reversed(text) if "questions (" in l), None)
        if cur and cur.replace("|", "|") not in done:
            print(f"\n  in flight  {cur}")
            if prog:
                print(f"             {prog}")
    else:
        print(f"\n  (log {LOG} not found; pass its path as an argument)")

    if done:
        print(f"\n  {'cell':<34}{'span@5':>8}{'span@4k':>9}{'mrr':>7}{'correct':>9}{'faith':>7}")
        for p in PARSERS:
            for c in CHUNKERS:
                v = done.get(f"{p}|{c}")
                if not v:
                    continue
                def g(k):
                    x = v.get(k)
                    return f"{x:.3f}" if isinstance(x, (int, float)) else "  —"
                print(f"  {p + '|' + c:<34}{g('span_hit@5'):>8}{g('span_hit@4000ch'):>9}"
                      f"{g('mrr'):>7}{g('correctness'):>9}{g('faithfulness'):>7}")

    failed = {k: v for k, v in cells.items() if v.get("error")}
    if failed:
        print("\n  failed cells:")
        for k, v in failed.items():
            print(f"    {k}: {v['error'][:80]}")

    if not alive and len(done) < TOTAL:
        print("\n  the run stopped early. resume with:")
        print("    python evals/scripts/run_rag_eval.py --device cuda")
        print("  completed cells are skipped, so it picks up where it left off.")
    print()


if __name__ == "__main__":
    main()
