"""Status of the round-2 retrieval run (3 parsers x 3 chunkers).

    python evals/scripts/eval_status.py [log]

Read-only; safe to run at any time. Shows which of the 9 cells are done, the
cell in flight with its question count, an ETA, and the headline numbers so
far. The log defaults to evals/Reports/round2_run.log.
"""

import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
RESULTS = REPO / "evals" / "Reports" / "rag_results_round2.json"
ROWS = REPO / "evals" / "Reports" / "rag_round2_rows"
LOG = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "evals" / "Reports" / "round2_run.log"

PARSERS = ["oss_docling", "oss_pymupdf4llm", "current"]
CHUNKERS = ["fixed_token", "recursive_char", "grain_growth"]
TOTAL = len(PARSERS) * len(CHUNKERS)
QUESTIONS = 400


def running():
    return subprocess.run(["pgrep", "-f", "run_rag_eval.py"], capture_output=True).returncode == 0


def main():
    alive = running()
    cells = json.loads(RESULTS.read_text()).get("cells", {}) if RESULTS.exists() else {}
    done = {k: v for k, v in cells.items() if v.get("complete")}
    log = LOG.read_text(errors="ignore").splitlines() if LOG.exists() else []

    print(f"\n  status   {'RUNNING' if alive else 'NOT RUNNING'}")
    print(f"  cells    {len(done)} of {TOTAL} complete")

    # the cell in flight: last "== parser|chunker" header in the log not yet complete
    current = next((l[3:].split(" (")[0].strip() for l in reversed(log)
                    if re.match(r"== \w+\|\w+", l)), None)
    in_flight, answered, rate, phase = None, 0, None, None
    if current and current not in done:
        in_flight = current
        parser, chunker = current.split("|")
        rows = ROWS / f"{parser}__{chunker}.json"
        try:
            answered = len(json.loads(rows.read_text())) if rows.exists() else 0
        except json.JSONDecodeError:
            answered = 0
        tail = log[len(log) - 1 - log[::-1].index(next(l for l in reversed(log) if l.startswith("== " + current))):]
        prog = [l for l in tail if "questions (" in l]
        if prog:
            m = re.search(r"(\d+)/\d+ questions \((\d+)s each", prog[-1])
            answered, rate = max(answered, int(m.group(1))), int(m.group(2))
            phase = "answering questions"
        else:
            idx = [l.strip() for l in tail if "indexed" in l]
            phase = idx[-1] if idx else "starting"

    per_cell = (sum(v["seconds"] for v in done.values()) / len(done)) if done else None
    if in_flight:
        print(f"\n  in flight  {in_flight}")
        print(f"             {phase}; {answered}/{QUESTIONS} questions answered"
              + (f", {rate}s each" if rate else ""))
    if alive:
        left_cells = TOTAL - len(done) - (1 if in_flight else 0)
        est_cell = per_cell or (25 * 60 + QUESTIONS * (rate or 25))
        left = left_cells * est_cell
        if in_flight:
            left += (QUESTIONS - answered) * (rate or 25) + (0 if rate else 25 * 60)
        eta = time.strftime("%a %H:%M", time.localtime(time.time() + left))
        print(f"  eta        ~{left / 3600:.1f} h for the remaining cells (about {eta})")

    if done:
        print(f"\n  {'cell':<32}{'span@4k':>9}{'near':>7}{'span@5':>8}{'near':>7}{'prec@5':>8}{'minutes':>9}")
        for p in PARSERS:
            for c in CHUNKERS:
                v = done.get(f"{p}|{c}")
                if not v:
                    continue
                g = lambda k: f"{v[k]:.3f}" if isinstance(v.get(k), (int, float)) else "  —"  # noqa: E731
                print(f"  {p + '|' + c:<32}{g('span_hit@4000ch'):>9}{g('span_hit_near@4000ch'):>7}"
                      f"{g('span_hit@5'):>8}{g('span_hit_near@5'):>7}{g('precision_near@5'):>8}"
                      f"{v['seconds'] / 60:>9.0f}")

    failed = {k: v for k, v in cells.items() if v.get("error")}
    for k, v in failed.items():
        print(f"\n  FAILED {k}: {v['error'][:80]}")

    if not alive and len(done) < TOTAL:
        print("\n  the run is not running. resume with:")
        print("    bash evals/scripts/round2_resume.sh 2>&1 | tee -a evals/Reports/round2_run.log")
        print("  finished cells and answered questions are skipped.")
    print()


if __name__ == "__main__":
    main()
