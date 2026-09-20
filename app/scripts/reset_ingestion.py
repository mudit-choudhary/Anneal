"""Reset the pipeline so every downloaded paper is re-parsed, re-chunked and
re-embedded — needed after any change to parsing, chunking, the embedding
model, or the registry schema.

    python scripts/reset_ingestion.py          # dry run: shows what would happen
    python scripts/reset_ingestion.py --yes    # do it

STOP the registry, parse_manager and embedding_manager services first
(scripts/ops.py fresh-start does all of this in order). Raw PDFs in
data/raw_pdfs/ are never touched.
"""

import shutil
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.paths import DATA_HOME, PARSED_DIR, PROCESSED_DIR, REGISTRY_DB, VECTOR_DB  # noqa: E402


def reset(apply: bool, quiet: bool = False):
    out = (lambda *a: None) if quiet else print
    out("Dry run — pass --yes to apply.\n" if not apply else "Applying reset.\n")

    for d in (PARSED_DIR, PROCESSED_DIR):
        files = [p for p in d.glob("*") if p.is_file()] if d.exists() else []
        out(f"{d.relative_to(DATA_HOME)}: delete {len(files)} files")
        if apply:
            for p in files:
                p.unlink()

    if VECTOR_DB.exists():
        size = sum(p.stat().st_size for p in VECTOR_DB.rglob("*") if p.is_file()) / 1e6
        out(f"vector_db/: delete ({size:.0f} MB)")
        if apply:
            shutil.rmtree(VECTOR_DB)
    else:
        out("vector_db/: not present")

    # The registry DB is deleted outright rather than reset: the registry
    # recreates it with the current schema on startup, and schema changes
    # (CHECK constraints, new columns) never apply to an existing file.
    if REGISTRY_DB.exists():
        try:
            with sqlite3.connect(REGISTRY_DB) as conn:
                n = conn.execute("SELECT COUNT(*) FROM file_status_table").fetchone()[0]
        except sqlite3.Error:
            n = "?"
        out(f"registry DB: delete ({n} papers tracked) — recreated empty on next registry start")
        if apply:
            REGISTRY_DB.unlink()
            for extra in (REGISTRY_DB.with_suffix(".db-wal"), REGISTRY_DB.with_suffix(".db-shm")):
                extra.unlink(missing_ok=True)
    else:
        out(f"registry DB not found at {REGISTRY_DB} (nothing to delete)")

    if apply:
        out("\nDone. Papers must be re-registered (scripts/register_pdfs.py) after the "
            "registry starts — scripts/ops.py fresh-start does all of this.")


if __name__ == "__main__":
    if {"-h", "--help"} & set(sys.argv[1:]):
        print(__doc__)
        raise SystemExit(0)
    reset(apply="--yes" in sys.argv)
