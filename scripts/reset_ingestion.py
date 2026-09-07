"""Reset the pipeline so every downloaded paper is re-parsed, re-chunked and
re-embedded — needed after any change to parsing, chunking, or the embedding
model.

    python scripts/reset_ingestion.py          # dry run: shows what would happen
    python scripts/reset_ingestion.py --yes    # do it

STOP the registry, parse_manager and embedding_manager services first.
Then start them again and let the pipeline run (overnight for a big corpus).
Raw PDFs in data/raw_pdfs/ are never touched.
"""

import shutil
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VECTOR_DB = REPO_ROOT / "vector_db"
PARSED = REPO_ROOT / "data" / "parsed"
PROCESSED = REPO_ROOT / "data" / "processed"
REGISTRY_DB = REPO_ROOT / "registry_manager" / "rag_registry.db"


def main():
    do_it = "--yes" in sys.argv
    print("Dry run — pass --yes to apply.\n" if not do_it else "Applying reset.\n")

    for d in (PARSED, PROCESSED):
        files = [p for p in d.glob("*") if p.is_file()] if d.exists() else []
        print(f"{d.relative_to(REPO_ROOT)}: delete {len(files)} files")
        if do_it:
            for p in files:
                p.unlink()

    if VECTOR_DB.exists():
        size = sum(p.stat().st_size for p in VECTOR_DB.rglob("*") if p.is_file()) / 1e6
        print(f"vector_db/: delete ({size:.0f} MB)")
        if do_it:
            shutil.rmtree(VECTOR_DB)
    else:
        print("vector_db/: not present")

    # The registry DB is deleted outright rather than reset: the registry
    # recreates it with the current schema on startup, and schema changes
    # (CHECK constraints, new columns) never apply to an existing file.
    if REGISTRY_DB.exists():
        try:
            with sqlite3.connect(REGISTRY_DB) as conn:
                n = conn.execute("SELECT COUNT(*) FROM file_status_table").fetchone()[0]
        except sqlite3.Error:
            n = "?"
        print(f"registry DB: delete ({n} papers tracked) — recreated empty on next registry start")
        if do_it:
            REGISTRY_DB.unlink()
            for extra in (REGISTRY_DB.with_suffix(".db-wal"), REGISTRY_DB.with_suffix(".db-shm")):
                extra.unlink(missing_ok=True)
    else:
        print(f"registry DB not found at {REGISTRY_DB} (nothing to delete)")

    if do_it:
        print("\nDone. Papers must be re-registered (scripts/register_pdfs.py) after the "
              "registry starts — scripts/fresh_start.sh does all of this.")


if __name__ == "__main__":
    main()
