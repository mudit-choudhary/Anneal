"""Prune manager: removes intermediate files once a later stage has
consumed them, and (optionally) archives or deletes raw PDFs.

    directory        acted on once the paper's status is
    data/parsed      processed | embedded      -> deleted
    data/processed   embedded                  -> deleted
    data/raw_pdfs    embedded                  -> archived to RAW_PDF_ARCHIVE_DIR,
                                                 or deleted if RAW_PDF_DELETE,
                                                 or (default) left alone

    python pruning.py          # sweep every PRUNE_INTERVAL seconds
    python pruning.py --once   # one sweep, then exit
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.logsetup import get_logger
from common.registry_client import RegistryClient, RegistryUnavailable
from config import (
    PARSED_DIR,
    PDF_DIR,
    PROCESSED_DIR,
    PRUNE_INTERVAL,
    RAW_PDF_ARCHIVE_DIR,
    RAW_PDF_DELETE,
    RAW_PDF_KEEP_COUNT,
    RAW_PDF_KEEP_STRATEGY,
)

log = get_logger("prune")


def raw_pdf_action():
    """'archive' | 'delete' | None (leave PDFs alone)."""
    if RAW_PDF_DELETE:
        return "delete"
    if RAW_PDF_ARCHIVE_DIR and Path(RAW_PDF_ARCHIVE_DIR).expanduser().resolve() != PDF_DIR.resolve():
        return "archive"
    return None


def select_raw_pdfs(embedded_papers):
    """Apply the keep strategy; returns the papers to archive/delete."""
    present = [p for p in embedded_papers if (PDF_DIR / f"{p['filename']}.pdf").exists()]
    if RAW_PDF_KEEP_STRATEGY == "all":
        return present
    ordered = sorted(present, key=lambda p: p.get("downloaded_at") or "")
    if RAW_PDF_KEEP_STRATEGY == "newest":
        keep = set(p["filename"] for p in ordered[-RAW_PDF_KEEP_COUNT:]) if RAW_PDF_KEEP_COUNT else set()
    elif RAW_PDF_KEEP_STRATEGY == "oldest":
        keep = set(p["filename"] for p in ordered[:RAW_PDF_KEEP_COUNT])
    else:
        log.warning("unknown RAW_PDF_KEEP_STRATEGY %r; keeping everything", RAW_PDF_KEEP_STRATEGY)
        return []
    return [p for p in present if p["filename"] not in keep]


def sweep(registry: RegistryClient):
    """One pass over all three directories. Returns counts per action."""
    papers = {p["filename"]: p for p in registry.list_papers()}
    status = lambda path: papers.get(path.stem, {}).get("status")  # noqa: E731
    counts = {"parsed": 0, "processed": 0, "raw_archived": 0, "raw_deleted": 0}

    for path in sorted(PARSED_DIR.glob("*")) if PARSED_DIR.exists() else []:
        if path.is_file() and status(path) in ("processed", "embedded"):
            path.unlink()
            counts["parsed"] += 1
    for path in sorted(PROCESSED_DIR.glob("*")) if PROCESSED_DIR.exists() else []:
        if path.is_file() and status(path) == "embedded":
            path.unlink()
            counts["processed"] += 1

    action = raw_pdf_action()
    if action:
        targets = select_raw_pdfs([p for p in papers.values() if p["status"] == "embedded"])
        for p in targets:
            src = PDF_DIR / f"{p['filename']}.pdf"
            if action == "archive":
                dest_dir = Path(RAW_PDF_ARCHIVE_DIR).expanduser()
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dest_dir / src.name))
                counts["raw_archived"] += 1
            else:
                src.unlink()
                counts["raw_deleted"] += 1
    return counts


def deletion_loop(once: bool = False):
    registry = RegistryClient()
    action = raw_pdf_action()
    log.info("prune up — every %ss; raw PDFs: %s%s", PRUNE_INTERVAL,
             action or "kept in place",
             f" (keep {RAW_PDF_KEEP_STRATEGY} {RAW_PDF_KEEP_COUNT})" if action and RAW_PDF_KEEP_STRATEGY != "all" else "")
    while True:
        try:
            counts = sweep(registry)
            if any(counts.values()):
                log.info("sweep: %s", ", ".join(f"{k}={v}" for k, v in counts.items() if v))
        except RegistryUnavailable as e:
            log.warning("sweep skipped: %s", e)
        except Exception:
            log.exception("sweep failed")
        if once:
            return
        time.sleep(PRUNE_INTERVAL)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    try:
        deletion_loop(once=args.once)
    except KeyboardInterrupt:
        log.info("stopped")
