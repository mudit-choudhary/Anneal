# Anneal — local-first RAG over research papers.
# Copyright (C) 2026 Mudit Choudhary
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This program is free software: you may redistribute and/or modify it under
# the terms of the GNU Affero General Public License, version 3 or later. It
# is distributed WITHOUT ANY WARRANTY. See the LICENSE file, or
# <https://www.gnu.org/licenses/>.
"""Prune manager: removes intermediate files once a later stage has consumed
them, and optionally archives or deletes raw PDFs.

    directory        acted on once the paper's status is
    data/parsed      processed | embedded      -> deleted
    data/processed   embedded                  -> deleted
    data/raw_pdfs    embedded                  -> kept (default), archived, or
                                                 deleted, per the "prune"
                                                 settings (UI-editable)

    python pruning.py            # sweep on the configured interval
    python pruning.py --once     # one sweep, then exit
    python pruning.py --dry-run  # report what a sweep would do, change nothing
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import settings as settings_store
from common.logsetup import get_logger
from common.registry_client import RegistryClient, RegistryUnavailable
from config import MIN_INTERVAL, PARSED_DIR, PDF_DIR, PROCESSED_DIR

log = get_logger("prune")


def raw_pdf_action(cfg):
    """'archive' | 'delete' | None (leave raw PDFs alone)."""
    policy = cfg.get("raw_pdf_policy", "keep")
    if policy == "delete":
        return "delete"
    if policy == "archive":
        target = (cfg.get("archive_dir") or "").strip()
        if not target:
            log.warning("raw_pdf_policy is 'archive' but archive_dir is empty; keeping PDFs")
            return None
        if Path(target).expanduser().resolve() == PDF_DIR.resolve():
            return None
        return "archive"
    return None


def select_raw_pdfs(embedded_papers, cfg):
    """Apply the keep strategy; returns the papers to archive/delete."""
    present = [p for p in embedded_papers if (PDF_DIR / f"{p['filename']}.pdf").exists()]
    strategy = cfg.get("keep_strategy", "all")
    count = int(cfg.get("keep_count", 0) or 0)
    if strategy == "all":
        return present
    ordered = sorted(present, key=lambda p: p.get("downloaded_at") or "")
    if strategy == "newest":          # keep the newest N, act on the rest
        keep = {p["filename"] for p in ordered[-count:]} if count else set()
    elif strategy == "oldest":        # keep the oldest N, act on the rest
        keep = {p["filename"] for p in ordered[:count]}
    else:
        log.warning("unknown keep_strategy %r; keeping everything", strategy)
        return []
    return [p for p in present if p["filename"] not in keep]


def sweep(registry: RegistryClient, cfg=None, dry_run=False):
    """One pass over all three directories. Returns counts per action."""
    cfg = cfg if cfg is not None else settings_store.load()["prune"]
    papers = {p["filename"]: p for p in registry.list_papers()}
    status = lambda path: papers.get(path.stem, {}).get("status")  # noqa: E731
    counts = {"parsed": 0, "processed": 0, "raw_archived": 0, "raw_deleted": 0}

    for directory, key, when in ((PARSED_DIR, "parsed", ("processed", "embedded")),
                                 (PROCESSED_DIR, "processed", ("embedded",))):
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*")):
            if path.is_file() and status(path) in when:
                if not dry_run:
                    path.unlink()
                counts[key] += 1

    action = raw_pdf_action(cfg)
    if action:
        targets = select_raw_pdfs([p for p in papers.values() if p["status"] == "embedded"], cfg)
        for p in targets:
            src = PDF_DIR / f"{p['filename']}.pdf"
            if action == "archive":
                dest_dir = Path(cfg["archive_dir"]).expanduser()
                if not dry_run:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(dest_dir / src.name))
                counts["raw_archived"] += 1
            else:
                if not dry_run:
                    src.unlink()
                counts["raw_deleted"] += 1
    return counts


def deletion_loop(once=False, dry_run=False):
    registry = RegistryClient()
    while True:
        cfg = settings_store.load()["prune"]           # re-read: UI-editable
        try:
            counts = sweep(registry, cfg, dry_run)
            if any(counts.values()):
                log.info("%s: %s", "would remove" if dry_run else "swept",
                         ", ".join(f"{k}={v}" for k, v in counts.items() if v))
        except RegistryUnavailable as e:
            log.warning("sweep skipped: %s", e)
        except Exception:
            log.exception("sweep failed")
        if once or dry_run:
            return
        time.sleep(max(MIN_INTERVAL, int(cfg.get("interval_seconds", 1800))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="report only; change nothing")
    args = ap.parse_args()
    cfg = settings_store.load()["prune"]
    log.info("prune up — raw PDFs: %s%s", cfg.get("raw_pdf_policy", "keep"),
             f" (keep {cfg.get('keep_strategy')} {cfg.get('keep_count')})"
             if cfg.get("raw_pdf_policy") != "keep" and cfg.get("keep_strategy") != "all" else "")
    try:
        deletion_loop(once=args.once, dry_run=args.dry_run)
    except KeyboardInterrupt:
        log.info("stopped")
