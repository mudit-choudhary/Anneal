# Anneal — local-first RAG over research papers.
# Copyright (C) 2026 Mudit Choudhary
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This program is free software: you may redistribute and/or modify it under
# the terms of the GNU Affero General Public License, version 3 or later. It
# is distributed WITHOUT ANY WARRANTY. See the LICENSE file, or
# <https://www.gnu.org/licenses/>.
"""Parse manager service.

Two polling loops share one process:
- parser_loop:    papers with status "downloaded" -> layout JSON in parsed/
                  (status "parsed")
- processor_loop: papers with status "parsed" -> assembled text in processed/
                  (status "processed")
A paper whose stage raises gets status "error" with the message and is left
alone until re-registered (scripts/register_pdfs.py --retry-errors).
"""

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.logsetup import get_logger
from common.registry_client import RegistryClient, RegistryUnavailable
from config import PARSED_DIR, PDF_DIR, PROCESSED_DIR
from pdf_parser import parse
from txt_processor import process_layout_json

log = get_logger("parse")


def _loop(status, directory, suffix, work, poll_interval):
    registry = RegistryClient()
    while True:
        try:
            for paper in registry.list_papers(status=status):
                stem = paper["filename"]
                path = directory / f"{stem}{suffix}"
                if not path.exists():
                    registry.report_error(stem, f"input missing: {path.name}")
                    continue
                try:
                    work(path)
                except Exception as e:
                    log.exception("%s failed for %s", work.__name__, stem)
                    registry.report_error(stem, f"{work.__name__}: {e}")
        except RegistryUnavailable as e:
            log.warning("%s", e)
        except Exception:
            log.exception("loop error")
        time.sleep(poll_interval)


def parser_loop(poll_interval=5):
    _loop("downloaded", PDF_DIR, ".pdf", parse, poll_interval)


def processor_loop(poll_interval=5):
    _loop("parsed", PARSED_DIR, ".json", process_layout_json, poll_interval)


if __name__ == "__main__":
    for d in (PDF_DIR, PARSED_DIR, PROCESSED_DIR):
        d.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=parser_loop, daemon=True).start()
    threading.Thread(target=processor_loop, daemon=True).start()
    log.info("parse manager up")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        log.info("shutting down")
