"""Prune manager: deletes intermediate files once the pipeline has moved a
paper past the stage that produced them.

    directory        deleted once status is in
    data/parsed      processed, embedded
    data/processed   embedded
    data/raw_pdfs    parsed, processed, embedded   (only if PRUNE_RAW_PDFS)

Runs one sweep every PRUNE_INTERVAL seconds. Files the registry doesn't know
about are left alone.
"""

import os
import time

import requests

from config import (
    PARSED_DIR,
    PDF_DIR,
    PROCESSED_DIR,
    PRUNE_INTERVAL,
    PRUNE_RAW_PDFS,
    REGISTRY_URL,
)

RULES = [
    (PARSED_DIR, {"processed", "embedded"}),
    (PROCESSED_DIR, {"embedded"}),
]
if PRUNE_RAW_PDFS:
    RULES.append((PDF_DIR, {"parsed", "processed", "embedded"}))


def get_status(stem):
    r = requests.get(f"{REGISTRY_URL}/get_status", json={"filename": stem}, timeout=20)
    r.raise_for_status()
    return r.json().get("status")


def sweep(directory, delete_when):
    if not os.path.isdir(directory):
        return 0
    deleted = 0
    for file in sorted(os.listdir(directory)):
        stem = os.path.splitext(file)[0]
        try:
            status = get_status(stem)
        except requests.RequestException as e:
            print(f"[prune] registry unreachable ({e}); sweep aborted")
            return deleted
        if status in delete_when:
            os.remove(os.path.join(directory, file))
            deleted += 1
    return deleted


def deletion_loop():
    print(f"[prune] up — sweeping every {PRUNE_INTERVAL}s; raw PDFs "
          f"{'INCLUDED' if PRUNE_RAW_PDFS else 'kept'}")
    while True:
        for directory, delete_when in RULES:
            n = sweep(directory, delete_when)
            if n:
                print(f"[prune] removed {n} files from {directory}")
        time.sleep(PRUNE_INTERVAL)


if __name__ == "__main__":
    try:
        deletion_loop()
    except KeyboardInterrupt:
        print("[prune] stopped")
