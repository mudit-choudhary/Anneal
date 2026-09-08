"""One logging configuration for every service and script.

    from common.logsetup import get_logger
    log = get_logger("parse")
    log.info("parsed %s", stem)

Format: `2026-09-09 01:02:03 INFO [parse] parsed A_Paper`. Level from
RAG_LOG_LEVEL (default INFO). Output goes to stdout; the launcher
(scripts/ops) redirects each service's stdout to run/logs/<service>.log, so
nothing is written twice.
"""

import logging
import os
import sys

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        level = os.environ.get("RAG_LOG_LEVEL", "INFO").upper()
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
        root = logging.getLogger()
        root.handlers[:] = [handler]
        root.setLevel(level)
        # third-party chatter we never want at INFO
        for noisy in ("httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers",
                      "ultralytics", "trafilatura", "primp"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        _CONFIGURED = True
    return logging.getLogger(name)
