"""Download manager: crawls arXiv per domain and registers new PDFs.

Each cycle, every domain is searched newest-first back to its checkpoint
(the newest `published_at` the registry has for that domain, else
BACKFILL_DAYS ago). Papers already known to the registry are skipped; new
ones are downloaded into data/raw_pdfs/ and registered as "downloaded" with
their domain and publish date, which is what the parse_manager loop picks up.

Requires the registry service (port 4000) — a cycle aborts if it's down,
rather than downloading files the pipeline would never see.
"""

import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import arxiv
import requests
from loguru import logger

from config import (
    BACKFILL_DAYS,
    CHECK_INTERVAL,
    DOMAINS,
    MAX_PAPERS_PER_DOMAIN,
    PDF_DIR,
    REGISTRY_URL,
)

# Jitter so 10 domain threads don't hit arXiv in lockstep
START_JITTER = (5, 60)
DOWNLOAD_SLEEP = (3, 10)
BREAK_AFTER = (5, 12)
BREAK_DURATION = (30, 90)


def sanitize_filename(title):
    """Title -> registry key / PDF stem. Must stay in sync with the rest of the
    pipeline, which keys everything on the PDF's stem."""
    return re.sub(r"[^a-zA-Z0-9_\- ]", "", title).replace(" ", "_")[:100]


def registry_status(stem):
    r = requests.get(f"{REGISTRY_URL}/get_status", json={"filename": stem}, timeout=10)
    r.raise_for_status()
    return r.json().get("status")


def register_download(stem, domain, published):
    r = requests.post(
        f"{REGISTRY_URL}/update_status",
        json={"filename": stem, "status": "downloaded", "domain": domain,
              "published_at": published.isoformat()},
        timeout=10,
    )
    return r.ok and r.json().get("success", False)


def get_smart_cutoff(domain):
    """Newest publish date already downloaded for this domain, else a
    BACKFILL_DAYS fallback. Same-day papers are re-checked (and skipped via
    the registry) so nothing published on the checkpoint day is missed."""
    try:
        r = requests.post(f"{REGISTRY_URL}/get_last_checkpoint", json={"domain": domain}, timeout=5)
        date_str = r.json().get("last_checkpoint") if r.ok else None
        if date_str:
            checkpoint = datetime.fromisoformat(date_str)
            if checkpoint.tzinfo is None:
                checkpoint = checkpoint.replace(tzinfo=timezone.utc)
            logger.info(f"🧠 [{domain}] resuming from {checkpoint.date()}")
            return checkpoint
    except (requests.RequestException, ValueError) as e:
        logger.warning(f"⚠️ [{domain}] checkpoint lookup failed ({e}); using backfill")
    fallback = datetime.now(timezone.utc) - timedelta(days=BACKFILL_DAYS)
    logger.info(f"🔙 [{domain}] backfill from {fallback.date()}")
    return fallback


def process_domain(domain):
    time.sleep(random.uniform(*START_JITTER))

    try:
        registry_status("__healthcheck__")
    except requests.RequestException as e:
        logger.error(f"❌ [{domain}] registry unreachable ({e}); skipping this cycle")
        return f"{domain}: registry down"

    cutoff = get_smart_cutoff(domain)
    client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=5)
    search = arxiv.Search(query=f'ti:"{domain}" OR abs:"{domain}"',
                          max_results=None, sort_by=arxiv.SortCriterion.SubmittedDate)

    downloads = session = 0
    next_break = random.randint(*BREAK_AFTER)

    try:
        for result in client.results(search):
            if result.published < cutoff:
                logger.success(f"🛑 [{domain}] caught up (reached {result.published.date()})")
                break
            if MAX_PAPERS_PER_DOMAIN and downloads >= MAX_PAPERS_PER_DOMAIN:
                logger.info(f"⏸ [{domain}] per-cycle cap of {MAX_PAPERS_PER_DOMAIN} reached")
                break

            stem = sanitize_filename(result.title)
            try:
                if registry_status(stem):
                    continue
            except requests.RequestException as e:
                logger.error(f"❌ [{domain}] registry unreachable mid-cycle ({e}); stopping")
                break

            pdf_path = os.path.join(PDF_DIR, stem + ".pdf")
            try:
                if not os.path.exists(pdf_path):
                    logger.info(f"⬇️  [{domain}] {result.title[:50]}…")
                    result.download_pdf(dirpath=PDF_DIR, filename=stem + ".pdf")
                if not register_download(stem, domain, result.published):
                    logger.error(f"❌ [{domain}] registry rejected {stem}")
                    continue
                downloads += 1
                session += 1
            except Exception as e:
                logger.error(f"❌ [{domain}] failed {result.title[:40]}: {e}")
                continue

            if session >= next_break:
                pause = random.uniform(*BREAK_DURATION)
                logger.info(f"☕ [{domain}] break {pause:.0f}s")
                time.sleep(pause)
                session, next_break = 0, random.randint(*BREAK_AFTER)
            else:
                time.sleep(random.uniform(*DOWNLOAD_SLEEP))
    except Exception as e:
        logger.error(f"⚠️ [{domain}] crashed: {e}")

    return f"{domain}: {downloads} new"


def main_loop():
    os.makedirs(PDF_DIR, exist_ok=True)
    logger.info(f"🔥 downloader up — {len(DOMAINS)} domains, cap {MAX_PAPERS_PER_DOMAIN}/domain/cycle")
    while True:
        start = time.time()
        with ThreadPoolExecutor(max_workers=len(DOMAINS)) as pool:
            futures = {pool.submit(process_domain, d): d for d in DOMAINS}
            for f in as_completed(futures):
                try:
                    logger.info(f.result())
                except Exception as e:
                    logger.error(f"thread error [{futures[f]}]: {e}")
        logger.success(f"💤 cycle done in {time.time() - start:.0f}s; next in {CHECK_INTERVAL}s")
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        logger.info("🛑 downloader stopped")
