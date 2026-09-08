"""Download manager: gets PDFs into data/raw_pdfs/ and registers them.

    python downloader.py                          # loop: every CHECK_INTERVAL, all DOMAINS
    python downloader.py --once                   # one cycle over all DOMAINS, then exit
    python downloader.py --once --domain "Graph Neural Networks" --max 5
    python downloader.py --url https://arxiv.org/abs/2401.01234 --url https://x.org/paper.pdf
    python downloader.py --once --ensure-registry # start a registry if none is up (for timers)

Per domain, each cycle searches arXiv newest-first back to the domain's
checkpoint (the newest `published_at` the registry holds; else BACKFILL_DAYS
ago), skips papers the registry already knows, downloads new ones, and
registers them as "downloaded" with domain and publish date — which is what
parse_manager's loop picks up. A cycle aborts if the registry is down,
rather than downloading files the pipeline would never see.
"""

import argparse
import os
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import arxiv
import requests

from common.logsetup import get_logger
from common.paths import REPO_ROOT, VENV_PYTHON
from common.registry_client import RegistryClient, RegistryUnavailable
from config import BACKFILL_DAYS, CHECK_INTERVAL, DOMAINS, MAX_PAPERS_PER_DOMAIN, PDF_DIR

log = get_logger("download")

START_JITTER = (5, 60)      # so domain threads don't hit arXiv in lockstep
DOWNLOAD_SLEEP = (3, 10)
BREAK_AFTER = (5, 12)
BREAK_DURATION = (30, 90)
ARXIV_ID = re.compile(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)")


def sanitize_filename(title: str) -> str:
    """Title -> PDF stem = registry key. Must stay in sync with the rest of
    the pipeline, which keys everything on the stem."""
    return re.sub(r"[^a-zA-Z0-9_\- ]", "", title).replace(" ", "_")[:100]


# --------------------------------------------------------------- arXiv crawl
def get_smart_cutoff(registry: RegistryClient, domain: str) -> datetime:
    try:
        date_str = registry.checkpoint(domain)
        if date_str:
            cp = datetime.fromisoformat(date_str)
            cp = cp if cp.tzinfo else cp.replace(tzinfo=timezone.utc)
            log.info("[%s] resuming from %s", domain, cp.date())
            return cp
    except (RegistryUnavailable, ValueError) as e:
        log.warning("[%s] checkpoint lookup failed (%s); using backfill", domain, e)
    fallback = datetime.now(timezone.utc) - timedelta(days=BACKFILL_DAYS)
    log.info("[%s] backfill from %s", domain, fallback.date())
    return fallback


def process_domain(domain: str, max_papers, jitter: bool = True) -> str:
    registry = RegistryClient()
    if jitter:
        time.sleep(random.uniform(*START_JITTER))
    if not registry.health():
        log.error("[%s] registry unreachable; skipping this cycle", domain)
        return f"{domain}: registry down"

    cutoff = get_smart_cutoff(registry, domain)
    client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=5)
    search = arxiv.Search(query=f'ti:"{domain}" OR abs:"{domain}"',
                          max_results=None, sort_by=arxiv.SortCriterion.SubmittedDate)
    downloads = session = 0
    next_break = random.randint(*BREAK_AFTER)

    try:
        for result in client.results(search):
            if result.published < cutoff:
                log.info("[%s] caught up (reached %s)", domain, result.published.date())
                break
            if max_papers and downloads >= max_papers:
                log.info("[%s] per-cycle cap of %d reached", domain, max_papers)
                break
            stem = sanitize_filename(result.title)
            try:
                if registry.get_status(stem):
                    continue
            except RegistryUnavailable as e:
                log.error("[%s] registry unreachable mid-cycle (%s); stopping", domain, e)
                break
            try:
                _download_and_register(registry, result, stem, domain)
                downloads += 1
                session += 1
            except Exception as e:
                log.error("[%s] failed %s: %s", domain, result.title[:40], e)
                continue
            if session >= next_break:
                pause = random.uniform(*BREAK_DURATION)
                log.info("[%s] break %.0fs", domain, pause)
                time.sleep(pause)
                session, next_break = 0, random.randint(*BREAK_AFTER)
            else:
                time.sleep(random.uniform(*DOWNLOAD_SLEEP))
    except Exception:
        log.exception("[%s] crashed", domain)
    return f"{domain}: {downloads} new"


def _download_and_register(registry, result, stem, domain):
    pdf_path = PDF_DIR / f"{stem}.pdf"
    if not pdf_path.exists():
        log.info("[%s] downloading %s…", domain, result.title[:50])
        result.download_pdf(dirpath=str(PDF_DIR), filename=f"{stem}.pdf")
    if not registry.set_status(stem, "downloaded", domain=domain,
                               published_at=result.published.isoformat()):
        raise RuntimeError("registry rejected registration")


def run_cycle(domains, max_papers, jitter=True):
    with ThreadPoolExecutor(max_workers=len(domains)) as pool:
        futures = {pool.submit(process_domain, d, max_papers, jitter): d for d in domains}
        for f in as_completed(futures):
            try:
                log.info(f.result())
            except Exception as e:
                log.error("thread error [%s]: %s", futures[f], e)


# --------------------------------------------------------------- direct URLs
def download_url(registry: RegistryClient, url: str) -> str:
    """Download one PDF by URL (arXiv abs/pdf links get their real title as
    the stem, like crawled papers). Returns the stem."""
    m = ARXIV_ID.search(url)
    if m:
        result = next(arxiv.Client().results(arxiv.Search(id_list=[m.group(1)])))
        stem = sanitize_filename(result.title)
        _download_and_register(registry, result, stem, "manual")
        return stem

    r = requests.get(url, timeout=60, headers={"User-Agent": "RAGSetup/1.0"}, stream=True)
    r.raise_for_status()
    cd = r.headers.get("content-disposition", "")
    name = re.search(r'filename="?([^";]+)"?', cd)
    raw_name = name.group(1) if name else unquote(Path(urlparse(url).path).name) or "download"
    stem = sanitize_filename(Path(raw_name).stem) or f"paper_{int(time.time())}"
    pdf_path = PDF_DIR / f"{stem}.pdf"
    with open(pdf_path, "wb") as f:
        for chunk in r.iter_content(1 << 16):
            f.write(chunk)
    if pdf_path.read_bytes()[:5] != b"%PDF-":
        pdf_path.unlink()
        raise ValueError(f"{url} did not return a PDF")
    if not registry.set_status(stem, "downloaded", domain="manual"):
        raise RuntimeError("registry rejected registration")
    log.info("downloaded %s -> %s.pdf", url, stem)
    return stem


# --------------------------------------------------------------- registry helper
class EnsureRegistry:
    """Start a registry if none is running (for timer-driven runs); stop it
    on exit if we started it."""

    def __init__(self):
        self.proc = None

    def __enter__(self):
        registry = RegistryClient()
        if registry.health():
            return registry
        log.info("no registry running; starting one for this run")
        self.proc = subprocess.Popen([str(VENV_PYTHON), "main.py"], cwd=REPO_ROOT / "registry_manager",
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(30):
            if registry.health():
                return registry
            time.sleep(1)
        raise RuntimeError("registry did not come up")

    def __exit__(self, *exc):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


# --------------------------------------------------------------- entry
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true", help="one cycle, then exit")
    ap.add_argument("--domain", action="append", help="search this domain (repeatable; default: config DOMAINS)")
    ap.add_argument("--max", type=int, default=None, help="max new papers per domain this cycle")
    ap.add_argument("--url", action="append", help="download this PDF / arXiv link directly (repeatable)")
    ap.add_argument("--ensure-registry", action="store_true", help="start a registry if none is up")
    args = ap.parse_args()

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    domains = args.domain or DOMAINS
    max_papers = args.max if args.max is not None else MAX_PAPERS_PER_DOMAIN

    ctx = EnsureRegistry() if args.ensure_registry else None
    registry = ctx.__enter__() if ctx else RegistryClient()
    try:
        if args.url:
            for url in args.url:
                try:
                    download_url(registry, url)
                except Exception as e:
                    log.error("failed %s: %s", url, e)
            if not args.once and not args.domain:
                return
        if args.once or args.domain:
            log.info("one cycle: %d domain(s), cap %s", len(domains), max_papers)
            run_cycle(domains, max_papers, jitter=len(domains) > 1)
            return
        log.info("downloader up — %d domains, cap %s/domain/cycle, every %ss", len(domains), max_papers, CHECK_INTERVAL)
        while True:
            start = time.time()
            run_cycle(domains, max_papers)
            log.info("cycle done in %.0fs; next in %ss", time.time() - start, CHECK_INTERVAL)
            time.sleep(CHECK_INTERVAL)
    finally:
        if ctx:
            ctx.__exit__(None, None, None)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("downloader stopped")
