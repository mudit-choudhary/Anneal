"""Build a fresh arXiv corpus for the parser/chunker matrix, disjoint from the
YOLO training set.

    python evals/scripts/build_corpus.py            # build or reuse the cache
    python evals/scripts/build_corpus.py --report   # print the manifest only

arXiv Terms of Use (https://info.arxiv.org/help/api/tou.html) state: "make no
more than one request every three seconds, and limit requests to a single
connection at a time." Every call here — metadata query and PDF download alike
— goes through `_polite_get`, which sleeps to keep a 3 second floor between
requests and never runs concurrently. A descriptive User-Agent is sent even
though the ToU does not demand one.

Papers are cached in evals/corpus/. A second run re-reads the cache and never
re-downloads.
"""

import argparse
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

REPO = Path(__file__).resolve().parent.parent.parent
CORPUS = REPO / "evals" / "corpus"
MANIFEST = CORPUS / "manifest.json"
EXCLUDED = CORPUS / "excluded_ids.json"
TRAINING_PDFS = Path("/media/mudit/DarkDwine1/ResearchPapersYOLO_FT/PDFs")

API = "http://export.arxiv.org/api/query"
UA = "RAGSetup-eval/1.0 (research corpus build; contact themuditchoudhary@gmail.com)"
MIN_INTERVAL = 3.0          # seconds between requests, per arXiv ToU

# Chosen to span single-column (NeurIPS/ICML style) and two-column
# (IEEE / RevTeX style) layouts.
# arXiv split the astro-ph and cond-mat archives into subcategories in 2009;
# querying the bare archive name returns only pre-2009 papers, which is how an
# earlier build ended up with a corpus dated 2008.
CATEGORIES = ["cs.LG", "cs.CL", "cs.CV", "eess.IV",
              "astro-ph.GA", "astro-ph.CO", "cond-mat.mtrl-sci", "cond-mat.stat-mech"]
PER_CATEGORY = 10
SEED = 20260912
SAMPLE_SIZE = 15
MIN_TOTAL_PAGES = 320
MIN_TOTAL_BYTES = 2 * 1024 * 1024

_ARXIV_ID = re.compile(r"arXiv:\s*(\d{4}\.\d{4,5})", re.I)
_PAGES_IN_COMMENT = re.compile(r"(\d+)\s*pages?", re.I)
_LONG_HINT = re.compile(r"appendix|supplementary|supplemental", re.I)

_last_request = [0.0]


def _polite_get(url, **kwargs):
    """Single-connection GET honouring arXiv's 3-second floor."""
    import requests

    wait = MIN_INTERVAL - (time.time() - _last_request[0])
    if wait > 0:
        time.sleep(wait)
    resp = requests.get(url, headers={"User-Agent": UA}, timeout=60, **kwargs)
    _last_request[0] = time.time()
    resp.raise_for_status()
    return resp


# --------------------------------------------------------------- exclusions
def training_ids():
    """arXiv ids of every PDF in the YOLO training directory.

    Those papers may have contributed pages to round_03 training, so evaluating
    the `current` parser on them would flatter it by an unknown margin. The ids
    are read from the arXiv stamp on page 1 rather than the filenames, which
    carry titles only.
    """
    if EXCLUDED.exists():
        return set(json.loads(EXCLUDED.read_text())["ids"])

    import pymupdf

    ids, unreadable = set(), 0
    files = sorted(TRAINING_PDFS.glob("*.pdf"))
    for i, f in enumerate(files, 1):
        try:
            with pymupdf.open(f) as d:
                m = _ARXIV_ID.search(d[0].get_text())
            if m:
                ids.add(m.group(1))
            else:
                unreadable += 1
        except Exception:                                    # noqa: BLE001
            unreadable += 1
        if i % 200 == 0:
            print(f"  scanned {i}/{len(files)}", flush=True)
    CORPUS.mkdir(parents=True, exist_ok=True)
    EXCLUDED.write_text(json.dumps(
        {"ids": sorted(ids), "scanned": len(files), "no_id_found": unreadable}, indent=1))
    print(f"  {len(ids)} ids recovered from {len(files)} training PDFs "
          f"({unreadable} without a readable stamp)")
    return ids


# --------------------------------------------------------------- pool
def query_category(cat, want, start=0):
    """One arXiv API call. Returns (entries, query_string)."""
    import feedparser

    params = {
        "search_query": f"cat:{cat}",
        "start": start,
        "max_results": want,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = f"{API}?{urlencode(params)}"
    feed = feedparser.parse(_polite_get(url).text)
    out = []
    for e in feed.entries:
        raw = e.get("id", "")
        m = re.search(r"abs/(\d{4}\.\d{4,5})", raw)
        if not m:
            continue
        comment = e.get("arxiv_comment", "") or ""
        pages_hint = _PAGES_IN_COMMENT.search(comment)
        out.append({
            "arxiv_id": m.group(1),
            "title": " ".join(e.title.split()),
            "primary_category": e.get("arxiv_primary_category", {}).get("term", cat),
            "queried_category": cat,
            "comment": comment,
            "pages_hint": int(pages_hint.group(1)) if pages_hint else None,
            "long_hint": bool(_LONG_HINT.search(comment)),
            "pdf_url": f"https://arxiv.org/pdf/{m.group(1)}",
        })
    return out, url


def build_pool(exclude, target, queries):
    """Gather candidates, biased toward long submissions, growing if needed."""
    pool, seen, start = [], set(), 0
    while len(pool) < target and start < 200:
        for cat in CATEGORIES:
            entries, url = query_category(cat, PER_CATEGORY, start)
            queries.append(url)
            for e in entries:
                if e["arxiv_id"] in exclude or e["arxiv_id"] in seen:
                    continue
                seen.add(e["arxiv_id"])
                pool.append(e)
            print(f"  {cat:<10} start={start:<4} pool={len(pool)}", flush=True)
        if len(pool) >= target:
            break
        start += PER_CATEGORY
    # Longest first: the page target is what makes the corpus demanding.
    pool.sort(key=lambda e: (e["long_hint"], e["pages_hint"] or 0), reverse=True)
    return pool


# --------------------------------------------------------------- pdf facts
def fetch_pdf(entry):
    path = CORPUS / f"{entry['arxiv_id']}.pdf"
    if path.exists() and path.stat().st_size > 10_000:
        return path
    data = _polite_get(entry["pdf_url"], allow_redirects=True).content
    if not data.startswith(b"%PDF"):
        raise ValueError("not a PDF")
    path.write_bytes(data)
    return path


def inspect_pdf(path):
    """Page count, size, sha256, column count and table presence.

    Column count and tables come from PyMuPDF geometry, never from our own
    layout model — using the thing under test to pick the corpus would make the
    sampling depend on it.
    """
    import pymupdf

    with pymupdf.open(path) as doc:
        pages = len(doc)
        cols, tables = [], False
        # tables can appear anywhere; sample across the document rather than
        # only the opening pages
        for pno in sorted({p for p in range(1, pages) if p < 4 or p % max(pages // 8, 1) == 0}):
            try:
                if not tables and len(doc[pno].find_tables().tables) > 0:
                    tables = True
            except Exception:                                # noqa: BLE001
                pass
        for pno in range(1, min(4, pages)):
            page = doc[pno]
            width = page.rect.width
            words = page.get_text("words")
            if words:
                # A two-column page has a gutter no word crosses. Counting word
                # *centres* in a middle band does not detect this — words ending
                # just left of the gutter have centres inside the band — so the
                # test is whether a word spans the midline at all.
                mid = width / 2
                straddlers = sum(1 for w in words if w[0] < mid < w[2])
                cols.append(2 if straddlers / max(len(words), 1) < 0.01 else 1)
        column_count = max(set(cols), key=cols.count) if cols else 1
    return {
        "pages": pages,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "columns": column_count,
        "has_tables": tables,
    }


# --------------------------------------------------------------- sampling
def sample(pool, seed):
    """Rejection-sample until the corpus is large enough.

    Page count and total size are hard constraints. Column and table mix are
    recorded but not enforced: the category spread already guarantees both
    single- and two-column material.
    """
    rng = random.Random(seed)
    tried = 0
    while tried < 4000:
        tried += 1
        pick = rng.sample(pool, SAMPLE_SIZE)
        pages = sum(p["pages"] for p in pick)
        size = sum(p["bytes"] for p in pick)
        if pages > MIN_TOTAL_PAGES and size > MIN_TOTAL_BYTES:
            return pick, tried, pages, size
    return None, tried, 0, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--pool", type=int, default=60)
    args = ap.parse_args()

    if args.report:
        if not MANIFEST.exists():
            sys.exit("no manifest yet")
        m = json.loads(MANIFEST.read_text())
        print(json.dumps(m["totals"], indent=1))
        return

    CORPUS.mkdir(parents=True, exist_ok=True)
    print("Reading arXiv ids from the YOLO training set (cached after the first run)")
    exclude = training_ids()

    queries = []
    print(f"\nQuerying arXiv, one request per {MIN_INTERVAL}s, single connection")
    pool = build_pool(exclude, args.pool, queries)
    print(f"\npool: {len(pool)} candidates after excluding the training set")

    print("\nDownloading (cached; 3s apart)")
    downloaded, failed = [], []
    for i, e in enumerate(pool, 1):
        try:
            path = fetch_pdf(e)
            e.update(inspect_pdf(path))
            downloaded.append(e)
        except Exception as ex:                              # noqa: BLE001
            failed.append({"arxiv_id": e["arxiv_id"], "error": str(ex)[:120]})
            continue
        if i % 10 == 0 or i == len(pool):
            print(f"  {i}/{len(pool)}  usable={len(downloaded)}", flush=True)

    picked, tried, pages, size = sample(downloaded, SEED)
    if picked is None:
        sys.exit(f"no sample of {SAMPLE_SIZE} met the constraints after {tried} draws; "
                 f"grow the pool with --pool")

    manifest = {
        "seed": SEED,
        "sample_size": SAMPLE_SIZE,
        "draws_until_valid": tried,
        "api_queries": queries,
        "arxiv_tou": "1 request / 3s, single connection (info.arxiv.org/help/api/tou.html)",
        "excluded_training_ids": len(exclude),
        "excluded_from_pool": json.loads(EXCLUDED.read_text()).get("scanned"),
        "pool_size": len(downloaded),
        "download_failures": failed,
        "totals": {
            "papers": len(picked), "pages": pages, "bytes": size,
            "two_column": sum(1 for p in picked if p["columns"] == 2),
            "with_tables": sum(1 for p in picked if p["has_tables"]),
        },
        "papers": sorted(picked, key=lambda p: p["arxiv_id"]),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1))
    t = manifest["totals"]
    print(f"\nmanifest: {t['papers']} papers, {t['pages']} pages, "
          f"{t['bytes']/1048576:.1f} MB, {t['two_column']} two-column, "
          f"{t['with_tables']} with tables  ->  {MANIFEST}")


if __name__ == "__main__":
    main()
