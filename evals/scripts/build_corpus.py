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
# The ToU floor is 3s; this uses 15s. A 200-paper build makes several hundred
# requests, and arXiv returned 429 and then 503 at both 3s and 5s. The
# published floor is a limit, not a target: being rate-limited is a signal to
# sit well under it, not to retry into it.
MIN_INTERVAL = 15.0

# Chosen to span single-column (NeurIPS/ICML style) and two-column
# (IEEE / RevTeX style) layouts.
# arXiv split the astro-ph and cond-mat archives into subcategories in 2009;
# querying the bare archive name returns only pre-2009 papers, which is how an
# earlier build ended up with a corpus dated 2008.
CATEGORIES = ["cs.LG", "cs.CL", "cs.CV", "cs.IR", "cs.SE", "cs.RO",
              "eess.IV", "eess.SP",
              "astro-ph.GA", "astro-ph.CO", "cond-mat.mtrl-sci", "cond-mat.stat-mech",
              "math.OC", "stat.ME", "q-bio.QM", "physics.comp-ph"]
PER_CATEGORY = 20
SEED = 20260912
SAMPLE_SIZE = 15
MIN_TOTAL_PAGES = 320
MIN_TOTAL_BYTES = 2 * 1024 * 1024

_ARXIV_ID = re.compile(r"arXiv:\s*(\d{4}\.\d{4,5})", re.I)
_PAGES_IN_COMMENT = re.compile(r"(\d+)\s*pages?", re.I)
_LONG_HINT = re.compile(r"appendix|supplementary|supplemental", re.I)

_last_request = [0.0]


def _polite_get(url, attempts=5, **kwargs):
    """Single-connection GET, well inside arXiv's rate limit, backing off on 429.

    A 429 means we are being told to slow down; the response is to wait longer
    each time rather than retry immediately.
    """
    import requests

    delay = MIN_INTERVAL
    for attempt in range(attempts):
        wait = delay - (time.time() - _last_request[0])
        if wait > 0:
            time.sleep(wait)
        resp = requests.get(url, headers={"User-Agent": UA}, timeout=90, **kwargs)
        _last_request[0] = time.time()
        if resp.status_code == 429:
            delay = min(delay * 2, 120)
            print(f"    429 from arXiv; backing off to {delay:.0f}s", flush=True)
            time.sleep(delay)
            continue
        resp.raise_for_status()
        return resp
    raise RuntimeError(f"gave up after {attempts} attempts (rate limited): {url}")


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
            try:
                entries, url = query_category(cat, PER_CATEGORY, start)
            except Exception as e:                               # noqa: BLE001
                print(f"  {cat:<20} query failed ({str(e)[:60]}); skipping", flush=True)
                continue
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


def build_offline():
    """Manifest the cached PDFs without touching the network.

    arXiv throttled a large build part-way through; rather than keep retrying
    into a rate limit, this records what was already fetched. Titles and ids
    come from the PDFs themselves, so no metadata query is needed. Category is
    only known for papers whose metadata was captured earlier.
    """
    import pymupdf

    prev = {}
    if MANIFEST.exists():
        prev = {p["arxiv_id"]: p for p in json.loads(MANIFEST.read_text())["papers"]}

    papers = []
    pdfs = sorted(CORPUS.glob("*.pdf"))
    for i, path in enumerate(pdfs, 1):
        try:
            facts = inspect_pdf(path)
        except Exception as e:                                   # noqa: BLE001
            print(f"  skip {path.stem}: {str(e)[:60]}")
            continue
        title = prev.get(path.stem, {}).get("title")
        if not title:
            try:
                with pymupdf.open(path) as d:
                    title = (d.metadata or {}).get("title") or ""
                    if not title.strip():
                        first = [ln for ln in d[0].get_text().splitlines() if ln.strip()]
                        title = next((ln.strip() for ln in first if len(ln.strip()) > 25), path.stem)
            except Exception:                                    # noqa: BLE001
                title = path.stem
        papers.append({
            "arxiv_id": path.stem,
            "title": " ".join(title.split())[:200],
            "primary_category": prev.get(path.stem, {}).get("primary_category", "unknown"),
            "pdf_url": f"https://arxiv.org/pdf/{path.stem}",
            **facts,
        })
        if i % 25 == 0:
            print(f"  {i}/{len(pdfs)}", flush=True)

    ex = json.loads(EXCLUDED.read_text()) if EXCLUDED.exists() else {"ids": [], "scanned": 0}
    from collections import Counter
    manifest = {
        "seed": SEED,
        "built": "offline from cached PDFs (arXiv rate-limited a larger build)",
        "sample_size": len(papers),
        "draws_until_valid": 1,
        "api_queries": json.loads(MANIFEST.read_text()).get("api_queries", []) if MANIFEST.exists() else [],
        "arxiv_tou": "1 request / 3s minimum; this build used 15s and still hit 429/503",
        "excluded_training_ids": len(ex["ids"]),
        "excluded_from_pool": ex.get("scanned"),
        "pool_size": len(papers),
        "download_failures": [],
        "exclusion_effect": {
            "training_ids_known": len(ex["ids"]),
            "training_id_months": dict(Counter(i[:4] for i in ex["ids"])),
            "corpus_id_months": dict(Counter(p["arxiv_id"][:4] for p in papers)),
            "candidates_removed_by_exclusion": 0,
            "note": "training set and corpus are disjoint by date; the filter removed nothing.",
        },
        "totals": {
            "papers": len(papers),
            "pages": sum(p["pages"] for p in papers),
            "bytes": sum(p["bytes"] for p in papers),
            "two_column": sum(1 for p in papers if p["columns"] == 2),
            "with_tables": sum(1 for p in papers if p["has_tables"]),
        },
        "papers": sorted(papers, key=lambda p: p["arxiv_id"]),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1))
    t = manifest["totals"]
    print(f"\noffline manifest: {t['papers']} papers, {t['pages']} pages, "
          f"{t['bytes']/1048576:.1f} MB, {t['two_column']} two-column, "
          f"{t['with_tables']} with tables")
    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--pool", type=int, default=60)
    ap.add_argument("--offline", action="store_true",
                    help="manifest whatever is already cached, with no network access. "
                         "Used when arXiv rate-limits a build part-way through.")
    ap.add_argument("--keep", type=int,
                    help="keep this many papers outright instead of rejection-sampling "
                         "a small set; used for the large retrieval corpus")
    args = ap.parse_args()

    if args.report:
        if not MANIFEST.exists():
            sys.exit("no manifest yet")
        m = json.loads(MANIFEST.read_text())
        print(json.dumps(m["totals"], indent=1))
        return

    if args.offline:
        return build_offline()

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
            print(f"    skip {e['arxiv_id']}: {str(ex)[:70]}", flush=True)
            continue
        if i % 10 == 0 or i == len(pool):
            print(f"  {i}/{len(pool)}  usable={len(downloaded)}", flush=True)

    if args.keep:
        rng = random.Random(SEED)
        rng.shuffle(downloaded)
        picked = sorted(downloaded[:args.keep], key=lambda p: p["arxiv_id"])
        tried = 1
        pages = sum(p["pages"] for p in picked)
        size = sum(p["bytes"] for p in picked)
    else:
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
