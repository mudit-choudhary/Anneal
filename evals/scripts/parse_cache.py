"""Parse every corpus PDF with every parser once, and cache the blocks.

    python evals/scripts/parse_cache.py [--only current] [--limit N]

The retrieval matrix runs 4 chunking strategies against 3 parsers. Parsing
inside that loop would parse each paper 4 times per parser for no reason —
about 8 wasted hours at this corpus size. The parsers run once here and the
chunkers read the cache.

Writes evals/parsed/<parser>/<stem>.json holding the typed blocks, and
evals/parsed/<parser>/_stats.json holding pages, bytes, seconds and VRAM.
Re-running skips papers already cached.
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "app"))
sys.path.insert(0, str(REPO / "evals" / "scripts"))

CORPUS = REPO / "evals" / "corpus"
PARSED = REPO / "evals" / "parsed"

PARSERS = ["oss_docling", "oss_pymupdf4llm", "current"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    # the parser implementations and the VRAM probe already exist in the matrix
    from run_matrix import PARSERS as IMPL, VramProbe
    from build_questions import write_json

    manifest = json.loads((CORPUS / "manifest.json").read_text())
    pdfs = [CORPUS / f"{p['arxiv_id']}.pdf" for p in manifest["papers"]]
    if args.limit:
        pdfs = pdfs[:args.limit]

    for name in (args.only or PARSERS):
        out = PARSED / name
        out.mkdir(parents=True, exist_ok=True)
        stats_path = out / "_stats.json"
        stats = json.loads(stats_path.read_text()) if stats_path.exists() else {
            "parser": name, "pages": 0, "bytes": 0, "seconds": 0.0,
            "vram_rise_mb": 0.0, "documents": 0, "failures": []}

        fn = IMPL[name]
        todo = [p for p in pdfs if not (out / f"{p.stem}.json").exists()]
        print(f"\n== {name}: {len(todo)} to parse, "
              f"{len(pdfs) - len(todo)} already cached", flush=True)

        for i, pdf in enumerate(todo, 1):
            t0 = time.time()
            try:
                with VramProbe() as probe:
                    pages, blocks = fn(pdf)
                rise = probe.peak - (probe.baseline or 0.0)
            except Exception as e:                               # noqa: BLE001
                print(f"  [{i}/{len(todo)}] FAILED {pdf.stem}: {str(e)[:90]}", flush=True)
                stats["failures"].append({"paper": pdf.stem, "error": str(e)[:200]})
                write_json(stats_path, stats, indent=1)
                continue
            dt = time.time() - t0
            write_json(out / f"{pdf.stem}.json", {"stem": pdf.stem, "pages": pages, "blocks": blocks})
            stats["pages"] += pages or 0
            stats["seconds"] += dt
            stats["documents"] += 1
            stats["bytes"] += sum(len(b.get("text", "").encode()) for b in blocks)
            stats["vram_rise_mb"] = max(stats["vram_rise_mb"], rise)
            if i % 10 == 0 or i == len(todo):
                write_json(stats_path, stats, indent=1)
                print(f"  [{i}/{len(todo)}] {pdf.stem} {pages}p {dt:.1f}s "
                      f"(total {stats['seconds']/60:.0f} min)", flush=True)

        stats["sec_per_page"] = (stats["seconds"] / stats["pages"]) if stats["pages"] else None
        write_json(stats_path, stats, indent=1)
        print(f"  {name}: {stats['documents']} docs, {stats['pages']} pages, "
              f"{stats['seconds']/60:.1f} min, {stats['vram_rise_mb']:.0f} MB peak rise")


if __name__ == "__main__":
    main()
