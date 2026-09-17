"""Chunk shape on the round-2 corpus: 3 parsers x 3 chunkers, from cached parses.

    nice -n 10 python evals/scripts/shape_round2.py

Round 1 measured shape on 15 papers and retrieval on 103, so its shape-versus-
retrieval correlation compared different corpora. This measures the same shape
metrics (sizes, overlap, mid-sentence starts and ends, tables kept whole,
captions separated) on the 514 papers the round-2 retrieval run uses.

CPU only; safe to run beside the retrieval run at low priority. Table ground
truth comes from PyMuPDF geometry and is cached, and finished cells are
skipped, so it can be stopped and resumed.

Writes evals/Reports/shape_round2.json.
"""

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "app"))
sys.path.insert(0, str(REPO / "app" / "embedding_manager"))
sys.path.insert(0, str(REPO / "evals" / "scripts"))

from build_questions import write_json                                    # noqa: E402

CORPUS = REPO / "evals" / "corpus"
PARSED = REPO / "evals" / "parsed"
TABLES = PARSED / "geometry" / "table_signatures.json"
OUT = REPO / "evals" / "Reports" / "shape_round2.json"
PARSERS = ["oss_docling", "oss_pymupdf4llm", "current"]
CHUNKERS = ["fixed_token", "recursive_char", "grain_growth"]


def main():
    from run_matrix import CHUNKERS as IMPL, _CAPTION, _GG_META, measure, table_signatures
    from chunking import FENCED_TYPES, fence

    stems = sorted(p["arxiv_id"] for p in json.loads((CORPUS / "manifest.json").read_text())["papers"])

    TABLES.parent.mkdir(parents=True, exist_ok=True)
    tables = json.loads(TABLES.read_text()) if TABLES.exists() else {}
    todo = [s for s in stems if s not in tables]
    print(f"table ground truth: {len(stems) - len(todo)} cached, {len(todo)} to scan", flush=True)
    for i, stem in enumerate(todo, 1):
        tables[stem] = table_signatures(CORPUS / f"{stem}.pdf")
        if i % 25 == 0 or i == len(todo):
            write_json(TABLES, tables)
            print(f"  scanned {i}/{len(todo)}", flush=True)

    out = json.loads(OUT.read_text()) if OUT.exists() else {
        "corpus_papers": len(stems), "chunkers": CHUNKERS, "parsers": PARSERS, "cells": {}}
    for parser in PARSERS:
        if all(f"{parser}|{c}" in out["cells"] for c in CHUNKERS):
            continue
        docs = []
        for stem in stems:
            blocks = json.loads((PARSED / parser / f"{stem}.json").read_text(encoding="utf-8"))["blocks"]
            text = "\n\n".join(fence(b["type"], b["text"]) if b["type"] in FENCED_TYPES else b["text"]
                               for b in blocks if b.get("text"))
            docs.append({"stem": stem, "blocks": blocks, "text": text})
        # table captions found in this parser's own output, paired with the
        # geometric table rows of the same paper, as in run_matrix
        caption_pairs = [(b["text"], tables[d["stem"]]) for d in docs if tables.get(d["stem"])
                         for b in d["blocks"]
                         if b["type"] in ("caption", "paragraph") and _CAPTION.match(b["text"])]
        for chunker in CHUNKERS:
            key = f"{parser}|{chunker}"
            if key in out["cells"]:
                continue
            t0 = time.time()
            _GG_META["titles"].clear()
            _GG_META["stems"].clear()
            chunks = [c for d in docs for c in IMPL[chunker](d)]
            m = measure(chunks, [tables.get(d["stem"], []) for d in docs], caption_pairs)
            m["seconds"] = round(time.time() - t0, 1)
            if chunker == "grain_growth" and _GG_META["titles"]:
                m["title_fallback_pct"] = sum(1 for t, st in zip(_GG_META["titles"], _GG_META["stems"])
                                              if t == st) / len(_GG_META["titles"])
            out["cells"][key] = m
            write_json(OUT, out, indent=1)
            print(f"  {key:<32} {m['chunks']:>6} chunks  median {m['median_chars']:>5}  "
                  f"mid-start {100 * m['mid_start_pct']:.1f}%  mid-end {100 * m['mid_end_pct']:.1f}%  "
                  f"({m['seconds']:.0f}s)", flush=True)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
