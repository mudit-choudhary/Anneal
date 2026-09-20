"""Inspect the output quality of each RAG pipeline stage.

Usage (from anywhere, with annealenv active):

    python scripts/rag_inspect.py parse     Some_Paper.pdf [--pages 3] [--show 12]
    python scripts/rag_inspect.py chunks    Some_Paper [--chunk-size 512] [--overlap 0.2] [--show 10]
    python scripts/rag_inspect.py retrieve  "your question" [-k 8] [--files f1.txt f2.txt]
    python scripts/rag_inspect.py answer    "your question" [--backend local|gemini]

Stages map to the RAG cycle:
    parse    -> stage 1: PDF -> layout regions -> tagged text (no services needed)
    chunks   -> stage 2: how the embedder would split a processed file (no services)
    retrieve -> stage 3: ranked chunks for a question (needs embedding service :4001)
    answer   -> stage 4: full grounded answer (needs embedding service + Ollama)
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent

RULE = "=" * 72
THIN = "-" * 72


def use_package(name):
    """Put one manager's directory at the front of sys.path (each has its own
    flat config.py, so only ever load one per process)."""
    sys.path.insert(0, str(APP_ROOT / name))


# --------------------------------------------------------------- stage 1
def cmd_parse(args):
    use_package("parse_manager")
    from pdf_parser import parse, resolve_pdf
    from txt_processor import process_layout_json, resolve_layout_json
    import json

    pdf = resolve_pdf(args.pdf)
    print(f"Parsing {pdf.name}" + (f" (first {args.pages} pages)" if args.pages else ""))
    layout_json = parse(pdf, max_pages=args.pages)
    txt_path = process_layout_json(resolve_layout_json(layout_json))
    json_path = txt_path.with_suffix(".json")

    layout = json.loads(Path(layout_json).read_text())
    labels = Counter(r["label"] for p in layout["pages"] for r in p["regions"])
    fallbacks = sum(1 for p in layout["pages"] for r in p["regions"] if r.get("fallback"))
    swallowed = sum(len(p.get("swallowed_text", [])) for p in layout["pages"])

    processed = json.loads(json_path.read_text())
    blocks = processed["blocks"]

    print(f"\n{RULE}\nSTAGE 1 SUMMARY — {pdf.name}\n{RULE}")
    print(f"Pages: {layout['num_pages']}   Blocks: {len(blocks)}")
    print("Region detections:", dict(labels.most_common()))
    print(f"Fallback Text regions (words YOLO missed): {fallbacks}")
    print(f"Words swallowed (inside Picture/header/footer): {swallowed}")
    d = processed["dropped"]
    print(f"Dropped page-headers: {len(d['page_headers'])}, "
          f"page-footers: {len(d['page_footers'])}, "
          f"picture-text words: {len(d['picture_text'])}")

    print(f"\nFirst {args.show} blocks of assembled text:\n{THIN}")
    for b in blocks[:args.show]:
        text = b["text"] if len(b["text"]) <= 300 else b["text"][:300] + " …"
        print(f"[p{b['page']} | {b['type']}] {text}\n")
    print(f"{THIN}\nFull outputs:\n  layout : {layout_json}\n  text   : {txt_path}\n  blocks : {json_path}")


# --------------------------------------------------------------- stage 1b: tables
NUMBER = None


def cmd_tables(args):
    """Verify table extraction: for every Table region, save a crop image and
    check that every numeric token visible in the PDF box survived into the
    extracted text."""
    import json
    import re

    import fitz

    use_package("parse_manager")
    from pdf_parser import resolve_pdf
    from txt_processor import resolve_layout_json

    sys.path.insert(0, str(APP_ROOT))
    from common.paths import DEBUG_DIR

    number = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
    pdf = resolve_pdf(args.pdf)
    layout = json.loads(Path(resolve_layout_json(pdf.stem)).read_text())
    out_dir = DEBUG_DIR / "tables" / pdf.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf)

    print(f"{RULE}\nTABLE CHECK — {pdf.name}\n{RULE}")
    total = ok = 0
    for page_entry in layout["pages"]:
        page = doc[page_entry["page"]]
        for i, region in enumerate(r for r in page_entry["regions"] if r["label"] == "Table"):
            total += 1
            rect = fitz.Rect(*region["bbox"])
            png = out_dir / f"table_p{page_entry['page'] + 1}_{i + 1}.png"
            page.get_pixmap(clip=rect, dpi=150).save(str(png))
            pdf_numbers = sorted(number.findall(" ".join(w[4] for w in page.get_text("words", clip=rect))))
            extracted = "\n".join(region.get("lines", []))
            got_numbers = sorted(number.findall(extracted))
            missing = sorted(set(pdf_numbers) - set(got_numbers))
            status = "OK " if not missing else "MISSING"
            ok += not missing
            print(f"{status} p{page_entry['page'] + 1} table {i + 1}: {len(pdf_numbers)} numbers in PDF box, "
                  f"{len(got_numbers)} extracted{'' if not missing else ' — missing ' + ', '.join(missing[:10])}")
            print(f"     crop: {png}")
            for ln in extracted.splitlines()[:6]:
                print(f"     | {ln[:110]}")
    print(f"{THIN}\n{ok}/{total} tables preserve every number. Compare the crops with the text above.")


# --------------------------------------------------------------- stage 2
CHARS_PER_TOKEN = 5.1   # measured for bge tokenizer on paper text
TOKEN_WINDOW = 512


def cmd_chunks(args):
    use_package("embedding_manager")
    from grain_growth import chunk_file
    from config import PROCESSED_DIR, CHUNK_TARGET_CHARS, CHUNK_MAX_CHARS

    p = Path(args.file)
    if not p.exists() or p.suffix != ".json":
        for candidate in (Path(PROCESSED_DIR) / p.name, Path(PROCESSED_DIR) / f"{p.stem}.json"):
            if candidate.exists() and candidate.suffix == ".json":
                p = candidate
                break
        else:
            sys.exit(f"'{args.file}' not found as a processed .json (looked in {PROCESSED_DIR})")

    target = args.target or CHUNK_TARGET_CHARS
    maximum = args.max or CHUNK_MAX_CHARS
    chunks, skipped = chunk_file(p, target, maximum)
    if not chunks:
        sys.exit("No embeddable blocks in this file.")

    lengths = [c["metadata"]["n_chars"] for c in chunks]
    full = [len(c["text"]) for c in chunks]
    prose = [c for c in chunks if c["metadata"]["block_types"] == "paragraph"]
    mid = sum(1 for c in prose if c["body"].rstrip()[-1] not in ".!?\"')]”")
    sections = {c["metadata"]["section"] for c in chunks}
    over = sum(1 for n in full if n / CHARS_PER_TOKEN > TOKEN_WINDOW * 0.95)

    print(f"{RULE}\nSTAGE 2 SUMMARY — {p.name}\n{RULE}")
    print(f"target={target} chars, max={maximum} chars  ->  {len(chunks)} chunks "
          f"({len(skipped)} blocks not embedded: authors, footnotes, bare equation numbers)")
    print(f"Body length: min={min(lengths)}, avg={sum(lengths)//len(lengths)}, max={max(lengths)}"
          f"   (embedded text incl. heading: max={max(full)} chars ≈ {int(max(full)/CHARS_PER_TOKEN)} tokens)")
    print(f"Chunks near/over the {TOKEN_WINDOW}-token window: {over}")
    print(f"Prose chunks ending mid-sentence: {mid}/{len(prose)}")
    print(f"Sections covered: {len(sections)}")
    types = Counter(c["metadata"]["block_types"] for c in chunks)
    print("By block type:", dict(types.most_common()))

    show = chunks if args.show is None else chunks[:args.show]
    print(f"\nShowing {len(show)} of {len(chunks)} chunks:\n")
    for c in show:
        m = c["metadata"]
        pages = f"p{m['page_start']}" + (f"-{m['page_end']}" if m["page_end"] != m["page_start"] else "")
        print(f"┌── chunk {m['chunk_index']} · {m['n_chars']} chars · {pages} · {m['block_types']} "
              + "─" * 20)
        print(c["text"])
        print("└" + "─" * 66 + "\n")


# --------------------------------------------------------------- stage 3
def _rag():
    sys.path.insert(0, str(APP_ROOT))
    use_package("rag_setup")
    import rag
    return rag


def cmd_retrieve(args):
    rag = _rag()
    import requests
    try:
        results = rag.fetch_chunks(args.question, args.files, args.k, ["papers", "chats"] if args.chats else ["papers"])
    except requests.RequestException as e:
        sys.exit("Embedding service unreachable — start it with  cd embedding_manager && python main.py\n"
                 f"({e})")

    print(f"{RULE}\nSTAGE 3 — top {len(results)} chunks for: {args.question!r}\n{RULE}")
    print("(cosine distance: 0 = identical, ~1 = unrelated; "
          "watch for a big jump between consecutive ranks)\n")
    for i, r in enumerate(results, 1):
        meta = r.get("metadata") or {}
        if r.get("source") == "chats":
            print(f"#{i}  dist={r['distance']:.4f}  [saved chat] {meta.get('title', '?')}")
        else:
            name = meta.get("title") or meta.get("filename", "?")
            where = f" › {meta['section']}" if meta.get("section") else ""
            page = f"  p.{meta['page_start'] + 1}" if meta.get("page_start") is not None else ""
            print(f"#{i}  dist={r['distance']:.4f}  {name}{where}{page}  (chunk {meta.get('chunk_index', '?')})")
        text = r["text"]
        print(f"    {text[:400]}{' …' if len(text) > 400 else ''}\n")


# --------------------------------------------------------------- stage 4
def cmd_answer(args):
    rag = _rag()
    from common import settings as settings_store
    import requests

    cfg = settings_store.load()
    if args.backend:
        cfg["llm"]["backend"] = args.backend
    try:
        context, sources, warnings = rag.retrieve(args.question, args.files, args.web, cfg)
    except requests.RequestException as e:
        sys.exit(f"Embedding service unreachable ({e})")

    print(f"{RULE}\nSTAGE 4 — backend: {cfg['llm']['backend']}\n{RULE}")
    for w in warnings:
        print(f"  ! {w}")
    for s in sources:
        print(f"  [{s['n']}] {s['label']}")
    print(f"{THIN}\n")
    for delta in rag.answer_stream(context, args.question, cfg):
        print(delta, end="", flush=True)
    print()


# ---------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("parse", help="stage 1: parse a PDF and summarize layout quality")
    p.add_argument("pdf", help="PDF path, or bare filename in data/raw_pdfs/")
    p.add_argument("--pages", type=int, default=None, help="limit to first N pages")
    p.add_argument("--show", type=int, default=12, help="blocks of text to print")
    p.set_defaults(fn=cmd_parse)

    t = sub.add_parser("tables", help="stage 1b: verify table extraction (crops + number check)")
    t.add_argument("pdf", help="PDF path or bare filename; its layout JSON must exist in data/parsed/")
    t.set_defaults(fn=cmd_tables)

    c = sub.add_parser("chunks", help="stage 2: preview how a processed file would be chunked")
    c.add_argument("file", help="processed .json path, filename, or bare stem")
    c.add_argument("--target", type=int, default=None, help="pack paragraphs up to N chars")
    c.add_argument("--max", type=int, default=None, help="split a single block only beyond N chars")
    c.add_argument("--show", type=int, default=10, help="chunks to print")
    c.set_defaults(fn=cmd_chunks)

    r = sub.add_parser("retrieve", help="stage 3: show ranked chunks for a question")
    r.add_argument("question")
    r.add_argument("-k", type=int, default=8)
    r.add_argument("--files", nargs="*", default=None, help="restrict to these filenames")
    r.add_argument("--chats", action="store_true", help="also search saved conversations")
    r.set_defaults(fn=cmd_retrieve)

    a = sub.add_parser("answer", help="stage 4: full grounded answer")
    a.add_argument("question")
    a.add_argument("--backend", choices=["local", "openai"], default=None,
                   help="override the configured backend for this run")
    a.add_argument("--files", nargs="*", default=None)
    a.add_argument("--web", action="store_true", help="also fetch web pages for the question")
    a.set_defaults(fn=cmd_answer)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
