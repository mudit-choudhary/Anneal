"""Inspect the output quality of each RAG pipeline stage.

Usage (from anywhere, with globalragsetup_env active):

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

REPO_ROOT = Path(__file__).resolve().parent.parent

RULE = "=" * 72
THIN = "-" * 72


def use_package(name):
    """Put one manager's directory at the front of sys.path (each has its own
    flat config.py, so only ever load one per process)."""
    sys.path.insert(0, str(REPO_ROOT / name))


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


# --------------------------------------------------------------- stage 2
def cmd_chunks(args):
    use_package("embedding_manager")
    from chunking import chunk_text
    from config import PROCESSED_DIR, CHUNK_SIZE, CHUNK_OVERLAP_PCT

    p = Path(args.file)
    if not p.exists():
        for candidate in (Path(PROCESSED_DIR) / p.name, Path(PROCESSED_DIR) / f"{p.stem}.txt"):
            if candidate.exists():
                p = candidate
                break
        else:
            sys.exit(f"'{args.file}' not found (also looked in {PROCESSED_DIR})")

    size = args.chunk_size or CHUNK_SIZE
    overlap = args.overlap if args.overlap is not None else CHUNK_OVERLAP_PCT
    text = p.read_text(encoding="utf-8")
    chunks = chunk_text(text, size, overlap)

    lengths = [len(c) for c in chunks]
    print(f"{RULE}\nSTAGE 2 SUMMARY — {p.name}\n{RULE}")
    print(f"chunk_size={size} chars, overlap={overlap:.0%}  ->  {len(chunks)} chunks")
    print(f"Chunk lengths: min={min(lengths)}, avg={sum(lengths)//len(lengths)}, max={max(lengths)}")
    mid_sentence = sum(1 for c in chunks if c and c[-1].isalnum())
    print(f"Chunks ending mid-word/mid-sentence: {mid_sentence}/{len(chunks)}")

    show = chunks if args.show is None else chunks[:args.show]
    print(f"\nShowing {len(show)} of {len(chunks)} chunks:\n")
    for i, c in enumerate(show):
        print(f"┌── chunk {i} ({len(c)} chars) " + "─" * 40)
        print(c)
        print("└" + "─" * 66 + "\n")


# --------------------------------------------------------------- stage 3
def _get_chunks(question, k, files):
    use_package("rag_setup")
    import requests
    from config import EMBEDDING_URL

    try:
        r = requests.get(f"{EMBEDDING_URL}/get_chunks",
                         json={"query": question, "filenames": files}, timeout=90)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        sys.exit(f"Embedding service unreachable at {EMBEDDING_URL} — start it with\n"
                 f"  cd embedding_manager && python main.py\n({e})")


def cmd_retrieve(args):
    result = _get_chunks(args.question, args.k, args.files)
    docs = result["documents"][:args.k]
    metas = result["metadatas"][:args.k]
    dists = result["distances"][:args.k]

    print(f"{RULE}\nSTAGE 3 — top {len(docs)} chunks for: {args.question!r}\n{RULE}")
    print("(cosine distance: 0 = identical, ~1 = unrelated; "
          "watch for a big jump between consecutive ranks)\n")
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), 1):
        name = (meta or {}).get("filename", "?")
        print(f"#{i}  dist={dist:.4f}  {name} (chunk {meta.get('chunk_id', '?')})")
        print(f"    {doc[:400]}{' …' if len(doc) > 400 else ''}\n")


# --------------------------------------------------------------- stage 4
def cmd_answer(args):
    result = _get_chunks(args.question, None, args.files)  # rag_setup now on path
    import rag
    from config import N_RESULTS

    docs = result["documents"][:N_RESULTS]
    metas = result["metadatas"][:N_RESULTS]
    context = rag.build_context(docs, metas)

    print(f"{RULE}\nSTAGE 4 — backend: {args.backend}\n{RULE}")
    for i, m in enumerate(metas, 1):
        print(f"  [{i}] {(m or {}).get('filename', '?')}")
    print(f"{THIN}\n")
    if args.backend == "local":
        for delta in rag.stream_local(context, args.question):
            print(delta, end="", flush=True)
        print()
    else:
        print(rag.answer_gemini(context, args.question))


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

    c = sub.add_parser("chunks", help="stage 2: preview how a processed file would be chunked")
    c.add_argument("file", help="processed .txt path, filename, or bare stem")
    c.add_argument("--chunk-size", type=int, default=None)
    c.add_argument("--overlap", type=float, default=None, help="fraction, e.g. 0.2")
    c.add_argument("--show", type=int, default=10, help="chunks to print (omit for default 10)")
    c.set_defaults(fn=cmd_chunks)

    r = sub.add_parser("retrieve", help="stage 3: show ranked chunks for a question")
    r.add_argument("question")
    r.add_argument("-k", type=int, default=8)
    r.add_argument("--files", nargs="*", default=None, help="restrict to these filenames")
    r.set_defaults(fn=cmd_retrieve)

    a = sub.add_parser("answer", help="stage 4: full grounded answer")
    a.add_argument("question")
    a.add_argument("--backend", choices=["local", "gemini"], default="local")
    a.add_argument("--files", nargs="*", default=None)
    a.set_defaults(fn=cmd_answer)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
