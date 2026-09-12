"""Parser x chunker matrix over the fresh arXiv corpus.

    python evals/scripts/run_matrix.py                 # full run, checkpointed
    python evals/scripts/run_matrix.py --limit 3       # pilot
    python evals/scripts/run_matrix.py --only current  # one parser

Writes evals/Reports/results.json after every cell, so a crash costs at most
one cell. Nothing here estimates or interpolates: a cell that fails records its
error and is reported as NOT RUN.

Parsers
  raw_dump          PyMuPDF page.get_text(), no structure, no reading-order fix
  legacy            the parser as of commit 3a577c0, loaded from a git worktree
  oss_docling       Docling layout -> our stage-2 assembler
  oss_pymupdf4llm   PyMuPDF4LLM markdown -> blocks
  current           YOLOv11 layout + column-aware reading order

Chunkers
  fixed_token       fixed token window with stride overlap (bge tokenizer)
  recursive_char    recursive split on a separator hierarchy
  semantic          embedding-similarity breakpoints
  grain_growth      next-fit packing of layout regions, nucleating at section
                    headings, terminating at a barrier or the token budget,
                    tables and formulas pinned indivisible
"""

import argparse
import json
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "embedding_manager"))

CORPUS = REPO / "evals" / "corpus"
REPORTS = REPO / "evals" / "Reports"
RESULTS = REPORTS / "results.json"
LEGACY_TREE = REPO / ".worktrees" / "legacy"

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
TARGET_CHARS, MAX_CHARS = 1500, 2000
FIXED_TOKENS, FIXED_STRIDE = 256, 32       # window, overlap in tokens


# ============================================================ vram
class VramProbe:
    """Peak VRAM of this process while a block runs.

    nvidia-smi rather than torch: the current parser runs on onnxruntime, which
    torch's allocator counters never see.
    """

    def __init__(self, interval=0.25):
        self.interval, self.peak, self._stop = interval, 0.0, False
        self.baseline = None

    def _poll(self):
        import os
        pid = str(os.getpid())
        while not self._stop:
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout
                for line in out.splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) == 2 and parts[0] == pid:
                        self.peak = max(self.peak, float(parts[1]))
            except Exception:                                # noqa: BLE001
                pass
            time.sleep(self.interval)

    def _sample(self):
        import os
        pid = str(os.getpid())
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5).stdout
            for line in out.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 2 and parts[0] == pid:
                    return float(parts[1])
        except Exception:                                    # noqa: BLE001
            pass
        return 0.0

    def __enter__(self):
        # Models loaded by an earlier parser stay resident, so absolute peak
        # would credit every later parser with their memory. The baseline makes
        # the figure this parser's own additional cost.
        self.baseline = self._sample()
        self.peak = self.baseline
        self._t = threading.Thread(target=self._poll, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop = True
        self._t.join(timeout=2)


# ============================================================ parsers
def _blocks_from_pages(page_texts):
    return [{"text": t.strip(), "type": "paragraph", "page": i}
            for i, t in enumerate(page_texts) if t.strip()]


def parse_raw_dump(pdf):
    import pymupdf
    with pymupdf.open(pdf) as doc:
        pages = len(doc)
        texts = [doc[i].get_text() for i in range(pages)]
    blocks = _blocks_from_pages(texts)
    return pages, blocks


_LEGACY = {}


def _legacy_module():
    """Load commit 3a577c0's txt_processor with its registry call stubbed."""
    if _LEGACY:
        return _LEGACY["mod"]
    import importlib.util
    import types

    cfg = types.ModuleType("config")
    cfg.REGISTRY_URL = "http://127.0.0.1:0"
    sys.modules["config"] = cfg
    spec = importlib.util.spec_from_file_location(
        "legacy_txt", LEGACY_TREE / "parse_manager" / "txt_processor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class _Resp:
        @staticmethod
        def json():
            return {"success": True}

    mod.requests = types.SimpleNamespace(post=lambda *a, **k: _Resp())
    sys.modules.pop("config", None)
    _LEGACY["mod"] = mod
    return mod


def parse_legacy(pdf):
    """The 3a577c0 pipeline: page dump, then its regex paragraph builder."""
    import pymupdf
    import tempfile

    with pymupdf.open(pdf) as doc:
        pages = len(doc)
        raw = ""
        for n in range(pages):
            t = doc[n].get_text().strip()
            if t and t[-1] == ".":
                t += "\n"
            raw += f"\n\nPage: {n + 1}\n" + t

    mod = _legacy_module()
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "in.txt"
        src.write_text(raw, encoding="utf-8")
        out_json = Path(td) / "out.json"
        mod.process_pdf_txt(str(src), str(Path(td) / "out.txt"), str(out_json))
        data = json.loads(out_json.read_text(encoding="utf-8"))
    blocks = [{"text": p["text"], "type": "paragraph", "page": p.get("page", 1) - 1}
              for p in data.get("paragraphs", [])]
    return pages, blocks


_DOCLING_DIRS = {}


def parse_oss_docling(pdf):
    """Docling layout detection feeding our stage-2 assembler.

    Stage 2 is deliberately shared with `current`, so the cell isolates the
    difference in layout detection rather than in block assembly.
    """
    import tempfile
    sys.path.insert(0, str(REPO / "parse_manager"))
    import docling_backend
    import txt_processor
    import pymupdf

    with pymupdf.open(pdf) as doc:
        pages = len(doc)

    layout_pages = docling_backend.detect_pdf(str(pdf))
    with tempfile.TemporaryDirectory() as td:
        lj = Path(td) / f"{pdf.stem}.json"
        lj.write_text(json.dumps({"source_pdf": str(pdf), "backend": "docling",
                                  "num_pages": len(layout_pages),
                                  "pdf_pages": pages, "pages": layout_pages}), encoding="utf-8")
        txt_processor.process_layout_json(lj, output_txt=Path(td) / "o.txt",
                                          output_json=Path(td) / "o.json",
                                          update_registry=False)
        data = json.loads((Path(td) / "o.json").read_text(encoding="utf-8"))
    return pages, data.get("blocks", [])


_MD_HEAD = re.compile(r"^(#{1,6})\s+(.*)$")


def parse_oss_pymupdf4llm(pdf):
    """PyMuPDF4LLM markdown, mapped onto typed blocks.

    Markdown carries real structure: headings become section barriers and pipe
    tables become table blocks, so grain_growth has something to nucleate on.
    """
    import pymupdf4llm

    pages_md = pymupdf4llm.to_markdown(str(pdf), page_chunks=True, show_progress=False)
    blocks = []
    for pno, page in enumerate(pages_md):
        buf, in_table = [], []
        for line in (page.get("text") or "").split("\n"):
            head = _MD_HEAD.match(line)
            if head:
                if buf:
                    blocks.append({"text": "\n".join(buf).strip(), "type": "paragraph", "page": pno})
                    buf = []
                if in_table:
                    blocks.append({"text": "\n".join(in_table).strip(), "type": "table", "page": pno})
                    in_table = []
                blocks.append({"text": head.group(2).strip(), "type": "section", "page": pno})
            elif line.strip().startswith("|"):
                if buf:
                    blocks.append({"text": "\n".join(buf).strip(), "type": "paragraph", "page": pno})
                    buf = []
                in_table.append(line)
            elif not line.strip():
                if in_table:
                    blocks.append({"text": "\n".join(in_table).strip(), "type": "table", "page": pno})
                    in_table = []
                if buf:
                    blocks.append({"text": "\n".join(buf).strip(), "type": "paragraph", "page": pno})
                    buf = []
            else:
                buf.append(line)
        if in_table:
            blocks.append({"text": "\n".join(in_table).strip(), "type": "table", "page": pno})
        if buf:
            blocks.append({"text": "\n".join(buf).strip(), "type": "paragraph", "page": pno})
    return len(pages_md), [b for b in blocks if b["text"]]


_DETECTOR = {}


def parse_current(pdf):
    import tempfile
    sys.path.insert(0, str(REPO / "parse_manager"))
    import pdf_parser
    import txt_processor
    import layout_detector

    if "d" not in _DETECTOR:
        _DETECTOR["d"] = layout_detector.LayoutDetector()
    with tempfile.TemporaryDirectory() as td:
        pdf_parser.parse(str(pdf), detector=_DETECTOR["d"], out_dir=td)
        lj = Path(td) / f"{pdf.stem}.json"
        txt_processor.process_layout_json(lj, output_txt=Path(td) / "o.txt",
                                          output_json=Path(td) / "o.json",
                                          update_registry=False)
        data = json.loads((Path(td) / "o.json").read_text(encoding="utf-8"))
    return data.get("pdf_pages") or data.get("num_pages"), data.get("blocks", [])


PARSERS = {
    "raw_dump": parse_raw_dump,
    "legacy": parse_legacy,
    "oss_docling": parse_oss_docling,
    "oss_pymupdf4llm": parse_oss_pymupdf4llm,
    "current": parse_current,
}


# ============================================================ chunkers
_TOKENIZER = {}


def _tokenizer():
    if "t" not in _TOKENIZER:
        from transformers import AutoTokenizer
        _TOKENIZER["t"] = AutoTokenizer.from_pretrained(EMBED_MODEL)
    return _TOKENIZER["t"]


def chunk_fixed_token(doc):
    tok = _tokenizer()
    ids = tok.encode(doc["text"], add_special_tokens=False)
    step = FIXED_TOKENS - FIXED_STRIDE
    out = []
    for i in range(0, max(len(ids), 1), step):
        window = ids[i:i + FIXED_TOKENS]
        if not window:
            break
        out.append(tok.decode(window, skip_special_tokens=True).strip())
        if i + FIXED_TOKENS >= len(ids):
            break
    return [c for c in out if c]


def chunk_recursive_char(doc):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    sp = RecursiveCharacterTextSplitter(chunk_size=TARGET_CHARS, chunk_overlap=150,
                                        separators=["\n\n", "\n", " ", ""])
    return [c for c in sp.split_text(doc["text"]) if c.strip()]


_SEM = {}
_FENCED_BLOCK = re.compile(r"^(~{3,})\S*\n.*?\n\1$", re.M | re.S)


def _sem_model():
    if "m" not in _SEM:
        from sentence_transformers import SentenceTransformer
        _SEM["m"] = SentenceTransformer(EMBED_MODEL, device="cpu")
    return _SEM["m"]


def chunk_semantic(doc):
    """Breakpoints where consecutive units stop being similar.

    A fenced block is one indivisible unit and rejoining preserves line
    structure; without both, the splitter cuts inside a block and the closing
    bar lands in the next chunk.
    """
    import numpy as np
    from chunking import _FENCE_RUN, split_sentences

    text = doc["text"]
    units, last = [], 0
    for m in _FENCED_BLOCK.finditer(text):
        units += [s for s in split_sentences(text[last:m.start()]) if s.strip()]
        units.append(m.group(0))
        last = m.end()
    units += [s for s in split_sentences(text[last:]) if s.strip()]
    units = [u for u in units if u.strip()]
    if len(units) < 3:
        return [text] if text.strip() else []

    v = _sem_model().encode(units, normalize_embeddings=True, batch_size=64,
                            show_progress_bar=False)
    d = 1.0 - np.sum(v[:-1] * v[1:], axis=1)
    cutoff = float(np.mean(d) + 3 * np.std(d))

    def join(parts):
        out = ""
        for u in parts:
            if not out:
                out = u
                continue
            nl = _FENCE_RUN.match(u) or "\n" in u or out.rstrip().endswith("~")
            out += ("\n\n" if nl else " ") + u
        return out

    chunks, cur = [], []
    for i, u in enumerate(units):
        cur.append(u)
        if (i < len(d) and d[i] > cutoff) or sum(len(x) + 1 for x in cur) >= MAX_CHARS:
            chunks.append(join(cur))
            cur = []
    if cur:
        chunks.append(join(cur))
    return [c for c in chunks if c.strip()]


# grain_growth is the only chunker that builds a heading path, so the
# title-fallback rate is recorded here rather than inferred from chunk text.
_GG_META = {"titles": [], "stems": []}


def chunk_grain_growth(doc):
    from chunking import chunk_document
    chunks, _ = chunk_document(doc["blocks"], doc["stem"],
                               target_chars=TARGET_CHARS, max_chars=MAX_CHARS)
    for c in chunks:
        _GG_META["titles"].append(c["metadata"]["title"])
        _GG_META["stems"].append(doc["stem"])
    return [c["body"] for c in chunks if c["body"].strip()]


CHUNKERS = {
    "fixed_token": chunk_fixed_token,
    "recursive_char": chunk_recursive_char,
    "semantic": chunk_semantic,
    "grain_growth": chunk_grain_growth,
}


# ============================================================ metrics
_NORM = re.compile(r"\s+")
_CAPTION = re.compile(r"^\s*(table|tab\.)\s*(\d+)", re.I)


def norm(s):
    return _NORM.sub(" ", s or "").strip()


def overlap(a, b, cap=800):
    """Longest common suffix of `a` and prefix of `b`, whitespace-normalised.

    This is the single definition of overlap used everywhere in the report.
    """
    a, b = norm(a), norm(b)
    for k in range(min(len(a), len(b), cap), 0, -1):
        if a[-k:] == b[:k]:
            return k
    return 0


def table_signatures(pdf):
    """Row signatures of every table PyMuPDF can find.

    Ground truth comes from PyMuPDF geometry, never from a parser under test.
    """
    import pymupdf
    sigs = []
    try:
        with pymupdf.open(pdf) as doc:
            for page in doc:
                for t in page.find_tables().tables:
                    cells = []
                    for row in t.extract():
                        for c in row:
                            v = norm(c)
                            # distinctive values only: long enough to be
                            # unlikely to appear elsewhere by chance
                            if len(v) >= 6 and not v.replace(".", "").isdigit():
                                cells.append(v[:40])
                    uniq = list(dict.fromkeys(cells))[:8]
                    if len(uniq) >= 3:
                        sigs.append(uniq)
    except Exception:                                        # noqa: BLE001
        pass
    return sigs


def measure(chunks, tables_by_paper, caption_pairs):
    n = len(chunks)
    if not n:
        return {"chunks": 0}
    lengths = sorted(len(c) for c in chunks)
    ovs = [overlap(chunks[i], chunks[i + 1]) for i in range(n - 1)]
    nonzero = [o for o in ovs if o > 0]

    # the pipeline's own definition, so the evaluation cannot disagree with the
    # chunker about what a clean edge is: a fence bar is a real boundary
    from chunking import ends_cleanly, starts_cleanly
    starts = sum(1 for c in chunks if not starts_cleanly(c))
    ends = sum(1 for c in chunks if not ends_cleanly(c))

    normalised = [norm(c) for c in chunks]

    whole = 0
    total_tables = 0
    for sigs in tables_by_paper:
        for cells in sigs:
            total_tables += 1
            # "wholly inside one chunk": at least 80% of the table's
            # distinctive cell values appear in a single chunk
            need = max(2, int(0.8 * len(cells)))
            if any(sum(1 for v in cells if v in c) >= need for c in normalised):
                whole += 1

    # A table caption counts as kept with its table when some chunk holds both
    # the caption text and a majority of some table's distinctive cells. Figure
    # captions cannot be checked this way — a figure contributes no text — so
    # only table captions are counted.
    cap_ok = cap_total = 0
    for cap_text, sigs in caption_pairs:
        cap_total += 1
        cn = norm(cap_text)[:50]
        hit = False
        for c in normalised:
            if cn not in c:
                continue
            for cells in sigs:
                if sum(1 for v in cells if v in c) >= max(2, int(0.6 * len(cells))):
                    hit = True
                    break
            if hit:
                break
        cap_ok += hit

    return {
        "chunks": n,
        "avg_chars": round(statistics.mean(lengths), 1),
        "median_chars": int(statistics.median(lengths)),
        "min_chars": lengths[0],
        "max_chars": lengths[-1],
        "ov_zero_pct": (len(ovs) - len(nonzero)) / len(ovs) if ovs else 1.0,
        "ov_median_chars": int(statistics.median(nonzero)) if nonzero else 0,
        "ov_max_chars": max(nonzero) if nonzero else 0,
        "mid_start_pct": starts / n,
        "mid_end_pct": ends / n,
        "tables_detected": total_tables,
        "tables_whole_pct": whole / total_tables if total_tables else None,
        "captions_checked": cap_total,
        "captions_separated_pct": (cap_total - cap_ok) / cap_total if cap_total else None,
    }


# ============================================================ run
def load_manifest():
    m = json.loads((CORPUS / "manifest.json").read_text())
    return m, [CORPUS / f"{p['arxiv_id']}.pdf" for p in m["papers"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    manifest, pdfs = load_manifest()
    if args.limit:
        pdfs = pdfs[:args.limit]

    results = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "corpus": {"papers": len(pdfs), "manifest_seed": manifest["seed"],
                          "manifest_totals": manifest["totals"]},
               "config": {"target_chars": TARGET_CHARS, "max_chars": MAX_CHARS,
                          "fixed_tokens": FIXED_TOKENS, "fixed_stride": FIXED_STRIDE,
                          "embed_model": EMBED_MODEL,
                          "overlap_definition": "longest common suffix/prefix of adjacent "
                                                "chunks, whitespace-normalised"},
               "parsers": {}, "cells": {}}

    # Re-running a single parser must not discard the others.
    if args.only and RESULTS.exists():
        prev = json.loads(RESULTS.read_text())
        results["parsers"] = prev.get("parsers", {})
        results["cells"] = prev.get("cells", {})
        results["corpus"] = prev.get("corpus", results["corpus"])

    def save():
        RESULTS.write_text(json.dumps(results, indent=1))

    # ---- ground truth, independent of every parser ----
    print("Collecting table ground truth from PyMuPDF geometry")
    tables = {p.stem: table_signatures(p) for p in pdfs}
    caption_pairs = []   # filled per parser (captions come from parsed text)

    names = args.only or list(PARSERS)
    corpora = {}
    for name in names:
        fn = PARSERS[name]
        docs, pages, secs, peak = [], 0, 0.0, 0.0
        print(f"\n== parser {name}")
        for i, pdf in enumerate(pdfs, 1):
            t0 = time.time()
            try:
                with VramProbe() as probe:
                    npages, blocks = fn(pdf)
                peak = max(peak, probe.peak - (probe.baseline or 0.0))
            except Exception as e:                           # noqa: BLE001
                print(f"  [{i}/{len(pdfs)}] FAILED {pdf.stem}: {e}")
                results["parsers"].setdefault(name, {}).setdefault("failures", []).append(
                    {"paper": pdf.stem, "error": str(e)[:200]})
                continue
            dt = time.time() - t0
            secs += dt
            pages += npages or 0
            from chunking import FENCED_TYPES, fence
            text = "\n\n".join(
                fence(b["type"], b["text"]) if b["type"] in FENCED_TYPES else b["text"]
                for b in blocks if b.get("text"))
            docs.append({"stem": pdf.stem, "blocks": blocks, "text": text})
            print(f"  [{i}/{len(pdfs)}] {pdf.stem:<12} {npages:>3}p {len(text)//1024:>4}KB "
                  f"{dt:5.1f}s", flush=True)
        corpora[name] = docs
        barriers = sum(1 for d in docs for b in d["blocks"] if b["type"] == "section")
        # Two of our twelve classes have no equivalent in some parsers'
        # taxonomies. Without `title` the heading path falls back to the
        # filename; without `authors` names are embedded as body prose.
        counts = {t: sum(1 for d in docs for b in d["blocks"] if b["type"] == t)
                  for t in ("title", "authors", "caption", "table", "formula",
                            "footnote", "list", "paragraph")}
        docs_with_title = sum(
            1 for d in docs if any(b["type"] == "title" for b in d["blocks"]))
        results["parsers"][name] = {
            **results["parsers"].get(name, {}),
            "pages": pages, "corpus_bytes": sum(len(d["text"].encode()) for d in docs),
            "seconds": round(secs, 1),
            "sec_per_page": round(secs / pages, 4) if pages else None,
            "vram_rise_mb": round(peak, 1),
            "documents": len(docs),
            "section_barriers": barriers,
            "block_types": counts,
            "documents_with_title": docs_with_title,
        }
        save()

        # captions: table captions found in this parser's own output, paired
        # with the geometric table rows they name
        pairs = []
        for d in docs:
            sigs = tables.get(d["stem"], [])
            if not sigs:
                continue
            for b in d["blocks"]:
                if b["type"] in ("caption", "paragraph") and _CAPTION.match(b["text"]):
                    pairs.append((b["text"], sigs))
        caption_pairs = pairs

        for cname, cfn in CHUNKERS.items():
            key = f"{name}|{cname}"
            t0 = time.time()
            _GG_META["titles"].clear()
            _GG_META["stems"].clear()
            try:
                allc = []
                for d in docs:
                    allc.extend(cfn(d))
                m = measure(allc, [tables.get(d["stem"], []) for d in docs], caption_pairs)
                m["seconds"] = round(time.time() - t0, 1)
                m["degenerate"] = (cname == "grain_growth" and barriers == 0)
                if cname == "grain_growth" and _GG_META["titles"]:
                    fell_back = sum(1 for t, st in zip(_GG_META["titles"], _GG_META["stems"])
                                    if t == st)
                    m["title_fallback_pct"] = fell_back / len(_GG_META["titles"])
                results["cells"][key] = m
                print(f"    {cname:<16} {m['chunks']:>6} chunks  "
                      f"median {m.get('median_chars', 0):>5}"
                      f"{'  DEGENERATE' if m['degenerate'] else ''}", flush=True)
            except Exception as e:                           # noqa: BLE001
                results["cells"][key] = {"error": str(e)[:300], "chunks": 0}
                print(f"    {cname:<16} FAILED: {e}", flush=True)
            save()

    save()
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
