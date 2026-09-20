"""Legacy vs current parsing, across the main chunking strategies.

    python scripts/chunking_bench.py --n 12
    python scripts/chunking_bench.py --n 12 --pdf-dir /path/to/PDFs --seed 7
    python scripts/chunking_bench.py --report        # reprint the last run

Fully sandboxed: reads PDFs from an external directory and writes only under
a scratch directory (default /tmp/chunking_bench). It never touches
`data/`, the registry, or the vector store, and it starts no services.

Companion to `scripts/retrieval_eval.py`, which measures *retrieval quality*
against a question set. This script measures the **shape** of what the two
pipelines produce, which needs no ground truth and so is fast and repeatable.

Three parsers
-------------
`RawDumpParser` PyMuPDF page text and nothing else — no paragraph rules, no
                cleanup. The floor both other arms are measured against.
`LegacyParser`  reproduces commit 057ce0e9 (identical to c41e756 for these
                files): PyMuPDF `page.get_text()` per page prefixed "Page: N",
                then `txt_processor.process_pdf_txt`'s regex paragraph builder.
`CurrentParser` runs today's pipeline: YOLO layout detection through
                `parse_manager`, then `txt_processor` into typed blocks.

Both expose the same two outputs — `parsed` (the JSON a pipeline stores) and
`chunks` (what would be embedded) — so they can be compared directly.

Three chunking strategies, with variations
------------------------------------------
recursive   the classic fixed-size character splitter. Variations sweep the
            two knobs everyone tunes: chunk size and overlap.
semantic    breakpoints where consecutive sentences stop being similar.
            Its unique knob is the *threshold type* — percentile, standard
            deviation, or interquartile — so all three are run.
structure   today's approach — Grain-Growth Chunking: next-fit packing under
            structural barriers. A chunk grows by whole blocks until the budget
            is reached or it meets a section heading or standalone block, and a
            closed chunk is never reopened. Its unique knob is the heading-path
            prefix, so it is run with and without.

recursive and semantic consume flat text, so they run on both parsers.
structure consumes typed blocks; on the legacy parser those blocks carry no
headings or types, which is itself the point of the comparison.
"""

import argparse
import json
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))
sys.path.insert(0, str(APP_ROOT / "embedding_manager"))

DEFAULT_PDF_DIR = Path("/media/mudit/DarkDwine1/ResearchPapersYOLO_FT/PDFs")
SCRATCH = Path(os.environ.get("BENCH_DIR", "/tmp/chunking_bench"))
RESULTS = SCRATCH / "results.json"

EMBED_MODEL = "BAAI/bge-base-en-v1.5"


# ============================================================ parsers
class LegacyParser:
    """The pipeline as of 057ce0e9: no layout model, regex paragraph rules."""

    name = "legacy"
    label = "legacy (page dump + regex paragraphs)"

    def parse(self, pdf_path):
        import fitz

        doc = fitz.open(pdf_path)
        pages = len(doc)
        raw = ""
        for n in range(pages):
            text = doc[n].get_text().strip()
            if text and text[-1] == ".":
                text += "\n"
            raw += f"\n\nPage: {n + 1}\n" + text
        doc.close()

        paragraphs = self._paragraphs(raw)
        parsed = {
            "filename": Path(pdf_path).stem,
            "num_pages": pages,
            "total_paragraphs": len(paragraphs),
            "total_words": sum(p["word_count"] for p in paragraphs),
            "paragraphs": paragraphs,
        }
        blocks = [{"text": p["text"], "type": "paragraph", "page": p["page"]} for p in paragraphs]
        text = "\n\n".join(p["text"] for p in paragraphs)
        return {"pages": pages, "parsed": parsed, "blocks": blocks, "text": text}

    @staticmethod
    def _paragraphs(content):
        """Verbatim port of txt_processor.process_pdf_txt at 057ce0e9.

        The hardcoded `title_pattern` in the original matched one specific
        paper's running header; it is kept, because removing it would be
        improving the baseline rather than reproducing it.
        """
        content = re.sub(r'Page:\s*\d+\n[^.\n]*?\n\d+\n', '', content, flags=re.MULTILINE)
        content = re.sub(r'(?m)^Page\s*\d+\s*$', '', content)

        paragraphs, current_para = [], []
        current_page = 1
        title_pattern = r'Completion by Comprehension.*Understanding'

        def flush(min_words):
            nonlocal current_para
            if current_para:
                para_text = ' '.join(current_para).strip()
                if para_text and len(para_text.split()) > min_words:
                    paragraphs.append({'text': para_text, 'page': current_page,
                                       'word_count': len(para_text.split())})
                current_para = []

        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
            words = len(line.split())
            if re.match(r'Page\s*\d+', line):
                flush(5)
                continue
            if re.search(title_pattern, line):
                continue

            is_break = False
            if (re.match(r'^\d+[\.\s]', line) or re.match(r'^[A-Z][a-z]+:', line)
                    or re.match(r'^Fig\.', line) or re.match(r'^Table\s', line)):
                is_break = True
            elif words < 20 and current_para and current_para[-1].rstrip().endswith('.'):
                is_break = True
            elif words < 10:
                is_break = True

            if is_break and current_para:
                flush(3)
            current_para.append(line)

        flush(3)
        return paragraphs


class RawDumpParser:
    """PyMuPDF page text and nothing else.

    The legacy pipeline's regex paragraph builder is a processing step layered
    on top of the dump, and porting it means porting its quirks — including a
    running-header pattern hardcoded to one specific paper. This arm removes
    that question entirely: extract the text, add nothing, judge the dump on
    its own. It is the true floor for both other arms.
    """

    name = "raw-dump"
    label = "raw PyMuPDF page dump, no processing"

    def parse(self, pdf_path):
        import fitz

        with fitz.open(pdf_path) as doc:
            pages = len(doc)
            page_texts = [doc[n].get_text() for n in range(pages)]

        # blocks are pages, because a dump has no smaller unit it can defend
        blocks = [{"text": t.strip(), "type": "paragraph", "page": i}
                  for i, t in enumerate(page_texts) if t.strip()]
        text = "\n\n".join(b["text"] for b in blocks)
        parsed = {
            "filename": Path(pdf_path).stem,
            "num_pages": pages,
            "pdf_pages": pages,
            "total_words": len(text.split()),
            "pages_text": page_texts,
        }
        return {"pages": pages, "parsed": parsed, "blocks": blocks, "text": text}


class CurrentParser:
    """Today's pipeline: YOLO layout detection, then typed blocks."""

    name = "current"
    label = "current (YOLO layout + typed blocks)"

    def __init__(self):
        sys.path.insert(0, str(APP_ROOT / "parse_manager"))
        import layout_detector
        self._detector = layout_detector.LayoutDetector()

    def parse(self, pdf_path):
        import fitz

        import pdf_parser
        import txt_processor

        stage1 = SCRATCH / "parsed"
        stage2 = SCRATCH / "processed"
        stage1.mkdir(parents=True, exist_ok=True)
        stage2.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(pdf_path)
        real_pages = len(doc)
        doc.close()

        pdf_parser.parse(str(pdf_path), detector=self._detector, out_dir=str(stage1))
        stem = Path(pdf_path).stem
        txt_processor.process_layout_json(
            stage1 / f"{stem}.json",
            output_txt=stage2 / f"{stem}.txt",
            output_json=stage2 / f"{stem}.json",
            update_registry=False)

        parsed = json.loads((stage2 / f"{stem}.json").read_text(encoding="utf-8"))
        blocks = parsed.get("blocks", [])
        # The flat text carries the same fences the structure chunker emits, so
        # a fixed-size splitter run over it can be caught cutting through one.
        from grain_growth import FENCED_TYPES, fence
        parts = []
        for b in blocks:
            t = b.get("text", "")
            if not t:
                continue
            parts.append(fence(b["type"], t) if b["type"] in FENCED_TYPES else t)
        return {"pages": real_pages, "parsed": parsed, "blocks": blocks,
                "text": "\n\n".join(parts)}


# ============================================================ chunkers
def split_sentences(text):
    from grain_growth import split_sentences as _s
    return _s(text)


class RecursiveChunker:
    """The classic fixed-size splitter. Knobs: size and overlap."""

    strategy = "recursive"

    def __init__(self, size, overlap):
        self.size, self.overlap = size, overlap
        pct = round(100 * overlap / size)
        self.name = f"recursive-{size}-ov{pct}"
        self.label = f"{size} chars, {pct}% overlap"

    def chunk(self, parsed):
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.size, chunk_overlap=self.overlap,
            separators=["\n\n", "\n", " ", ""])
        return splitter.split_text(parsed["text"])


class SemanticChunker:
    """Break where consecutive sentences stop being similar.

    The threshold type is what distinguishes semantic implementations from one
    another, so all three of the common ones are run. Sentence embeddings are
    computed once per document and shared across the three.
    """

    strategy = "semantic"
    _model = None
    _cache = {}

    def __init__(self, mode, max_chars=2000):
        self.mode, self.max_chars = mode, max_chars
        self.name = f"semantic-{mode}"
        self.label = f"breakpoints by {mode}"

    @classmethod
    def model(cls):
        if cls._model is None:
            from sentence_transformers import SentenceTransformer
            cls._model = SentenceTransformer(EMBED_MODEL, device=os.environ.get("BENCH_DEVICE", "cpu"))
        return cls._model

    @staticmethod
    def _units(text):
        """Sentences, except that a fenced block is one indivisible unit.

        Without this the sentence splitter cuts inside a block — an equation
        number like "(72)" reads as the start of a new sentence because "(" is
        in its lookahead — and the closing bar ends up in the next chunk. Any
        production semantic chunker has to mask fenced regions first.
        """
        from grain_growth import FENCE_CHAR
        block = re.compile(rf"^({re.escape(FENCE_CHAR)}{{3,}})\S*\n.*?\n\1$", re.M | re.S)
        out, last = [], 0
        for m in block.finditer(text):
            out += [s for s in split_sentences(text[last:m.start()]) if s.strip()]
            out.append(m.group(0))
            last = m.end()
        out += [s for s in split_sentences(text[last:]) if s.strip()]
        return out

    @staticmethod
    def _join(units):
        """Rejoin preserving line structure.

        A plain " ".join puts an opening bar mid-line, where it no longer
        matches "^~~~", so an intact block reads as severed.
        """
        from grain_growth import FENCE_CHAR
        from grain_growth.chunker import _FENCE_RUN  # private: fence run regex
        out = ""
        for u in units:
            if not out:
                out = u
                continue
            needs_line = _FENCE_RUN.match(u) or "\n" in u or out.rstrip().endswith(FENCE_CHAR)
            out += ("\n\n" if needs_line else " ") + u
        return out

    def chunk(self, parsed):
        key = (parsed.get("_key"), len(parsed["text"]))
        if key in SemanticChunker._cache:
            sentences, distances = SemanticChunker._cache[key]
        else:
            sentences = self._units(parsed["text"])
            if len(sentences) < 3:
                return [parsed["text"]] if parsed["text"].strip() else []
            import numpy as np
            vectors = self.model().encode(sentences, normalize_embeddings=True,
                                          batch_size=64, show_progress_bar=False)
            distances = 1.0 - np.sum(vectors[:-1] * vectors[1:], axis=1)
            SemanticChunker._cache[key] = (sentences, distances)

        if len(sentences) < 3:
            return [parsed["text"]] if parsed["text"].strip() else []

        import numpy as np
        if self.mode == "percentile":
            cutoff = float(np.percentile(distances, 95))
        elif self.mode == "stddev":
            cutoff = float(np.mean(distances) + 3 * np.std(distances))
        elif self.mode == "interquartile":
            q1, q3 = np.percentile(distances, [25, 75])
            cutoff = float(q3 + 1.5 * (q3 - q1))
        else:
            raise ValueError(self.mode)

        chunks, cur = [], []
        for i, sentence in enumerate(sentences):
            cur.append(sentence)
            over_budget = sum(len(s) + 1 for s in cur) >= self.max_chars
            breakpoint_here = i < len(distances) and distances[i] > cutoff
            if breakpoint_here or over_budget:
                chunks.append(self._join(cur))
                cur = []
        if cur:
            chunks.append(self._join(cur))
        return [c for c in chunks if c.strip()]


class StructureChunker:
    """Today's approach: pack whole typed blocks, never cross a section.

    Its unique knob is the heading-path prefix, so it is run both ways.
    """

    strategy = "structure"

    def __init__(self, target, headings=True):
        self.target, self.headings = target, headings
        self.name = f"structure-{target}" + ("" if headings else "-noheading")
        self.label = f"target {target} chars" + ("" if headings else ", no heading prefix")

    def chunk(self, parsed):
        from grain_growth import chunk_document
        blocks = parsed["blocks"]
        if not blocks:
            return []
        chunks, _skipped = chunk_document(blocks, parsed.get("_key", "doc"),
                                          target_chars=self.target,
                                          max_chars=int(self.target * 4 / 3))
        key = "text" if self.headings else "body"
        return [c[key] for c in chunks if c.get(key, "").strip()]


class ProductionChunker:
    """Today's pipeline exactly as shipped — `chunk_file`'s own defaults.

    Deliberately takes no arguments. `StructureChunker(1500)` happens to
    resolve to the same numbers today, but it reaches them by computing
    max_chars from target; this arm calls the production entry point with
    nothing overridden, so it stays correct if those defaults ever change.
    """

    strategy = "production"
    name = "production-as-shipped"
    label = "chunk_document defaults, heading prefix on"

    def chunk(self, parsed):
        from grain_growth import DEFAULT_MAX_CHARS, DEFAULT_TARGET_CHARS, chunk_document
        blocks = parsed["blocks"]
        if not blocks:
            return []
        chunks, _skipped = chunk_document(blocks, parsed.get("_key", "doc"),
                                          target_chars=DEFAULT_TARGET_CHARS,
                                          max_chars=DEFAULT_MAX_CHARS)
        return [c["text"] for c in chunks if c.get("text", "").strip()]


def build_chunkers():
    return [
        # the two knobs everyone tunes on a fixed-size splitter
        RecursiveChunker(512, 0),
        RecursiveChunker(512, 103),      # 20% — the c41e756 setting
        RecursiveChunker(1024, 0),
        RecursiveChunker(1024, 102),     # 10%
        # unique to semantic: how the breakpoint threshold is chosen
        SemanticChunker("percentile"),
        SemanticChunker("stddev"),
        SemanticChunker("interquartile"),
        # unique to structure-aware: the heading path prefix
        StructureChunker(1500, headings=True),
        StructureChunker(2000, headings=True),
        StructureChunker(1500, headings=False),
        # the shipped configuration, recorded as its own row
        ProductionChunker(),
    ]


# ============================================================ metrics
# The boundary rules live in chunking.py and are imported where used, so the
# evaluation and the chunker cannot disagree about what a clean edge is.


def boundary_overlap(a, b, cap=600):
    """Longest suffix of `a` that is also a prefix of `b`."""
    limit = min(len(a), len(b), cap)
    for k in range(limit, 0, -1):
        if a[-k:] == b[:k]:
            return k
    return 0


def measure(chunks, pages, corpus_bytes, elapsed):
    if not chunks:
        return {"chunks": 0, "pages": pages, "corpus_bytes": corpus_bytes, "seconds": elapsed}

    lengths = sorted(len(c) for c in chunks)
    overlaps = [boundary_overlap(chunks[i], chunks[i + 1]) for i in range(len(chunks) - 1)]
    with_overlap = [o for o in overlaps if o > 0]

    # one definition of a clean boundary, shared with the chunker, so the
    # evaluation cannot drift from what the pipeline considers correct
    from grain_growth import (ends_cleanly, fences_balanced, has_fence,
                          starts_cleanly, strip_fences)

    starts_mid = ends_mid = 0
    p_starts = p_ends = p_total = 0
    fenced = broken = 0

    for c in chunks:
        body = c.strip()
        # the heading prefix is not the chunk's prose; judge the body
        if "\n\n" in body:
            body = body.split("\n\n", 1)[1].strip()
        if body and not starts_cleanly(body):
            starts_mid += 1
        if body and not ends_cleanly(body):
            ends_mid += 1

        # A table that ends in a number was not cut off mid-sentence, so prose
        # rules are applied only to what remains once fenced blocks are taken
        # out. An arm that cannot identify a table has nothing to take out —
        # which is the deficiency itself, not an unfair comparison.
        if has_fence(body):
            fenced += 1
            if not fences_balanced(body):
                broken += 1
        prose = strip_fences(body)
        if prose:
            p_total += 1
            if not starts_cleanly(prose):
                p_starts += 1
            if not ends_cleanly(prose):
                p_ends += 1

    n = len(chunks)
    return {
        "pages": pages,
        "corpus_bytes": corpus_bytes,
        "chunks": n,
        "mean_chars": round(statistics.mean(lengths), 1),
        "median_chars": int(statistics.median(lengths)),
        "min_chars": lengths[0],
        "max_chars": lengths[-1],
        "overlap_zero_pct": (len(overlaps) - len(with_overlap)) / len(overlaps) if overlaps else 1.0,
        "overlap_median_chars": int(statistics.median(with_overlap)) if with_overlap else 0,
        "overlap_max_chars": max(with_overlap) if with_overlap else 0,
        "starts_mid_sentence_pct": starts_mid / n,
        "ends_mid_sentence_pct": ends_mid / n,
        "prose_chunks": p_total,
        "prose_starts_mid_pct": p_starts / p_total if p_total else 0.0,
        "prose_ends_mid_pct": p_ends / p_total if p_total else 0.0,
        "fenced_pct": fenced / n,
        "fence_broken_pct": broken / fenced if fenced else 0.0,
        "seconds": round(elapsed, 1),
    }


# ============================================================ run
def pick_pdfs(pdf_dir, n, seed):
    pdfs = sorted(p for p in Path(pdf_dir).glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs in {pdf_dir}")
    random.Random(seed).shuffle(pdfs)
    return pdfs[:n]


def run(args):
    SCRATCH.mkdir(parents=True, exist_ok=True)
    pdfs = pick_pdfs(args.pdf_dir, args.n, args.seed)
    print(f"Sandbox: {SCRATCH}")
    print(f"Sample: {len(pdfs)} PDFs from {args.pdf_dir} (seed {args.seed})\n")

    parsers = [RawDumpParser(), LegacyParser(), CurrentParser()]
    chunkers = build_chunkers()

    # --- parse every PDF with both parsers, once ---
    corpora = {}
    parse_stats = {}
    for parser in parsers:
        docs, pages, started = [], 0, time.time()
        for i, pdf in enumerate(pdfs, 1):
            try:
                out = parser.parse(pdf)
            except Exception as e:                                # noqa: BLE001
                print(f"  {parser.name}: [{i}/{len(pdfs)}] FAILED {pdf.stem[:44]}: {e}")
                continue
            out["_key"] = pdf.stem
            docs.append(out)
            pages += out["pages"]
            print(f"  {parser.name}: [{i}/{len(pdfs)}] {pdf.stem[:44]:<46} "
                  f"{out['pages']:>3}p {len(out['text']) // 1024:>4} KB", flush=True)
        corpora[parser.name] = docs
        parse_stats[parser.name] = {
            "label": parser.label,
            "documents": len(docs),
            "pages": pages,
            "text_bytes": sum(len(d["text"].encode()) for d in docs),
            "seconds": round(time.time() - started, 1),
        }
        print(f"  {parser.name}: {pages} pages, "
              f"{parse_stats[parser.name]['text_bytes'] / 1024:.0f} KB text, "
              f"{parse_stats[parser.name]['seconds']}s\n")

        # keep the artefacts the user asked for: parsed JSON + chunks
        dump = SCRATCH / "output" / parser.name
        dump.mkdir(parents=True, exist_ok=True)
        for d in docs:
            (dump / f"{d['_key']}.parsed.json").write_text(
                json.dumps(d["parsed"], indent=2, ensure_ascii=False), encoding="utf-8")

    # --- every chunker on every corpus ---
    rows = []
    for parser in parsers:
        docs = corpora[parser.name]
        pages = parse_stats[parser.name]["pages"]
        corpus_bytes = parse_stats[parser.name]["text_bytes"]
        for chunker in chunkers:
            if isinstance(chunker, (StructureChunker, ProductionChunker)) and parser.name != "current":
                # runs, but on untyped single-type blocks; that IS the finding
                pass
            started = time.time()
            all_chunks = []
            for d in docs:
                try:
                    all_chunks.extend(chunker.chunk(d))
                except Exception as e:                            # noqa: BLE001
                    print(f"  {parser.name}/{chunker.name}: {d['_key'][:36]} failed: {e}")
            m = measure(all_chunks, pages, corpus_bytes, time.time() - started)
            m.update({"parser": parser.name, "chunker": chunker.name,
                      "strategy": chunker.strategy, "chunker_label": chunker.label})
            rows.append(m)
            print(f"  {parser.name:<10} {chunker.name:<26} {m['chunks']:>6} chunks  "
                  f"median {m.get('median_chars', 0):>5}", flush=True)

            dump = SCRATCH / "output" / parser.name / "chunks"
            dump.mkdir(parents=True, exist_ok=True)
            (dump / f"{chunker.name}.json").write_text(
                json.dumps({"parser": parser.name, "chunker": chunker.name,
                            "metrics": m, "chunks": all_chunks[:2000]},
                           indent=2, ensure_ascii=False), encoding="utf-8")

    results = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pdf_dir": str(args.pdf_dir), "seed": args.seed,
        "pdfs": [p.name for p in pdfs],
        "parsers": parse_stats, "rows": rows,
        "sandbox": str(SCRATCH),
    }
    RESULTS.write_text(json.dumps(results, indent=2), encoding="utf-8")
    report(results)


# ============================================================ report
def report(r):
    print("\n" + "=" * 100)
    print(f"CHUNKING BENCHMARK — {len(r['pdfs'])} PDFs, seed {r['seed']}")
    print("=" * 100)

    print("\nParsing")
    print(f"  {'parser':<10}{'docs':>6}{'pages':>8}{'text':>11}{'time':>9}   what it is")
    for name, p in r["parsers"].items():
        print(f"  {name:<10}{p['documents']:>6}{p['pages']:>8}"
              f"{p['text_bytes'] / 1048576:>10.2f}M{p['seconds']:>8}s   {p['label']}")

    header = (f"  {'parser':<10}{'chunker':<26}{'chunks':>7}{'mean':>7}{'med':>6}"
              f"{'min':>6}{'max':>7}{'ov=0':>7}{'ov med':>8}{'mid-start':>11}{'mid-end':>9}")
    for strategy in ("recursive", "semantic", "structure", "production"):
        rows = [x for x in r["rows"] if x["strategy"] == strategy and x["chunks"]]
        if not rows:
            continue
        print(f"\n{strategy.upper()}  —  {rows[0]['chunker_label'] if len(rows) == 1 else 'variations below'}")
        print(header)
        for x in rows:
            print(f"  {x['parser']:<10}{x['chunker']:<26}{x['chunks']:>7}"
                  f"{x['mean_chars']:>7.0f}{x['median_chars']:>6}{x['min_chars']:>6}{x['max_chars']:>7}"
                  f"{100 * x['overlap_zero_pct']:>6.0f}%{x['overlap_median_chars']:>8}"
                  f"{100 * x['starts_mid_sentence_pct']:>10.1f}%{100 * x['ends_mid_sentence_pct']:>8.1f}%")

    print("\nProse only — fenced tables and formulas removed before judging")
    print(f"  {'parser':<10}{'chunker':<26}{'prose':>8}{'mid-start':>11}{'mid-end':>9}"
          f"{'fenced':>9}{'broken':>9}")
    for strategy in ("recursive", "semantic", "structure", "production"):
        for x in [q for q in r["rows"] if q["strategy"] == strategy and q["chunks"]]:
            print(f"  {x['parser']:<10}{x['chunker']:<26}{x.get('prose_chunks', 0):>8}"
                  f"{100 * x.get('prose_starts_mid_pct', 0):>10.1f}%"
                  f"{100 * x.get('prose_ends_mid_pct', 0):>8.1f}%"
                  f"{100 * x.get('fenced_pct', 0):>8.1f}%"
                  f"{100 * x.get('fence_broken_pct', 0):>8.1f}%")

    print("\n  ov=0     share of neighbouring chunk pairs sharing no text")
    print("  ov med   median shared characters where they do overlap")
    print("  mid-*    chunk begins / ends mid-sentence (fragmenting a thought)")
    print("  prose    chunks with prose left after fenced blocks are removed")
    print("  fenced   chunks containing a ~~~table / ~~~formula block")
    print("  broken   of those, the share whose fence is left unclosed by the split")
    print(f"\nArtefacts (parsed JSON + chunks): {r['sandbox']}/output/")
    print(f"Full results: {RESULTS}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    ap.add_argument("--n", type=int, default=12, help="how many PDFs to sample")
    ap.add_argument("--seed", type=int, default=20260911)
    ap.add_argument("--report", action="store_true", help="reprint the last run")
    args = ap.parse_args()

    if args.report:
        if not RESULTS.exists():
            sys.exit("No results yet.")
        report(json.loads(RESULTS.read_text(encoding="utf-8")))
        return
    run(args)


if __name__ == "__main__":
    main()
