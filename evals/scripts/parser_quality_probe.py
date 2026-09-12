"""Does Docling, on its own, do the noise removal and paragraph merging we built?

    python evals/scripts/parser_quality_probe.py [--papers N]

The parser x chunker matrix cannot answer this. Its `oss_docling` arm feeds
Docling's layout into *our* stage-2 assembler, which is precisely the component
that drops page furniture and merges paragraph continuations. That arm
therefore borrows our work, and its good scores are partly ours.

This probe separates the three:

    docling_native      Docling's own document model, nothing of ours
    docling_plus_ours   Docling layout -> our assembler   (the matrix's arm)
    ours                YOLO layout    -> our assembler   (production)

Writes evals/Reports/parser_quality.json.
"""

import argparse
import collections
import json
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "parse_manager"))

CORPUS = REPO / "evals" / "corpus"
OUT = REPO / "evals" / "Reports" / "parser_quality.json"

_LOWER = re.compile(r"^[a-z]")
MIN_PARAGRAPH = 40          # ignore fragments too short to judge


def norm(s):
    return re.sub(r"\s+", " ", s or "").strip()


def score(paragraphs):
    """A paragraph beginning lower case is a continuation nobody merged."""
    kept = [t for t in paragraphs if len(t) > MIN_PARAGRAPH]
    bad = sum(1 for t in kept if _LOWER.match(t))
    return {"paragraphs": len(kept), "start_lowercase": bad,
            "start_lowercase_pct": bad / len(kept) if kept else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", type=int, default=5)
    args = ap.parse_args()

    import docling_backend as db
    import layout_detector
    import pdf_parser
    import txt_processor

    manifest = json.loads((CORPUS / "manifest.json").read_text())
    pdfs = [CORPUS / f"{p['arxiv_id']}.pdf" for p in manifest["papers"]][:args.papers]
    det = layout_detector.LayoutDetector()

    buckets = {k: [] for k in ("docling_native", "docling_plus_ours", "ours")}
    labels = {"docling": collections.Counter(), "yolo": collections.Counter()}
    furniture = {"docling_native_items": 0, "docling_native_in_markdown": 0}

    def via_assembler(layout_pages, stem, backend):
        with tempfile.TemporaryDirectory() as td:
            lj = Path(td) / f"{stem}.json"
            lj.write_text(json.dumps({"source_pdf": stem, "backend": backend,
                                      "num_pages": len(layout_pages),
                                      "pages": layout_pages}), encoding="utf-8")
            txt_processor.process_layout_json(lj, output_txt=Path(td) / "o.txt",
                                              output_json=Path(td) / "o.json",
                                              update_registry=False)
            return json.loads((Path(td) / "o.json").read_text(encoding="utf-8"))

    for pdf in pdfs:
        doc = db.get_converter().convert(str(pdf)).document
        native, page_furniture = [], []
        for item, _lvl in doc.iterate_items():
            lab = str(getattr(item, "label", "")).split(".")[-1].lower()
            labels["docling"][lab] += 1
            text = (getattr(item, "text", "") or "").strip()
            if lab in ("text", "paragraph"):
                native.append(text)
            if lab in ("page_header", "page_footer", "footnote") and text:
                page_furniture.append(norm(text))
        buckets["docling_native"].append(score(native))
        md = norm(doc.export_to_markdown())
        furniture["docling_native_items"] += len(page_furniture)
        furniture["docling_native_in_markdown"] += sum(1 for t in page_furniture if t in md)

        d = via_assembler(db.detect_pdf(str(pdf)), pdf.stem, "docling")
        buckets["docling_plus_ours"].append(
            score([b["text"] for b in d["blocks"] if b["type"] == "paragraph"]))

        with tempfile.TemporaryDirectory() as td:
            pdf_parser.parse(str(pdf), detector=det, out_dir=td)
            lj = Path(td) / f"{pdf.stem}.json"
            layout = json.loads(lj.read_text())
            for p in layout["pages"]:
                for r in p["regions"]:
                    labels["yolo"][r["label"]] += 1
            txt_processor.process_layout_json(lj, output_txt=Path(td) / "o.txt",
                                              output_json=Path(td) / "o.json",
                                              update_registry=False)
            o = json.loads((Path(td) / "o.json").read_text(encoding="utf-8"))
        buckets["ours"].append(
            score([b["text"] for b in o["blocks"] if b["type"] == "paragraph"]))
        print(f"  {pdf.stem} done", flush=True)

    summary = {}
    for k, rows in buckets.items():
        tot = sum(r["paragraphs"] for r in rows)
        bad = sum(r["start_lowercase"] for r in rows)
        summary[k] = {"paragraphs": tot, "start_lowercase": bad,
                      "start_lowercase_pct": bad / tot if tot else None}

    result = {
        "papers": [p.stem for p in pdfs],
        "metric": "share of paragraph blocks over 40 characters that begin lower case, "
                  "i.e. a continuation the parser did not merge",
        "pipelines": summary,
        "docling_labels": dict(labels["docling"].most_common()),
        "yolo_labels": dict(labels["yolo"].most_common()),
        "page_furniture": furniture,
    }
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\n{'pipeline':<22}{'paragraphs':>12}{'start lowercase':>18}")
    for k, v in summary.items():
        print(f"{k:<22}{v['paragraphs']:>12}{v['start_lowercase']:>10} "
              f"({100 * (v['start_lowercase_pct'] or 0):.1f}%)")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
