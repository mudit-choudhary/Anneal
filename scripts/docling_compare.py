"""Benchmark the YOLO parser against Docling on the same PDFs.

    python scripts/docling_compare.py Paper_A.pdf Paper_B.pdf [--pages 5] [--docling-native]

For each PDF and backend: wall time, pages/s, peak GPU memory, number of
blocks/tables in the assembled output, and the first table's text — so you
can judge table structure by eye. Outputs go to data/debug/docling_compare/
(nothing in data/parsed or data/processed is touched, no registry calls).

Both backends are warmed on the first PDF before timing, so model-load time
is excluded; the YOLO arm is the *complete* pipeline (PyMuPDF rendering and
word extraction included), so the two are comparable end to end. Docling's
OCR pass is off by default (born-digital PDFs don't need it) — set
DOCLING_OCR=1 for scanned documents.

`--docling-native` additionally saves Docling's own end-to-end Markdown
(its reading order + tables) next to the pipeline outputs for comparison.
Docling downloads its models (~500 MB) on the first run.
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "parse_manager"))

from common.paths import DEBUG_DIR, PDF_DIR  # noqa: E402

OUT = DEBUG_DIR / "docling_compare"


def peak_vram_mb():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e6
    except Exception:
        pass
    return None


def reset_vram():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def run(pdf: Path, backend: str, pages):
    from pdf_parser import parse
    from txt_processor import process_layout_json

    out_dir = OUT / backend
    out_dir.mkdir(parents=True, exist_ok=True)
    reset_vram()
    t0 = time.perf_counter()
    layout_json = parse(pdf, max_pages=pages, out_dir=out_dir, backend=backend)
    t1 = time.perf_counter()
    txt = process_layout_json(layout_json, output_txt=out_dir / f"{pdf.stem}.txt",
                              output_json=out_dir / f"{pdf.stem}.processed.json", update_registry=False)
    t2 = time.perf_counter()
    blocks = json.loads((out_dir / f"{pdf.stem}.processed.json").read_text())["blocks"]
    n_pages = json.loads(Path(layout_json).read_text())["num_pages"]
    tables = [b for b in blocks if b["type"] == "table"]
    return {"backend": backend, "pages": n_pages, "stage1_s": t1 - t0, "stage2_s": t2 - t1,
            "pages_per_s": n_pages / (t2 - t0) if t2 > t0 else 0, "peak_vram_mb": peak_vram_mb(),
            "blocks": len(blocks), "tables": len(tables), "captions": sum(b["type"] == "caption" for b in blocks),
            "first_table": tables[0]["text"][:600] if tables else "(none)", "txt": str(txt)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdfs", nargs="+")
    ap.add_argument("--pages", type=int, default=None, help="limit pages per PDF")
    ap.add_argument("--docling-native", action="store_true", help="also save Docling's own Markdown export")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    # Warm both backends so model-load time doesn't land in the first result.
    first = Path(args.pdfs[0]) if Path(args.pdfs[0]).exists() else PDF_DIR / Path(args.pdfs[0]).name
    if first.exists():
        print("warming both backends (model load excluded from results)…")
        for backend in ("yolo", "docling"):
            try:
                run(first, backend, 2)
            except Exception as e:
                print(f"  {backend} warmup failed: {e}")

    report = []
    for name in args.pdfs:
        pdf = Path(name) if Path(name).exists() else PDF_DIR / Path(name).name
        if not pdf.exists():
            print(f"skip: {name} not found")
            continue
        print(f"\n=== {pdf.name}")
        for backend in ("yolo", "docling"):
            try:
                r = run(pdf, backend, args.pages)
            except Exception as e:
                print(f"  {backend:8s} FAILED: {e}")
                report.append({"pdf": pdf.name, "backend": backend, "error": str(e)})
                continue
            report.append({"pdf": pdf.name, **r})
            vram = f"{r['peak_vram_mb']:.0f} MB" if r["peak_vram_mb"] is not None else "n/a"
            print(f"  {backend:8s} {r['pages']:3d} pages  stage1 {r['stage1_s']:5.1f}s  stage2 {r['stage2_s']:4.1f}s  "
                  f"{r['pages_per_s']:4.2f} pages/s  peak VRAM {vram}  blocks {r['blocks']}  tables {r['tables']}  captions {r['captions']}")
            print("           first table:\n" + "\n".join("             " + ln for ln in r["first_table"].splitlines()[:8]))
        if args.docling_native:
            try:
                from docling_backend import export_markdown
                (OUT / "docling_native").mkdir(exist_ok=True)
                (OUT / "docling_native" / f"{pdf.stem}.md").write_text(export_markdown(pdf), encoding="utf-8")
                print(f"  docling native markdown -> {OUT / 'docling_native' / (pdf.stem + '.md')}")
            except Exception as e:
                print(f"  docling native export FAILED: {e}")

    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport: {OUT / 'report.json'}   outputs: {OUT}/{{yolo,docling}}/<paper>.txt")


if __name__ == "__main__":
    main()
