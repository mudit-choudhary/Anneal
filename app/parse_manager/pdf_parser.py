"""Stage 1 of parsing: PDF -> layout JSON.

Combines YOLO layout detections with PyMuPDF word extraction. Every word is
assigned to the detected region containing its center (smallest region wins,
so a Caption sitting on top of a Picture claims its own words). Words covered
by no detection are grouped into fallback Text regions so no content is lost —
unless they sit inside a Picture/Page-header/Page-footer box, in which case
they are recorded but excluded from the text flow.

Output: data/parsed/<name>.json

The YOLO path is the `recrystal` library; this module adds file I/O, the
Docling backend and the registry update around it.
"""

import json
from pathlib import Path

import fitz

# Detection, word assignment, column splitting and line building are the
# published `recrystal` library (AGPL-3.0), extracted from this file; these
# names are re-exported so callers and tests keep importing them from here.
# What stays is Anneal's glue: paths, the Docling arm, the registry, the CLI.
from recrystal import (  # noqa: F401
    SPLITTABLE_LABELS,
    SWALLOW_LABELS,
    assign_words_to_regions,
    extract_pages,
    find_gutters,
    split_region_columns,
    words_to_lines,
)

from config import PARSED_DIR
from layout_detector import LayoutDetector

_detector = None


def get_detector():
    global _detector
    if _detector is None:
        _detector = LayoutDetector()
    return _detector


def resolve_pdf(arg):
    """Resolve a CLI argument to a PDF path: absolute/relative as given, else
    by filename inside PDF_DIR (so `python pdf_parser.py Some_Paper.pdf` works
    from any directory)."""
    from config import PDF_DIR

    p = Path(arg)
    if p.exists():
        return p
    candidate = PDF_DIR / p.name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"'{arg}' not found (also tried {candidate})")


def parse(pdf_path, detector=None, max_pages=None, out_dir=None, backend=None):
    """Parse one PDF into a layout JSON in PARSED_DIR (or out_dir). Returns the JSON path.

    backend: "yolo" (default, config PARSER_BACKEND) or "docling" — the
    Docling backend returns regions with lines already filled, so the word
    assignment below is skipped.
    """
    from config import PARSER_BACKEND

    pdf_path = Path(pdf_path)
    backend = backend or PARSER_BACKEND

    if backend == "docling":
        import docling_backend
        layout_pages = docling_backend.detect_pdf(pdf_path, max_pages=max_pages)
    else:
        layout_pages = extract_pages(pdf_path, detector=detector or get_detector(),
                                     max_pages=max_pages)

    # `num_pages` is what was actually parsed; `pdf_pages` is what the document
    # has. Recording both means a truncated parse is detectable later even
    # after the PDF has been pruned — see common/coverage.py.
    try:
        with fitz.open(pdf_path) as _doc:
            pdf_pages = len(_doc)
    except Exception:                                    # noqa: BLE001
        pdf_pages = None

    parsed = {
        "source_pdf": str(pdf_path),
        "backend": backend,
        "num_pages": len(layout_pages),
        "pdf_pages": pdf_pages,
        "pages": layout_pages,
    }

    out_dir = Path(out_dir) if out_dir else PARSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{pdf_path.stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2)

    if out_dir == PARSED_DIR:
        _update_registry(pdf_path.stem)
    print(f"[parser] {pdf_path.name}: {len(layout_pages)} pages -> {json_path.name} ({backend})")
    return json_path


def _update_registry(filename):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from common.registry_client import RegistryClient, RegistryUnavailable
    try:
        if not RegistryClient().set_status(filename, "parsed"):
            print(f"[parser] registry rejected status update for {filename} (not registered?)")
    except RegistryUnavailable as e:
        print(f"[parser] {e}; status not updated")


if __name__ == "__main__":
    import sys
    from config import PDF_DIR

    targets = sys.argv[1:] or [str(p) for p in sorted(PDF_DIR.glob("*.pdf"))[:1]]
    for target in targets:
        parse(resolve_pdf(target))
