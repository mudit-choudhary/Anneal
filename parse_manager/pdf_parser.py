"""Stage 1 of parsing: PDF -> layout JSON.

Combines YOLO layout detections with PyMuPDF word extraction. Every word is
assigned to the detected region containing its center (smallest region wins,
so a Caption sitting on top of a Picture claims its own words). Words covered
by no detection are grouped into fallback Text regions so no content is lost —
unless they sit inside a Picture/Page-header/Page-footer box, in which case
they are recorded but excluded from the text flow.

Output: data/parsed/<name>.json
"""

import json
from pathlib import Path

import fitz
import requests

from config import PARSED_DIR, REGISTRY_URL, SWALLOW_LABELS
from layout_detector import LayoutDetector

_detector = None


def get_detector():
    global _detector
    if _detector is None:
        _detector = LayoutDetector()
    return _detector


def _center_in(bbox, x, y):
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def _area(bbox):
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def assign_words_to_regions(words, regions):
    """Attach each PyMuPDF word to the smallest region containing its center.

    `words` are PyMuPDF tuples (x0, y0, x1, y1, text, block_no, line_no, word_no).
    Returns (regions, swallowed) where each region gains a "words" list and
    fallback Text regions are appended for uncovered words; `swallowed` holds
    words dropped because they sit inside Picture/header/footer boxes.
    """
    for region in regions:
        region["words"] = []

    # Sort candidate regions by area so the first containing hit is the smallest.
    by_area = sorted(regions, key=lambda r: _area(r["bbox"]))
    leftovers = {}  # PyMuPDF block_no -> words
    swallowed = []

    for w in words:
        x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4]
        if not text.strip():
            continue
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

        hit = next((r for r in by_area if _center_in(r["bbox"], cx, cy)), None)
        if hit is not None:
            if hit["label"] in SWALLOW_LABELS:
                swallowed.append({"text": text, "label": hit["label"],
                                  "bbox": [x0, y0, x1, y1]})
            else:
                hit["words"].append([x0, y0, x1, y1, text])
        else:
            leftovers.setdefault(w[5], []).append([x0, y0, x1, y1, text])

    for block_words in leftovers.values():
        xs0 = min(w[0] for w in block_words)
        ys0 = min(w[1] for w in block_words)
        xs1 = max(w[2] for w in block_words)
        ys1 = max(w[3] for w in block_words)
        regions.append({
            "label": "Text",
            "conf": 0.0,
            "fallback": True,
            "bbox": [xs0, ys0, xs1, ys1],
            "words": block_words,
        })

    return regions, swallowed


def words_to_lines(region_words):
    """Group a region's words into reading-order lines of text."""
    if not region_words:
        return []

    heights = sorted(w[3] - w[1] for w in region_words)
    line_tol = max(2.0, heights[len(heights) // 2] * 0.6)

    lines = []  # each: {"y": center, "words": [...]}
    for w in sorted(region_words, key=lambda w: ((w[1] + w[3]) / 2, w[0])):
        cy = (w[1] + w[3]) / 2
        if lines and abs(cy - lines[-1]["y"]) <= line_tol:
            lines[-1]["words"].append(w)
            n = len(lines[-1]["words"])
            lines[-1]["y"] += (cy - lines[-1]["y"]) / n
        else:
            lines.append({"y": cy, "words": [w]})

    return [" ".join(w[4] for w in sorted(l["words"], key=lambda w: w[0]))
            for l in lines]


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


def parse(pdf_path, detector=None, max_pages=None):
    """Parse one PDF into a layout JSON in PARSED_DIR. Returns the JSON path."""
    pdf_path = Path(pdf_path)
    detector = detector or get_detector()

    layout_pages = detector.detect_pdf(pdf_path, max_pages=max_pages)

    doc = fitz.open(pdf_path)
    for page_entry in layout_pages:
        page = doc[page_entry["page"]]
        words = page.get_text("words")
        regions, swallowed = assign_words_to_regions(words, page_entry["regions"])

        for region in regions:
            region["lines"] = words_to_lines(region.pop("words"))
        # Drop empty non-visual detections (visual ones are kept as anchors).
        page_entry["regions"] = [
            r for r in regions if r["lines"] or r["label"] in ("Picture", "Table", "Formula")
        ]
        page_entry["swallowed_text"] = swallowed
    doc.close()

    parsed = {
        "source_pdf": str(pdf_path),
        "num_pages": len(layout_pages),
        "pages": layout_pages,
    }

    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    json_path = PARSED_DIR / f"{pdf_path.stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2)

    _update_registry(pdf_path.stem)
    print(f"[parser] {pdf_path.name}: {len(layout_pages)} pages -> {json_path.name}")
    return json_path


def _update_registry(filename):
    try:
        response = requests.post(
            f"{REGISTRY_URL}/update_status",
            json={"filename": filename, "status": "parsed"},
            timeout=20,
        )
        if not response.json().get("success"):
            print(f"[parser] registry rejected status update for {filename}")
    except requests.RequestException as e:
        print(f"[parser] registry unreachable ({e}); status not updated")


if __name__ == "__main__":
    import sys
    from config import PDF_DIR

    targets = sys.argv[1:] or [str(p) for p in sorted(PDF_DIR.glob("*.pdf"))[:1]]
    for target in targets:
        parse(resolve_pdf(target))
