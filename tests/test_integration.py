"""End-to-end test: PDF -> YOLO layout JSON -> assembled text.

Needs the layout model weights and a sample PDF; skipped otherwise. Runs on
CPU if the GPU is busy. Kept to 2 pages so it finishes quickly.
"""

import json

import pytest

from config import MODEL_CANDIDATES, PDF_DIR

pytestmark = pytest.mark.skipif(
    not any(p.exists() for p in MODEL_CANDIDATES),
    reason="no layout model available",
)


@pytest.fixture(scope="module")
def sample_pdf():
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        pytest.skip("no sample PDFs in data/raw_pdfs")
    return pdfs[0]


def test_parse_and_process(sample_pdf, tmp_path):
    from pdf_parser import get_detector, assign_words_to_regions, words_to_lines
    import fitz

    detector = get_detector()
    layout_pages = detector.detect_pdf(sample_pdf, max_pages=2)
    assert len(layout_pages) == 2

    doc = fitz.open(sample_pdf)
    page_entry = layout_pages[0]
    labels = {r["label"] for r in page_entry["regions"]}
    assert "Text" in labels or "Title" in labels

    words = doc[0].get_text("words")
    regions, _ = assign_words_to_regions(words, page_entry["regions"])
    for r in regions:
        r["lines"] = words_to_lines(r.pop("words"))
    assert any(r["lines"] for r in regions)

    # Assemble just this page's regions into text.
    from txt_processor import LayoutAssembler, order_regions, render_txt

    assembler = LayoutAssembler()
    for r in order_regions(regions, page_entry["width"]):
        assembler.add_region(r, 0)
    blocks = assembler.finish()
    txt = render_txt(blocks)
    assert len(txt) > 200
