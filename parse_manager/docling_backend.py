"""Docling as an alternative layout detector (PARSER_BACKEND = "docling").

Docling (IBM, MIT) runs its own DocLayNet-trained layout model plus
TableFormer for table structure. This module maps its output onto the same
layout JSON the YOLO detector produces — regions per page with our 12 labels,
boxes in PDF points (top-left origin), and `lines` already filled — so
stage 2 (txt_processor.py) runs unchanged. Tables arrive as Markdown rows
instead of word soup, which is the main thing this trial is about.

Docling is imported lazily: it's heavy and downloads its models on first use.
"""

import os
from pathlib import Path

# Docling item labels -> our YOLO class names
LABEL_MAP = {
    "title": "Title",
    "section_header": "Section-header",
    "paragraph": "Text",
    "text": "Text",
    "list_item": "List-item",
    "caption": "Caption",
    "table": "Table",
    "formula": "Formula",
    "footnote": "Footnote",
    "page_header": "Page-header",
    "page_footer": "Page-footer",
    "picture": "Picture",
    "code": "Text",
    "reference": "List-item",
    "checkbox_selected": "Text",
    "checkbox_unselected": "Text",
    "document_index": "Table",
    "key_value_region": "Text",
    "form": "Text",
}

_converter = None

# arXiv PDFs are born-digital: their text layer is already there, so Docling's
# OCR pass is pure overhead (and runs on CPU unless onnxruntime-gpu is
# installed). Set DOCLING_OCR=1 to re-enable it for scanned documents.
DO_OCR = os.environ.get("DOCLING_OCR", "0") == "1"


def get_converter():
    global _converter
    if _converter is None:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        opts = PdfPipelineOptions()
        opts.do_ocr = DO_OCR
        opts.do_table_structure = True          # TableFormer: the point of using Docling
        opts.table_structure_options.do_cell_matching = True
        _converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    return _converter


def detect_pdf(pdf_path, max_pages=None):
    """Docling -> list of {page, width, height, regions:[{label, conf, bbox, lines}]}.

    Returns the same structure as LayoutDetector.detect_pdf, with `lines`
    included (Docling already grouped the text), so pdf_parser skips its own
    word assignment when this backend is active.
    """
    from docling_core.types.doc import DocItemLabel  # noqa: F401  (ensures docling_core present)

    result = get_converter().convert(str(pdf_path))
    doc = result.document

    pages = {}
    for page_no, page in doc.pages.items():
        if max_pages is not None and page_no > max_pages:
            continue
        pages[page_no] = {"page": page_no - 1, "width": page.size.width, "height": page.size.height,
                          "regions": [], "swallowed_text": []}

    for item, _level in doc.iterate_items():
        label = LABEL_MAP.get(str(getattr(item, "label", "")).split(".")[-1].lower(), "Text")
        if not getattr(item, "prov", None):
            continue
        prov = item.prov[0]
        page = pages.get(prov.page_no)
        if page is None:
            continue
        bb = prov.bbox
        # Docling boxes are bottom-left origin; ours are top-left.
        if str(getattr(bb, "coord_origin", "")).lower().endswith("bottomleft"):
            y0, y1 = page["height"] - bb.t, page["height"] - bb.b
        else:
            y0, y1 = bb.t, bb.b
        bbox = [round(bb.l, 2), round(min(y0, y1), 2), round(bb.r, 2), round(max(y0, y1), 2)]

        if label == "Table":
            try:
                md = item.export_to_markdown(doc)
            except Exception:
                md = getattr(item, "text", "") or ""
            lines = [ln for ln in md.splitlines() if ln.strip()]
        elif label == "Picture":
            lines = []
        else:
            text = getattr(item, "text", "") or ""
            lines = [ln for ln in text.splitlines() if ln.strip()] or ([text] if text.strip() else [])

        page["regions"].append({"label": label, "conf": 1.0, "bbox": bbox, "lines": lines,
                                "backend": "docling"})

    return [pages[k] for k in sorted(pages)]


def export_markdown(pdf_path) -> str:
    """Docling's own end-to-end output (its reading order + tables), for
    side-by-side comparison in scripts/docling_compare.py."""
    return get_converter().convert(str(pdf_path)).document.export_to_markdown()
