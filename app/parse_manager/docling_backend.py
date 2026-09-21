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

# Docling item labels -> our YOLO class names. The mapping and the conversion
# live in the `textreflow` library's Docling adapter; LABEL_MAP is re-exported
# because the published report cites it by this path.
from textreflow.adapters.docling import LABEL_MAP, from_docling  # noqa: F401

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
    pages = from_docling(result.document, max_pages=max_pages)["pages"]
    for page in pages:
        for region in page["regions"]:
            region["backend"] = "docling"
    return pages


def export_markdown(pdf_path) -> str:
    """Docling's own end-to-end output (its reading order + tables), for
    side-by-side comparison in scripts/docling_compare.py."""
    return get_converter().convert(str(pdf_path)).document.export_to_markdown()
