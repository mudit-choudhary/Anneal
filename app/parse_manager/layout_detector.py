"""Document-layout detection over PDF pages.

Renders each page with PyMuPDF and runs the fine-tuned YOLOv11 layout model
through **onnxruntime** — ultralytics is not imported at runtime (see
scripts/export_onnx.py and docs/PENDING_IMPROVEMENTS.md item 4). Detections
come back mapped into PDF coordinate space (points, origin top-left) so they
can be intersected with PyMuPDF text extraction.

Model classes (DocLayNet + fine-tuned "Authors"):
    Caption, Footnote, Formula, List-item, Page-footer, Page-header,
    Picture, Section-header, Table, Text, Title, Authors
"""

import fitz
import numpy as np

from config import (
    MODEL_CANDIDATES,
    RENDER_DPI,
    YOLO_BATCH,
    YOLO_CONF,
    YOLO_IMGSZ,
    YOLO_IOU,
)
from onnx_detector import OnnxYolo


def resolve_model(model_path=None):
    """First existing entry of MODEL_CANDIDATES, preferring its .onnx form."""
    if model_path is not None:
        return model_path
    for candidate in MODEL_CANDIDATES:
        onnx = candidate.with_suffix(".onnx")
        if onnx.exists():
            return onnx
        if candidate.exists():
            raise FileNotFoundError(
                f"{candidate} has no ONNX export. Build it once with:\n"
                f"  python scripts/export_onnx.py\n"
                f"(that script is the only place ultralytics is used)")
    raise FileNotFoundError(
        f"No layout model found; tried: {[str(p.with_suffix('.onnx')) for p in MODEL_CANDIDATES]}")


class LayoutDetector:
    def __init__(self, model_path=None, providers=None):
        path = resolve_model(model_path)
        self.model = OnnxYolo(path, providers=providers)
        self.model_path = path
        self.class_names = self.model.names
        if path != MODEL_CANDIDATES[0].with_suffix(".onnx"):
            print(f"[layout] preferred model missing, using {path}")

    @property
    def provider(self):
        return self.model.provider

    @staticmethod
    def _render_page(page):
        """Rasterize a page to an RGB numpy array at RENDER_DPI."""
        pix = page.get_pixmap(dpi=RENDER_DPI, colorspace=fitz.csRGB, alpha=False)
        return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

    def _predict(self, images):
        """Run inference, dropping to CPU if the GPU is full (the 4GB card may
        be shared with the embedder or a warm LLM)."""
        try:
            return self.model.predict(images, conf=YOLO_CONF, iou=YOLO_IOU, imgsz=YOLO_IMGSZ,
                                      fixed_batch=YOLO_BATCH)
        except Exception as e:
            # `get_providers()` always lists CPU as onnxruntime's own fallback,
            # so ask which provider is actually *first* — otherwise this never
            # falls back and a busy GPU turns into a hard failure.
            if self.model.provider == "CPUExecutionProvider":
                raise
            if not any(s in str(e).lower() for s in ("memory", "cuda", "cudnn", "cublas", "allocate")):
                raise
            print(f"[layout] GPU inference failed ({type(e).__name__}); falling back to CPU")
            self.model = OnnxYolo(self.model_path, providers=["CPUExecutionProvider"])
            return self.model.predict(images, conf=YOLO_CONF, iou=YOLO_IOU, imgsz=YOLO_IMGSZ,
                                      fixed_batch=YOLO_BATCH)

    def detect_pdf(self, pdf_path, max_pages=None):
        """Run layout detection on every page of a PDF.

        Returns a list (one entry per page) of dicts:
            {"page": int, "width": float, "height": float,
             "regions": [{"label", "conf", "bbox": [x0, y0, x1, y1]}]}
        with bbox in PDF points.
        """
        doc = fitz.open(pdf_path)
        scale = 72.0 / RENDER_DPI  # rendered pixels -> PDF points
        pages_out = []
        n_pages = len(doc) if max_pages is None else min(max_pages, len(doc))

        for start in range(0, n_pages, YOLO_BATCH):
            batch_pages = [doc[i] for i in range(start, min(start + YOLO_BATCH, n_pages))]
            images = [self._render_page(p) for p in batch_pages]
            results = self._predict(images)

            for page, detections in zip(batch_pages, results):
                regions = [{
                    "label": d["label"],
                    "conf": d["conf"],
                    "bbox": [round(v * scale, 2) for v in d["bbox"]],
                } for d in detections]
                pages_out.append({
                    "page": page.number,
                    "width": page.rect.width,
                    "height": page.rect.height,
                    "regions": regions,
                })

        doc.close()
        return pages_out
