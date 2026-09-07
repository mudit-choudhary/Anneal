"""YOLOv11 document-layout detection over PDF pages.

Renders each PDF page with PyMuPDF, runs batched YOLO inference, and returns
detections mapped back into PDF coordinate space (points, origin top-left) so
they can be intersected with PyMuPDF text extraction output.

Model classes (DocLayNet + fine-tuned "Authors"):
    Caption, Footnote, Formula, List-item, Page-footer, Page-header,
    Picture, Section-header, Table, Text, Title, Authors
"""

import fitz
import numpy as np

from config import (
    MODEL_CANDIDATES,
    RENDER_DPI,
    YOLO_IMGSZ,
    YOLO_CONF,
    YOLO_BATCH,
)


class LayoutDetector:
    def __init__(self, model_path=None, device=None):
        from ultralytics import YOLO
        import torch

        if model_path is None:
            model_path = next((p for p in MODEL_CANDIDATES if p.exists()), None)
            if model_path is None:
                raise FileNotFoundError(
                    f"No layout model found; tried: {[str(p) for p in MODEL_CANDIDATES]}"
                )
            if model_path != MODEL_CANDIDATES[0]:
                print(f"[layout] preferred model missing, using {model_path}")

        self.device = device if device is not None else (
            0 if torch.cuda.is_available() else "cpu"
        )
        self.model = YOLO(str(model_path))
        self.class_names = self.model.names

    def _predict(self, images):
        """Run inference, dropping to CPU if the GPU is full (4GB card may be
        shared with a training job)."""
        import torch

        try:
            return self.model.predict(
                images, imgsz=YOLO_IMGSZ, conf=YOLO_CONF,
                device=self.device, verbose=False,
            )
        except RuntimeError as e:
            # Covers torch.OutOfMemoryError and CUBLAS/CUDA allocation errors.
            msg = str(e)
            if self.device == "cpu" or ("CUDA" not in msg and "out of memory" not in msg):
                raise
            print("[layout] CUDA out of memory; falling back to CPU")
            torch.cuda.empty_cache()
            self.device = "cpu"
            return self.model.predict(
                images, imgsz=YOLO_IMGSZ, conf=YOLO_CONF,
                device=self.device, verbose=False,
            )

    @staticmethod
    def _render_page(page):
        """Rasterize a page to an RGB numpy array at RENDER_DPI."""
        pix = page.get_pixmap(dpi=RENDER_DPI, colorspace=fitz.csRGB, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        return img

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
            # Ultralytics expects BGR arrays when given raw numpy images.
            images = [self._render_page(p)[:, :, ::-1] for p in batch_pages]
            results = self._predict(images)

            for page, result in zip(batch_pages, results):
                regions = []
                for box in result.boxes:
                    x0, y0, x1, y1 = (box.xyxy[0] * scale).tolist()
                    regions.append({
                        "label": self.class_names[int(box.cls)],
                        "conf": round(float(box.conf), 4),
                        "bbox": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
                    })
                pages_out.append({
                    "page": page.number,
                    "width": page.rect.width,
                    "height": page.rect.height,
                    "regions": regions,
                })

        doc.close()
        return pages_out
