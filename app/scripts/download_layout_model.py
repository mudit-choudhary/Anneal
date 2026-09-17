"""Download a pretrained YOLOv11 document-layout model from Hugging Face.

Used as the fallback when the fine-tuned model
(models/yolo11n_doc_layout_imgsz_1024/weights/best.pt) is unavailable.

Usage: python download_layout_model.py [n|s|m]
"""

import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
SIZES = {"n": "yolo11n_doc_layout.pt", "s": "yolo11s_doc_layout.pt", "m": "yolo11m_doc_layout.pt"}

if __name__ == "__main__":
    size = sys.argv[1] if len(sys.argv) > 1 else "n"
    MODELS_DIR.mkdir(exist_ok=True)
    path = hf_hub_download(
        repo_id="Armaggheddon/yolo11-document-layout",
        filename=SIZES[size],
        repo_type="model",
        local_dir=MODELS_DIR,
    )
    print(f"Downloaded to {path}")
