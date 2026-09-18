import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))
from common.paths import (  # noqa: E402
    DATA_DIR, MODELS_DIR, PARSED_DIR, PDF_DIR, PROCESSED_DIR,   # noqa: F401
)

REGISTRY_URL = "http://127.0.0.1:4000"

# --- YOLO layout detection ---
# Tried in order; first existing wins. All fine-tuned at imgsz=1024 on
# research-paper layouts (12 classes incl. Authors).
MODEL_CANDIDATES = [
    MODELS_DIR / "yolo11s_doc_layout_imgsz_1024" / "weights" / "best.pt",       # primary
    MODELS_DIR / "yolo11_doc_layout_v2224_imgsz_1024" / "weights" / "best.pt",  # 1st fallback
    MODELS_DIR / "yolo11n_doc_layout_imgsz_1024" / "weights" / "best.pt",       # last fallback
]

# Parser backend: "yolo" (this pipeline) or "docling" (IBM Docling trial,
# parse_manager/docling_backend.py — same processed-JSON output contract).
PARSER_BACKEND = "yolo"

RENDER_DPI = 150          # page raster resolution fed to the layout model
YOLO_IMGSZ = 1024         # must match fine-tuning imgsz
YOLO_CONF = 0.30          # detection confidence threshold
YOLO_IOU = 0.70           # NMS IoU threshold (ultralytics' predict default,
                          # which is what the corpus was parsed with)
YOLO_BATCH = 4            # pages per inference batch (fits a 4GB GPU)

# --- Reading-order / assembly heuristics ---
# A region wider than this fraction of the page is treated as full-width
# (spans both columns of a two-column paper).
FULL_WIDTH_FRACTION = 0.6
# If at least this fraction of body regions are full-width, the page is
# treated as single-column.
SINGLE_COLUMN_FRACTION = 0.7
# Words whose center falls in no detected region are grouped into fallback
# Text regions unless they sit inside one of these region types.
SWALLOW_LABELS = {"Picture", "Page-header", "Page-footer"}
