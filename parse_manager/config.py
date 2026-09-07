from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = REPO_ROOT / "data"
PDF_DIR = DATA_DIR / "raw_pdfs"
PARSED_DIR = DATA_DIR / "parsed"          # layout JSONs (one per PDF)
PROCESSED_DIR = DATA_DIR / "processed"    # assembled text + structure JSON

REGISTRY_URL = "http://127.0.0.1:4000"

# --- YOLO layout detection ---
# Tried in order; first existing wins. All fine-tuned at imgsz=1024 on
# research-paper layouts except the last (pretrained
# Armaggheddon/yolo11-document-layout, via scripts/download_layout_model.py).
MODEL_CANDIDATES = [
    REPO_ROOT / "models" / "yolo11s_doc_layout_imgsz_1024" / "weights" / "best.pt",
    REPO_ROOT / "models" / "yolo11n_doc_layout_imgsz_1024" / "weights" / "best.pt",
    REPO_ROOT / "models" / "yolo11n_doc_layout.pt",
]

RENDER_DPI = 150          # page raster resolution fed to YOLO
YOLO_IMGSZ = 1024         # must match fine-tuning imgsz
YOLO_CONF = 0.30          # detection confidence threshold
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
