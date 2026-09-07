from pathlib import Path

# Embedding model. bge-base-en-v1.5: 512-token window (~2,600 chars of paper
# text), 768-dim, ~440MB — fits the 4GB GPU alongside YOLO overnight; runs
# on CPU for single queries during the day (EMBED_DEVICE=cpu).
MODEL_NAME = "BAAI/bge-base-en-v1.5"
# bge v1.5 takes an optional instruction on the *query* side only.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Collection name is tied to the model: embedding dimensions differ between
# models, so switching models means a new collection (and re-ingestion).
COLLECTION_NAME = "papers_bge_base_v1"

dir_path = Path(__file__).resolve().parent
EMBEDDING_VECTOR_PATH = dir_path.parent / 'vector_db'

DATA_DIR = dir_path.parent / "data"
PROCESSED_DIR = str(DATA_DIR / "processed")

REGISTRY_URL = 'http://127.0.0.1:4000'

# Structure-aware chunking (see chunking.py). Budgets are in characters of
# chunk body; the heading prefix adds ~100 more. Keep max well under the
# model's window: 2000 chars ~ 400 tokens.
CHUNK_TARGET_CHARS = 1500   # pack whole paragraphs up to this
CHUNK_MAX_CHARS = 2000      # split a single block only beyond this
