import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import paths  # noqa: E402

# Embedding model. bge-base-en-v1.5: 512-token window (~2,600 chars of paper
# text), 768-dim, ~440MB — fits the 4GB GPU alongside YOLO overnight; runs
# on CPU for single queries during the day (EMBED_DEVICE=cpu).
MODEL_NAME = "BAAI/bge-base-en-v1.5"
# bge v1.5 takes an optional instruction on the *query* side only.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Collection names are tied to the model: embedding dimensions differ between
# models, so switching models means new collections (and re-ingestion).
COLLECTION_NAME = "papers_bge_base_v1"        # paper chunks
CHATS_COLLECTION_NAME = "chats_bge_base_v1"   # saved conversations (kept separate: model output, not evidence)

# VECTOR_DB_PATH lets a test point at a throwaway store instead of the real
# one (scripts/smoke_test.py uses this so it never touches your corpus).
EMBEDDING_VECTOR_PATH = Path(os.environ.get("VECTOR_DB_PATH", paths.VECTOR_DB))

DATA_DIR = paths.DATA_DIR
PROCESSED_DIR = str(paths.PROCESSED_DIR)

REGISTRY_URL = 'http://127.0.0.1:4000'

# Structure-aware chunking (see chunking.py). Budgets are in characters of
# chunk body; the heading prefix adds ~100 more. Keep max well under the
# model's window: 2000 chars ~ 400 tokens.
CHUNK_TARGET_CHARS = 1500   # pack whole paragraphs up to this
CHUNK_MAX_CHARS = 2000      # split a single block only beyond this
