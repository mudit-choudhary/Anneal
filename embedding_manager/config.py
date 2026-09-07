from pathlib import Path

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking (used by embeddings.py and previewed by scripts/rag_inspect.py)
CHUNK_SIZE = 512          # characters per chunk
CHUNK_OVERLAP_PCT = 0.2   # fraction of chunk_size overlapped between chunks

dir_path = Path(__file__).resolve().parent
EMBEDDING_VECTOR_PATH = dir_path.parent / 'vector_db'

DATA_DIR = dir_path.parent / "data"
PROCESSED_DIR = str(DATA_DIR / "processed")

REGISTRY_URL = 'http://127.0.0.1:4000'