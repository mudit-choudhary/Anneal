from pathlib import Path

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

dir_path = Path(__file__).resolve().parent
EMBEDDING_VECTOR_PATH = dir_path.parent / 'vector_db'

DATA_DIR = "/home/mudit/Desktop/PaperParsing/data"
PROCESSED_DIR = DATA_DIR + "/processed"

REGISTRY_URL = 'http://127.0.0.1:4000'