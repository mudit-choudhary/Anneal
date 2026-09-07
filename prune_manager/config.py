from pathlib import Path

DATA_DIR = str(Path(__file__).resolve().parent.parent / "data")
PDF_DIR = DATA_DIR + "/raw_pdfs"
PARSED_DIR = DATA_DIR + "/parsed"
PROCESSED_DIR = DATA_DIR + "/processed"

REGISTRY_URL = 'http://127.0.0.1:4000'
