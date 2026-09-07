from pathlib import Path

# --- Paths ---
BASE_DIR = str(Path(__file__).resolve().parent.parent / "data")
PDF_DIR = BASE_DIR + "/raw_pdfs"

REGISTRY_URL = 'http://127.0.0.1:4000'

# --- Domains to Download ---
DOMAINS = [
    "Generative AI",
    "Large Language Models",
    "Retrieval Augmented Generation",
    "Machine Learning",
    "Quantum Artificial Intelligence",
    "Neural Networks",
    "Computer Vision",
    "Graph Neural Networks",
    "Deep Neural Networks",
    "RAG",
]

# --- Crawl budget ---
BACKFILL_DAYS = 32            # how far back to search when the registry has no checkpoint for a domain
CHECK_INTERVAL = 3600         # seconds between crawl cycles
MAX_PAPERS_PER_DOMAIN = 20    # new downloads per domain per cycle; None = unlimited
                              # (keeps an overnight ingestion bounded on a single 4GB GPU)
