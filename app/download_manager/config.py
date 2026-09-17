import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.paths import PDF_DIR, REGISTRY_URL  # noqa: E402,F401

# --- Domains searched on arXiv (searched one after another each cycle) ---
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
CHECK_INTERVAL = 3600         # seconds between cycles when running as a loop
MAX_PAPERS_PER_DOMAIN = 20    # new downloads per domain per cycle; None = unlimited
