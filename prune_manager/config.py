from pathlib import Path

DATA_DIR = str(Path(__file__).resolve().parent.parent / "data")
PDF_DIR = DATA_DIR + "/raw_pdfs"
PARSED_DIR = DATA_DIR + "/parsed"
PROCESSED_DIR = DATA_DIR + "/processed"

REGISTRY_URL = 'http://127.0.0.1:4000'

PRUNE_INTERVAL = 1800   # seconds between sweeps

# Raw PDFs are the only input the pipeline can be re-run from (every parser,
# chunker or embedding-model change re-ingests from them, and the planned
# VLM pass over figures/tables needs the page images). Keep them unless disk
# is genuinely scarce.
PRUNE_RAW_PDFS = False
