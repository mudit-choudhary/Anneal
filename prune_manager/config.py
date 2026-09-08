import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.paths import PDF_DIR, PARSED_DIR, PROCESSED_DIR, REGISTRY_URL  # noqa: E402,F401

PRUNE_INTERVAL = 1800   # seconds between sweeps

# --- Raw PDF policy ----------------------------------------------------
# Raw PDFs are the only input the pipeline can be rebuilt from, so by
# default they stay exactly where they were downloaded.
#
# RAW_PDF_ARCHIVE_DIR: None (default) = leave PDFs in data/raw_pdfs.
#                      A path        = move embedded PDFs there instead.
# RAW_PDF_DELETE:      True = delete instead of archive. Irreversible;
#                      re-ingestion then needs re-downloading.
# RAW_PDF_KEEP_STRATEGY / RAW_PDF_KEEP_COUNT: which embedded PDFs stay in
#   raw_pdfs when archiving/deleting is on —
#     "all"    -> none stay (every embedded PDF is archived/deleted)
#     "newest" -> the newest RAW_PDF_KEEP_COUNT (by download time) stay
#     "oldest" -> the oldest RAW_PDF_KEEP_COUNT stay
RAW_PDF_ARCHIVE_DIR = None
RAW_PDF_DELETE = False
RAW_PDF_KEEP_STRATEGY = "all"
RAW_PDF_KEEP_COUNT = 50
