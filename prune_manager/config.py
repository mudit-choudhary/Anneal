import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.paths import PDF_DIR, PARSED_DIR, PROCESSED_DIR, REGISTRY_URL  # noqa: E402,F401

# The raw-PDF policy (keep / archive / delete, and the keep strategy) lives in
# data/settings.json so it can be changed from the UI without a restart — see
# common/settings.py "prune". Only the poll floor is fixed here.
MIN_INTERVAL = 60
