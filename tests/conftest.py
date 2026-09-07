import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# parse_manager modules use flat intra-package imports (from config import ...)
sys.path.insert(0, str(REPO_ROOT / "parse_manager"))
