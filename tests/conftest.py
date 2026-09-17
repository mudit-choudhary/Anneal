import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent / "app"
# parse_manager modules use flat intra-package imports (from config import ...)
sys.path.insert(0, str(APP_ROOT / "parse_manager"))
