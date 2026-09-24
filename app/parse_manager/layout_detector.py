"""Document-layout detection over PDF pages.

The detector is the published `recrystal` library (AGPL-3.0), extracted from
this module and onnx_detector.py. What stays is Anneal's model selection:
the first existing entry of MODEL_CANDIDATES, not the library's download.
"""

import sys
from pathlib import Path

from recrystal import LayoutDetector as _LayoutDetector

from config import MODEL_CANDIDATES

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.logsetup import get_logger  # noqa: E402

get_logger("layout")   # configures logging, so recrystal's GPU/CPU warnings are shown


def resolve_model(model_path=None):
    """First existing entry of MODEL_CANDIDATES, preferring its .onnx form."""
    if model_path is not None:
        return model_path
    for candidate in MODEL_CANDIDATES:
        onnx = candidate.with_suffix(".onnx")
        if onnx.exists():
            return onnx
        if candidate.exists():
            raise FileNotFoundError(
                f"{candidate} has no ONNX export. Build it once with:\n"
                f"  python scripts/export_onnx.py\n"
                f"(that script is the only place ultralytics is used)")
    raise FileNotFoundError(
        f"No layout model found; tried: {[str(p.with_suffix('.onnx')) for p in MODEL_CANDIDATES]}")


class LayoutDetector(_LayoutDetector):
    def __init__(self, model_path=None, providers=None):
        path = resolve_model(model_path)
        super().__init__(path, providers=providers)
        if path != MODEL_CANDIDATES[0].with_suffix(".onnx"):
            print(f"[layout] preferred model missing, using {path}")
