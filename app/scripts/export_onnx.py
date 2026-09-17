"""Export the fine-tuned YOLO layout models to ONNX — a BUILD-TIME tool.

    python scripts/export_onnx.py                 # export every model in MODEL_CANDIDATES
    python scripts/export_onnx.py --imgsz 1024    # must match how the model was trained

This is the *only* script that imports `ultralytics`. Running it is private
use, which the AGPL places no restrictions on; the shipped pipeline loads the
resulting .onnx files with onnxruntime (MIT) and never imports ultralytics.
See docs/PENDING_IMPROVEMENTS.md item 4.

Each `<dir>/weights/best.pt` becomes `<dir>/weights/best.onnx`, with the
class names carried across in the ONNX metadata so inference needs nothing
else. Exported with a dynamic batch axis so pages can be batched.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "parse_manager"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights", nargs="*", help="specific .pt files (default: config MODEL_CANDIDATES)")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--opset", type=int, default=19)
    ap.add_argument("--force", action="store_true", help="re-export even if the .onnx already exists")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics is not installed. It is needed only for this build-time export:\n"
                 "  pip install ultralytics    # then you may uninstall it again")

    if args.weights:
        targets = [Path(w) for w in args.weights]
    else:
        from config import MODEL_CANDIDATES
        targets = [p for p in MODEL_CANDIDATES if p.exists()]
    if not targets:
        sys.exit("no .pt weights found")

    for pt in targets:
        onnx_path = pt.with_suffix(".onnx")
        if onnx_path.exists() and not args.force:
            print(f"skip (exists): {onnx_path}")
            continue
        print(f"exporting {pt} …")
        model = YOLO(str(pt))
        out = model.export(format="onnx", imgsz=args.imgsz, opset=args.opset,
                           dynamic=True, simplify=True, nms=False)
        produced = Path(out)
        if produced != onnx_path:
            produced.replace(onnx_path)
        size = onnx_path.stat().st_size / 1e6
        print(f"  -> {onnx_path} ({size:.1f} MB), classes: {list(model.names.values())}")


if __name__ == "__main__":
    main()
