"""YOLO layout detection through onnxruntime — no ultralytics at runtime.

The fine-tuned models are exported once with `scripts/export_onnx.py` (the
only place ultralytics is used, at build time) and loaded here as plain ONNX
graphs. Class names and the training image size travel inside the file's
metadata, so a `.onnx` is self-describing.

The exported graph takes a letterboxed float32 NCHW batch and returns raw
predictions shaped (batch, 4 + num_classes, anchors) — box centre/size in
input-image pixels, class scores already sigmoid-activated. Decoding,
confidence filtering, per-class NMS and the un-letterboxing back to original
page pixels are done here in numpy.
"""

import ast
import ctypes
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.logsetup import get_logger  # noqa: E402

log = get_logger("layout")

PAD_VALUE = 114  # ultralytics' letterbox grey; the models were trained with it

# Loaded on demand by _preload_cuda_libraries(); kept alive for the process.
_CUDA_PRELOADED = None

# Only the libraries onnxruntime's CUDA provider actually dlopen()s. This is
# an allowlist on purpose: loading *everything* under site-packages/nvidia
# pulls in libnvblas, which installs itself as a drop-in BLAS and then
# hijacks CPU BLAS calls process-wide — that breaks numpy/torch CPU maths in
# any process that also runs the embedder ("cublasXtSgemm failed").
_CUDA_LIB_PREFIXES = ("libcublas.", "libcublasLt.", "libcudart.", "libcufft.",
                      "libcurand.", "libcusparse.", "libcudnn")


def _preload_cuda_libraries():
    """Make the CUDA runtime visible to onnxruntime's CUDA provider.

    onnxruntime-gpu dlopen()s `libcublasLt.so.12`, `libcudnn.so.9` and friends
    by bare name. In a venv where CUDA comes from the `nvidia-*-cu12` wheels
    (torch's dependencies) those sit in `site-packages/nvidia/*/lib`, which is
    not on the loader path — so the provider silently falls back to CPU.
    Loading them here with RTLD_GLOBAL resolves the names.

    Returns True if anything was loaded. Safe to call when there is no GPU.
    """
    global _CUDA_PRELOADED
    if _CUDA_PRELOADED is not None:
        return _CUDA_PRELOADED

    lib_dirs = []
    for entry in sys.path:
        nvidia = Path(entry) / "nvidia"
        if nvidia.is_dir():
            lib_dirs += sorted(nvidia.glob("*/lib"))
    handles = []
    remaining = [so for d in lib_dirs for so in sorted(d.glob("lib*.so*"))
                 if so.name.startswith(_CUDA_LIB_PREFIXES)]
    # Two passes: some libraries depend on others that load later.
    for _ in range(2):
        deferred = []
        for so in remaining:
            try:
                handles.append(ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL))
            except OSError:
                deferred.append(so)
        if not deferred:
            break
        remaining = deferred
    _CUDA_PRELOADED = handles or False
    return bool(handles)


def _parse_names(raw):
    """Class names from ONNX metadata: "{0: 'Caption', 1: 'Footnote', …}"."""
    try:
        names = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        names = json.loads(raw)
    return {int(k): v for k, v in names.items()}


def letterbox(image, size, auto=True, stride=32):
    """Resize `image` (H, W, 3 uint8) to fit `size`, keeping aspect ratio and
    centring it on grey padding.

    Reproduces ultralytics' `LetterBox` exactly, which matters more than it
    looks. Two details each shift results measurably:

    - **cv2's INTER_LINEAR**, which unlike Pillow's BILINEAR does *not*
      antialias when downscaling. Swapping resamplers moved confidences by up
      to 0.35 — enough to push a real region under the detection threshold.
    - **`auto` padding**: ultralytics pads only to the next multiple of the
      model stride, not to a full square, whenever a batch is uniformly
      shaped. A 1650x1275 page therefore runs at 1024x800, not 1024x1024, and
      the model sees different context.

    Returns (padded, gain, left, top) where left/top are the *integer* borders
    actually added. The inverse transform must subtract exactly these — using
    the float half-difference instead leaves small regions (page footers,
    section headers) off by half a pixel.
    """
    import cv2

    h, w = image.shape[:2]
    gain = min(size / h, size / w)
    new_w, new_h = int(round(w * gain)), int(round(h * gain))
    pad_w, pad_h = size - new_w, size - new_h
    if auto:                                     # pad to a stride multiple only
        pad_w, pad_h = pad_w % stride, pad_h % stride
    pad_x, pad_y = pad_w / 2, pad_h / 2

    if (w, h) != (new_w, new_h):
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(pad_y - 0.1)), int(round(pad_y + 0.1))
    left, right = int(round(pad_x - 0.1)), int(round(pad_x + 0.1))
    padded = cv2.copyMakeBorder(image, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3)
    return padded, gain, left, top


def nms(boxes, scores, iou_threshold):
    """Greedy non-maximum suppression. `boxes` are (N, 4) xyxy."""
    if len(boxes) == 0:
        return []
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0, x1 - x0) * np.maximum(0, y1 - y0)
    order = scores.argsort()[::-1]

    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ix0 = np.maximum(x0[i], x0[rest])
        iy0 = np.maximum(y0[i], y0[rest])
        ix1 = np.minimum(x1[i], x1[rest])
        iy1 = np.minimum(y1[i], y1[rest])
        inter = np.maximum(0, ix1 - ix0) * np.maximum(0, iy1 - iy0)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_threshold]
    return keep


class OnnxYolo:
    """Minimal YOLO11 detector: preprocess → onnxruntime → decode → NMS."""

    def __init__(self, model_path, providers=None):
        self.model_path = Path(model_path)
        available = ort.get_available_providers()
        if providers is None:
            # CUDA when the GPU build is installed, else CPU. TensorRT is
            # skipped: it rebuilds an engine on first use, which would stall
            # the first paper of a run.
            providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        wanted_cuda = "CUDAExecutionProvider" in providers
        if wanted_cuda:
            _preload_cuda_libraries()
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3          # errors only; the CUDA provider is chatty
        # HEURISTIC picks convolution algorithms without the exhaustive search
        # the default runs on each new input shape.
        session_providers = [
            (p, {"cudnn_conv_algo_search": "HEURISTIC"}) if p == "CUDAExecutionProvider" else p
            for p in providers
        ]
        try:
            self.session = ort.InferenceSession(str(model_path), options, providers=session_providers)
        except Exception as e:
            # Creating the CUDA session fails outright when the GPU is full —
            # e.g. the LLM is loaded during the day. Parsing on CPU is slow
            # but correct, and far better than the service refusing to start.
            if not wanted_cuda:
                raise
            log.warning("could not create a GPU session (%s: %s); falling back to CPU",
                        type(e).__name__, str(e).splitlines()[0][:120])
            self.session = ort.InferenceSession(str(model_path), options,
                                                providers=["CPUExecutionProvider"])
            wanted_cuda = False
        self.provider = self.session.get_providers()[0]

        # onnxruntime falls back to CPU *silently* when the CUDA provider
        # cannot load its libraries — correct results, roughly 6x slower, no
        # error. Say so loudly rather than let an overnight run crawl.
        if wanted_cuda and self.provider != "CUDAExecutionProvider":
            log.warning(
                "layout model is running on %s, not the GPU — expect ~2 pages/s instead of ~13. "
                "The CUDA provider is installed but could not load CUDA 12.x / cuDNN 9.x. "
                "Check: python -c \"import onnxruntime as o; print(o.get_available_providers())\"",
                self.provider)
        elif "CUDAExecutionProvider" not in available:
            log.info("layout model on CPU (onnxruntime-gpu not installed); ~2 pages/s")
        else:
            log.info("layout model on %s", self.provider)

        meta = self.session.get_modelmeta().custom_metadata_map
        self.names = _parse_names(meta["names"]) if "names" in meta else {}
        imgsz = ast.literal_eval(meta["imgsz"]) if "imgsz" in meta else [1024, 1024]
        self.imgsz = int(imgsz[0] if isinstance(imgsz, (list, tuple)) else imgsz)
        self.stride = int(float(meta.get("stride", 32)))
        self.input_name = self.session.get_inputs()[0].name

    def _preprocess(self, images, size):
        # `auto` padding only when every page in the batch has the same shape,
        # otherwise the padded results could not be stacked (this is also
        # exactly when ultralytics enables it).
        auto = len({im.shape for im in images}) == 1
        tensors, transforms = [], []
        for img in images:
            padded, gain, left, top = letterbox(img, size, auto=auto, stride=self.stride)
            tensors.append(padded.transpose(2, 0, 1))       # HWC -> CHW
            transforms.append((gain, left, top))
        batch = np.stack(tensors).astype(np.float32) / 255.0
        return np.ascontiguousarray(batch), transforms

    def _decode(self, prediction, transform, shape, conf, iou):
        """One image's raw output (4 + C, anchors) -> list of detections."""
        gain, pad_x, pad_y = transform
        height, width = shape
        prediction = prediction.T                            # (anchors, 4 + C)
        scores_all = prediction[:, 4:]
        class_ids = scores_all.argmax(1)
        scores = scores_all[np.arange(len(scores_all)), class_ids]

        keep = scores >= conf
        if not keep.any():
            return []
        boxes_cxcywh, scores, class_ids = prediction[keep, :4], scores[keep], class_ids[keep]

        # centre/size -> corners, then undo the letterbox
        cx, cy, w, h = boxes_cxcywh.T
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / gain
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / gain
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)

        detections = []
        for cls in np.unique(class_ids):                     # per-class NMS
            idx = np.where(class_ids == cls)[0]
            for k in nms(boxes[idx], scores[idx], iou):
                j = idx[k]
                detections.append({"label": self.names.get(int(cls), str(cls)),
                                   "class_id": int(cls),
                                   "conf": round(float(scores[j]), 4),
                                   "bbox": [round(float(v), 2) for v in boxes[j]]})
        detections.sort(key=lambda d: -d["conf"])
        return detections

    def predict(self, images, conf=0.30, iou=0.45, imgsz=None, fixed_batch=None):
        """Detect on a batch of RGB uint8 arrays. Returns one detection list
        per image, with boxes in each image's own pixel coordinates.

        `fixed_batch` pads a short final batch by repeating its last page (the
        extra results are discarded). Every new input *shape* makes the CUDA
        provider re-tune its convolution algorithms, which costs far more than
        the wasted slots — a 3-page tail batch measured ~4x slower than the
        steady state without this.
        """
        if not images:
            return []
        size = int(imgsz or self.imgsz)
        real = len(images)
        if fixed_batch and 0 < real < fixed_batch:
            images = list(images) + [images[-1]] * (fixed_batch - real)

        batch, transforms = self._preprocess(images, size)
        outputs = self.session.run(None, {self.input_name: batch})[0]
        return [self._decode(outputs[i], transforms[i], images[i].shape[:2], conf, iou)
                for i in range(real)]
