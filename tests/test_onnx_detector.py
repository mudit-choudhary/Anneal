"""Tests for the ONNX layout detector's numpy geometry — the parts that
replaced ultralytics and therefore have to be right on their own.

The pure functions (letterbox, NMS, decode) run without a model. The
equivalence test against the real .onnx is skipped when no export exists.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "parse_manager"))

from onnx_detector import OnnxYolo, letterbox, nms  # noqa: E402


class TestLetterbox:
    def test_square_padding_centres_image(self):
        img = np.full((100, 200, 3), 255, dtype=np.uint8)
        padded, gain, left, top = letterbox(img, 400, auto=False)
        assert padded.shape == (400, 400, 3)
        assert gain == 2.0                       # 400/200 limits it
        assert (left, top) == (0, 100)           # 200*2=400 wide, 100*2=200 tall
        assert (padded[0, 0] == 114).all()       # grey pad at the top
        assert (padded[200, 200] == 255).all()   # image in the middle

    def test_auto_pads_only_to_stride_multiple(self):
        # A 150-DPI portrait page: 1650 tall x 1275 wide. gain = 1024/1650, so
        # it becomes 1024 tall x 791 wide, and `auto` pads the width 791 -> 800
        # (the next multiple of the stride) instead of out to a 1024 square.
        img = np.zeros((1650, 1275, 3), dtype=np.uint8)
        padded, gain, left, top = letterbox(img, 1024, auto=True, stride=32)
        assert padded.shape == (1024, 800, 3)
        assert padded.shape[1] % 32 == 0
        assert (left, top) == (4, 0)

    def test_auto_off_gives_full_square(self):
        img = np.zeros((1650, 1275, 3), dtype=np.uint8)
        padded, _, _, _ = letterbox(img, 1024, auto=False)
        assert padded.shape == (1024, 1024, 3)

    def test_returned_offsets_invert_the_transform(self):
        img = np.zeros((1650, 1275, 3), dtype=np.uint8)
        _, gain, left, top = letterbox(img, 1024, auto=True, stride=32)
        # a box at a known page position maps forward, then back exactly
        page_box = np.array([100.0, 200.0, 400.0, 500.0])
        fwd = page_box * gain + np.array([left, top, left, top])
        back = (fwd - np.array([left, top, left, top])) / gain
        assert np.allclose(back, page_box)


class TestNMS:
    def test_suppresses_overlapping_keeps_best(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=float)
        scores = np.array([0.9, 0.8, 0.7])
        keep = nms(boxes, scores, 0.5)
        assert sorted(keep) == [0, 2]

    def test_keeps_both_when_iou_below_threshold(self):
        boxes = np.array([[0, 0, 10, 10], [8, 8, 18, 18]], dtype=float)
        assert sorted(nms(boxes, np.array([0.9, 0.8]), 0.5)) == [0, 1]

    def test_empty_and_single(self):
        assert nms(np.zeros((0, 4)), np.array([]), 0.5) == []
        assert nms(np.array([[0.0, 0, 1, 1]]), np.array([0.5]), 0.5) == [0]

    def test_returns_highest_score_first(self):
        boxes = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], dtype=float)
        assert nms(boxes, np.array([0.2, 0.9]), 0.5)[0] == 1


class TestDecode:
    """_decode turns raw (4 + C, anchors) output into page-space detections."""

    def _detector(self, names):
        det = OnnxYolo.__new__(OnnxYolo)      # no model file needed
        det.names = names
        return det

    def _raw(self, entries, n_classes=3, n_anchors=8):
        raw = np.zeros((4 + n_classes, n_anchors), dtype=np.float32)
        for i, (cx, cy, w, h, cls, score) in enumerate(entries):
            raw[:4, i] = [cx, cy, w, h]
            raw[4 + cls, i] = score
        return raw

    def test_decodes_box_class_and_undoes_letterbox(self):
        det = self._detector({0: "Text", 1: "Table", 2: "Title"})
        # gain 0.5, pad (10, 20): a box at model coords (110, 120, w40, h20)
        raw = self._raw([(110, 120, 40, 20, 1, 0.9)])
        out = det._decode(raw, (0.5, 10, 20), (1000, 1000), conf=0.3, iou=0.5)
        assert len(out) == 1 and out[0]["label"] == "Table" and out[0]["conf"] == 0.9
        # centre 110 -> x0 = (110-20-10)/0.5 = 160, x1 = (110+20-10)/0.5 = 240
        assert out[0]["bbox"] == [160.0, 180.0, 240.0, 220.0]

    def test_confidence_threshold_filters(self):
        det = self._detector({0: "Text"})
        raw = self._raw([(50, 50, 10, 10, 0, 0.25)], n_classes=1)
        assert det._decode(raw, (1.0, 0, 0), (100, 100), conf=0.3, iou=0.5) == []

    def test_boxes_clipped_to_page(self):
        det = self._detector({0: "Text"})
        raw = self._raw([(5, 5, 40, 40, 0, 0.9)], n_classes=1)   # extends past the origin
        box = det._decode(raw, (1.0, 0, 0), (100, 100), conf=0.3, iou=0.5)[0]["bbox"]
        assert box[0] >= 0 and box[1] >= 0 and box[2] <= 100 and box[3] <= 100

    def test_nms_is_per_class(self):
        det = self._detector({0: "Text", 1: "Table"})
        # two heavily overlapping boxes of *different* classes: both survive
        raw = self._raw([(50, 50, 20, 20, 0, 0.9), (51, 51, 20, 20, 1, 0.8)], n_classes=2)
        out = det._decode(raw, (1.0, 0, 0), (200, 200), conf=0.3, iou=0.5)
        assert sorted(d["label"] for d in out) == ["Table", "Text"]
        # same class -> suppressed
        raw = self._raw([(50, 50, 20, 20, 0, 0.9), (51, 51, 20, 20, 0, 0.8)], n_classes=2)
        assert len(det._decode(raw, (1.0, 0, 0), (200, 200), conf=0.3, iou=0.5)) == 1

    def test_results_sorted_by_confidence(self):
        det = self._detector({0: "Text"})
        raw = self._raw([(20, 20, 5, 5, 0, 0.4), (80, 80, 5, 5, 0, 0.95)], n_classes=1)
        out = det._decode(raw, (1.0, 0, 0), (200, 200), conf=0.3, iou=0.5)
        assert [d["conf"] for d in out] == [0.95, 0.4]


class TestProviderReporting:
    """A CUDA provider that cannot load its libraries falls back to CPU
    silently: correct results, ~6x slower, no error. That must be loud."""

    def _build(self, monkeypatch, available, actual):
        import onnx_detector as od

        class FakeSession:
            def __init__(self, *a, **k): pass
            def get_providers(self): return [actual]
            def get_modelmeta(self):
                return type("M", (), {"custom_metadata_map": {
                    "names": "{0: 'Text'}", "imgsz": "[1024, 1024]", "stride": "32"}})()
            def get_inputs(self):
                return [type("I", (), {"name": "images"})()]

        monkeypatch.setattr(od.ort, "get_available_providers", lambda: available)
        monkeypatch.setattr(od.ort, "InferenceSession", FakeSession)
        monkeypatch.setattr(od, "_preload_cuda_libraries", lambda: False)
        return od.OnnxYolo("dummy.onnx")

    def test_warns_when_cuda_available_but_unused(self, monkeypatch, caplog):
        with caplog.at_level("WARNING"):
            det = self._build(monkeypatch, ["CUDAExecutionProvider", "CPUExecutionProvider"],
                              "CPUExecutionProvider")
        assert det.provider == "CPUExecutionProvider"
        assert any("not the GPU" in r.message for r in caplog.records), "silent CPU fallback was not reported"

    def test_no_warning_when_gpu_actually_used(self, monkeypatch, caplog):
        with caplog.at_level("WARNING"):
            self._build(monkeypatch, ["CUDAExecutionProvider", "CPUExecutionProvider"],
                        "CUDAExecutionProvider")
        assert not [r for r in caplog.records if r.levelname == "WARNING"]

    def test_no_warning_on_cpu_only_install(self, monkeypatch, caplog):
        with caplog.at_level("WARNING"):
            self._build(monkeypatch, ["CPUExecutionProvider"], "CPUExecutionProvider")
        assert not [r for r in caplog.records if r.levelname == "WARNING"]


ONNX_MODELS = [p.with_suffix(".onnx") for p in
               [REPO_ROOT / "models" / "yolo11s_doc_layout_imgsz_1024" / "weights" / "best.pt"]]


@pytest.mark.skipif(not any(p.exists() for p in ONNX_MODELS), reason="no ONNX export present")
class TestRealModel:
    def test_metadata_is_self_describing(self):
        det = OnnxYolo(next(p for p in ONNX_MODELS if p.exists()))
        assert len(det.names) == 12
        assert set(det.names.values()) >= {"Text", "Table", "Title", "Authors", "Picture"}
        assert det.imgsz == 1024 and det.stride == 32

    def test_fixed_batch_padding_does_not_change_results(self):
        det = OnnxYolo(next(p for p in ONNX_MODELS if p.exists()))
        rng = np.random.default_rng(0)
        pages = [rng.integers(0, 255, (330, 255, 3), dtype=np.uint8) for _ in range(2)]
        plain = det.predict(pages, conf=0.3, iou=0.7)
        padded = det.predict(pages, conf=0.3, iou=0.7, fixed_batch=4)
        assert len(plain) == len(padded) == 2
        assert [len(p) for p in plain] == [len(p) for p in padded]
