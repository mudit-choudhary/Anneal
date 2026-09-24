"""Anneal uses recrystal as its layout detector.

The detector's own 45 tests live with the library. What matters here is the
seam: the dependency is installed at the evaluated major version, Anneal's
model selection still finds *its* weights (the library's download path is
deliberately not used), those weights load, and the helpers Anneal re-exports
from pdf_parser are still there for callers and scripts.
"""

import sys
from pathlib import Path

import pytest

recrystal = pytest.importorskip("recrystal", reason="pip install -r app/requirements.txt")

APP = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP / "parse_manager"))


def test_the_evaluated_major_version_is_installed():
    # A different major could change boxes, and every parse cached under the
    # old one would no longer be comparable.
    assert recrystal.__version__.split(".")[0] == "1"


def test_pdf_parser_still_exposes_the_helpers_it_re_exports():
    import pdf_parser
    for name in ("assign_words_to_regions", "words_to_lines", "split_region_columns",
                 "find_gutters", "extract_pages", "SWALLOW_LABELS", "SPLITTABLE_LABELS"):
        assert hasattr(pdf_parser, name), f"pdf_parser no longer re-exports {name}"


def test_anneal_weights_are_found_not_downloaded(tmp_path, monkeypatch):
    """resolve_model picks Anneal's own ONNX export from MODELS_DIR."""
    import layout_detector
    monkeypatch.setenv("RECRYSTAL_HOME", str(tmp_path))      # a download would land here
    try:
        path = layout_detector.resolve_model()
    except FileNotFoundError as e:
        pytest.skip(f"no layout model in MODELS_DIR: {e}")
    assert path.suffix == ".onnx" and path.exists()
    assert not list(tmp_path.rglob("*.onnx")), "the library downloaded weights instead of using ours"


def test_those_weights_load_and_describe_themselves(tmp_path, monkeypatch):
    import layout_detector
    monkeypatch.setenv("RECRYSTAL_HOME", str(tmp_path))
    try:
        path = layout_detector.resolve_model()
    except FileNotFoundError as e:
        pytest.skip(f"no layout model in MODELS_DIR: {e}")
    det = recrystal.OnnxYolo(path, providers=["CPUExecutionProvider"])   # CPU: deterministic
    assert len(det.names) == 12
    assert {"Text", "Table", "Title", "Authors", "Picture"} <= set(det.names.values())
    assert det.imgsz == 1024
