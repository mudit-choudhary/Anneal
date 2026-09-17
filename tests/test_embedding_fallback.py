"""The embedder shares a 4GB GPU with the answering model, so every CUDA path
must degrade to the CPU instead of failing a paper.

This bit users for real: with the LLM warm, two papers were marked `error`
because embedding raised out-of-memory, and a later restart could not even
load the model. Both paths are covered here.

`embeddings.py` loads a real model at import, so these tests exercise the
logic against a stub module rather than importing it.
"""

from pathlib import Path

import pytest

# Deliberately *not* on sys.path: each manager has its own flat `config`
# module, and adding one here would shadow another test module's.
APP_ROOT = Path(__file__).resolve().parent.parent / "app"


def _is_oom(exc):
    """Mirror of embeddings._is_oom — kept in step by test_matches_source."""
    text = str(exc).lower()
    return "out of memory" in text or "cuda error" in text or "cublas" in text


class TestOomDetection:
    @pytest.mark.parametrize("message", [
        "CUDA out of memory. Tried to allocate 44.00 MiB",
        "CUDA error: out of memory",
        "cuBLAS error: CUBLAS_STATUS_ALLOC_FAILED",
        "torch.AcceleratorError: CUDA error: out of memory",
    ])
    def test_recognises_real_failures(self, message):
        assert _is_oom(RuntimeError(message))

    @pytest.mark.parametrize("message", [
        "no such collection", "connection refused", "malformed input",
    ])
    def test_ignores_unrelated_errors(self, message):
        assert not _is_oom(RuntimeError(message))

    def test_matches_source(self):
        """Guard against the source drifting away from this copy."""
        source = (APP_ROOT / "embedding_manager" / "embeddings.py").read_text()
        for needle in ('"out of memory" in text', '"cuda error" in text', '"cublas" in text'):
            assert needle in source, f"embeddings._is_oom no longer checks {needle}"


class TestFallbackBehaviour:
    """A stub standing in for the module's device handling."""

    class Fake:
        def __init__(self, fail_on_gpu_add=False, fail_on_gpu_load=False):
            self.device = "cuda"
            self.fail_add, self.fail_load = fail_on_gpu_add, fail_on_gpu_load
            self.builds = []
            if self.fail_load:
                self.device = "cpu"      # what the import-time guard does
            self.builds.append(self.device)

        def add(self):
            if self.device == "cuda" and self.fail_add:
                if not self.fall_back():
                    raise RuntimeError("CUDA out of memory")
                return self.add()
            return f"added on {self.device}"

        def fall_back(self):
            if self.device == "cpu":
                return False
            self.device = "cpu"
            self.builds.append("cpu")
            return True

    def test_add_retries_on_cpu(self):
        f = self.Fake(fail_on_gpu_add=True)
        assert f.add() == "added on cpu"
        assert f.builds == ["cuda", "cpu"]

    def test_load_failure_starts_on_cpu(self):
        f = self.Fake(fail_on_gpu_load=True)
        assert f.device == "cpu"
        assert f.add() == "added on cpu"

    def test_no_second_fallback_once_on_cpu(self):
        f = self.Fake()
        f.device = "cpu"
        assert f.fall_back() is False

    def test_source_wires_both_paths(self):
        """The import-time guard and the per-call retry must both exist."""
        source = (APP_ROOT / "embedding_manager" / "embeddings.py").read_text()
        assert "could not load the embedding model" in source, "no import-time CPU fallback"
        assert "fall_back_to_cpu" in source, "no runtime CPU fallback"
        assert source.count("fall_back_to_cpu(") >= 3, "fallback not wired into add and query"
