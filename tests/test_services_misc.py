"""Unit tests for downloader helpers, prune policy, web search, and the
embedding service's request models (no network, no models)."""

import importlib.util
import sys
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_ROOT))


def load(path, name):
    """Import a service module by path with its own directory first on
    sys.path (each has a flat `config` module)."""
    saved = sys.modules.pop("config", None)
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
        sys.modules.pop("config", None)
        if saved is not None:
            sys.modules["config"] = saved
    return module


# --------------------------------------------------------------- downloader
@pytest.fixture(scope="module")
def dl():
    return load(APP_ROOT / "download_manager" / "downloader.py", "downloader_mod")


class TestDownloader:
    def test_sanitize_matches_pipeline_key(self, dl):
        assert dl.sanitize_filename("A Plan Reuse: Mechanism (v2)!") == "A_Plan_Reuse_Mechanism_v2"
        assert len(dl.sanitize_filename("x" * 300)) == 100

    def test_arxiv_id_regex(self, dl):
        assert dl.ARXIV_ID.search("https://arxiv.org/abs/2401.01234v2").group(1) == "2401.01234v2"
        assert dl.ARXIV_ID.search("https://arxiv.org/pdf/2401.01234").group(1) == "2401.01234"
        assert dl.ARXIV_ID.search("https://example.org/paper.pdf") is None

    def test_download_url_rejects_non_pdf(self, dl, tmp_path, monkeypatch):
        monkeypatch.setattr(dl, "PDF_DIR", tmp_path)

        class R:
            ok = True
            headers = {"content-disposition": 'attachment; filename="notes.pdf"'}
            def raise_for_status(self): pass
            def iter_content(self, n): yield b"<html>nope</html>"
        monkeypatch.setattr(dl.requests, "get", lambda *a, **k: R())

        class Reg:
            def set_status(self, *a, **k): return True
        with pytest.raises(ValueError):
            dl.download_url(Reg(), "https://example.org/notes.pdf")
        assert not list(tmp_path.glob("*.pdf"))


# --------------------------------------------------------------- prune
@pytest.fixture(scope="module")
def prune():
    return load(APP_ROOT / "prune_manager" / "pruning.py", "pruning_mod")


class TestPrune:
    """Raw-PDF policy now lives in settings (UI-editable), passed in as a dict."""

    def _papers(self, tmp_path, n):
        papers = []
        for i in range(n):
            (tmp_path / f"p{i}.pdf").write_bytes(b"%PDF-")
            papers.append({"filename": f"p{i}", "status": "embedded",
                           "downloaded_at": f"2026-09-0{i + 1} 00:00:00"})
        return papers

    def test_default_policy_leaves_pdfs_alone(self, prune):
        assert prune.raw_pdf_action({"raw_pdf_policy": "keep"}) is None
        # archive without a destination must not silently delete
        assert prune.raw_pdf_action({"raw_pdf_policy": "archive", "archive_dir": ""}) is None

    def test_policies_resolve(self, prune):
        assert prune.raw_pdf_action({"raw_pdf_policy": "delete"}) == "delete"
        assert prune.raw_pdf_action({"raw_pdf_policy": "archive", "archive_dir": "/tmp/arch"}) == "archive"

    def test_keep_newest(self, prune, tmp_path, monkeypatch):
        monkeypatch.setattr(prune, "PDF_DIR", tmp_path)
        targets = prune.select_raw_pdfs(self._papers(tmp_path, 5),
                                        {"keep_strategy": "newest", "keep_count": 2})
        assert sorted(p["filename"] for p in targets) == ["p0", "p1", "p2"]

    def test_keep_oldest(self, prune, tmp_path, monkeypatch):
        monkeypatch.setattr(prune, "PDF_DIR", tmp_path)
        targets = prune.select_raw_pdfs(self._papers(tmp_path, 3),
                                        {"keep_strategy": "oldest", "keep_count": 1})
        assert sorted(p["filename"] for p in targets) == ["p1", "p2"]

    def test_unknown_strategy_keeps_everything(self, prune, tmp_path, monkeypatch):
        monkeypatch.setattr(prune, "PDF_DIR", tmp_path)
        assert prune.select_raw_pdfs(self._papers(tmp_path, 3),
                                     {"keep_strategy": "nonsense", "keep_count": 1}) == []

    def _dirs(self, tmp_path, prune, monkeypatch):
        raw, parsed, processed = (tmp_path / d for d in ("raw", "parsed", "processed"))
        for d in (raw, parsed, processed):
            d.mkdir()
        (raw / "a.pdf").write_bytes(b"%PDF-")
        (parsed / "a.json").write_text("{}")
        (processed / "a.txt").write_text("")
        (processed / "b.json").write_text("{}")     # b is only 'parsed' -> must survive
        monkeypatch.setattr(prune, "PDF_DIR", raw)
        monkeypatch.setattr(prune, "PARSED_DIR", parsed)
        monkeypatch.setattr(prune, "PROCESSED_DIR", processed)
        return raw, parsed, processed

    class Reg:
        def list_papers(self):
            return [{"filename": "a", "status": "embedded", "downloaded_at": "x"},
                    {"filename": "b", "status": "parsed", "downloaded_at": "y"}]

    def test_sweep_archives_to_dir(self, prune, tmp_path, monkeypatch):
        raw, parsed, processed = self._dirs(tmp_path, prune, monkeypatch)
        archive = tmp_path / "archive"
        counts = prune.sweep(self.Reg(), {"raw_pdf_policy": "archive", "archive_dir": str(archive),
                                          "keep_strategy": "all"})
        assert counts == {"parsed": 1, "processed": 1, "raw_archived": 1, "raw_deleted": 0}
        assert (archive / "a.pdf").exists() and not (raw / "a.pdf").exists()
        assert (processed / "b.json").exists()

    def test_sweep_keep_policy_leaves_pdf(self, prune, tmp_path, monkeypatch):
        raw, _, _ = self._dirs(tmp_path, prune, monkeypatch)
        counts = prune.sweep(self.Reg(), {"raw_pdf_policy": "keep"})
        assert counts["raw_archived"] == 0 and counts["raw_deleted"] == 0
        assert (raw / "a.pdf").exists()

    def test_dry_run_changes_nothing(self, prune, tmp_path, monkeypatch):
        raw, parsed, processed = self._dirs(tmp_path, prune, monkeypatch)
        counts = prune.sweep(self.Reg(), {"raw_pdf_policy": "delete", "keep_strategy": "all"},
                             dry_run=True)
        assert counts == {"parsed": 1, "processed": 1, "raw_archived": 0, "raw_deleted": 1}
        assert (raw / "a.pdf").exists() and (parsed / "a.json").exists()


# --------------------------------------------------------------- web search
class TestWebSearch:
    def test_search_web_extracts_and_truncates(self, monkeypatch):
        sys.path.insert(0, str(APP_ROOT / "rag_setup"))
        import websearch
        sys.path.pop(0)

        class FakeDDGS:
            def text(self, q, max_results):
                return [{"title": "T1", "href": "https://a", "body": "snippet"},
                        {"title": "T2", "href": "https://b", "body": "fallback snippet"}]

        class R:
            def __init__(self, ok, html):
                self.ok, self.text = ok, html
                self.headers = {"content-type": "text/html"}
        responses = {"https://a": R(True, "<html><body><p>" + "word " * 1000 + "</p></body></html>"),
                     "https://b": R(False, "")}
        monkeypatch.setitem(sys.modules, "ddgs", type("m", (), {"DDGS": FakeDDGS}))
        monkeypatch.setattr(websearch.requests, "get", lambda url, **k: responses[url])
        pages = websearch.search_web("q", k=2, chars_per_page=200)
        assert [p["title"] for p in pages] == ["T1", "T2"]
        assert len(pages[0]["text"]) <= 202 and pages[0]["text"].endswith("…")
        assert pages[1]["text"] == "fallback snippet"
