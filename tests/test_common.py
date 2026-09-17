"""Tests for the shared common/ package: settings store, chat store,
registry client (mocked HTTP), and the registry's SQLite layer."""

import importlib.util
import sys
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_ROOT))

from common import settings as settings_store  # noqa: E402
from common.chatstore import ChatStore  # noqa: E402
from common.registry_client import RegistryClient, RegistryUnavailable  # noqa: E402


# --------------------------------------------------------------- settings
class TestSettings:
    @pytest.fixture(autouse=True)
    def isolate(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings_store, "SETTINGS_FILE", tmp_path / "settings.json")
        for env in ("LLM_BACKEND", "OLLAMA_MODEL", "OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL"):
            monkeypatch.delenv(env, raising=False)

    def test_defaults_when_no_file(self):
        s = settings_store.load()
        assert s["llm"]["backend"] == "local"
        assert s["retrieval"]["n_results"] == 6

    def test_env_overrides_defaults_but_file_wins(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_MODEL", "from-env")
        assert settings_store.load()["llm"]["local"]["model"] == "from-env"
        settings_store.save({"llm": {"local": {"model": "from-file"}}})
        assert settings_store.load()["llm"]["local"]["model"] == "from-file"

    def test_save_merges_deeply(self):
        settings_store.save({"llm": {"openai": {"base_url": "http://gw/v1"}}})
        settings_store.save({"llm": {"openai": {"model": "m"}}})
        oa = settings_store.load()["llm"]["openai"]
        assert oa["base_url"] == "http://gw/v1" and oa["model"] == "m"

    def test_mask_and_unmask(self):
        s = settings_store.save({"llm": {"openai": {"api_key": "sk-abcdef1234"}}})
        masked = settings_store.masked(s)
        assert masked["llm"]["openai"]["api_key"] == "••••1234"
        update = settings_store.unmask({"llm": {"openai": {"api_key": "••••1234", "model": "x"}}}, s)
        assert update["llm"]["openai"]["api_key"] == "sk-abcdef1234"
        update = settings_store.unmask({"llm": {"openai": {"api_key": "sk-new"}}}, s)
        assert update["llm"]["openai"]["api_key"] == "sk-new"


# --------------------------------------------------------------- chat store
class TestChatStore:
    def test_lifecycle_and_pairs(self, tmp_path):
        store = ChatStore(tmp_path / "app.db")
        chat = store.create("My chat")
        store.add_message(chat["id"], "user", "q1")
        store.add_message(chat["id"], "assistant", "a1", sources=[{"n": 1}])
        store.add_message(chat["id"], "user", "q2 (unanswered)")
        assert store.qa_pairs(chat["id"]) == [{"question": "q1", "answer": "a1"}]
        got = store.get(chat["id"])
        assert got["messages"][1]["sources"] == [{"n": 1}]
        assert store.list()[0]["n_messages"] == 3
        store.mark_embedded(chat["id"])
        assert store.get(chat["id"])["embedded"] is True
        store.add_message(chat["id"], "assistant", "a2")   # new content -> needs re-embedding
        assert store.get(chat["id"])["embedded"] is False
        assert store.rename(chat["id"], "Renamed") and store.get(chat["id"])["title"] == "Renamed"
        assert store.delete(chat["id"]) and store.get(chat["id"]) is None


# --------------------------------------------------------------- registry client
class FakeResponse:
    def __init__(self, status=200, body=None):
        self.status_code, self._body, self.text = status, body or {}, ""

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


class TestRegistryClient:
    def test_calls_and_encoding(self, monkeypatch):
        calls = []

        def fake_request(method, url, timeout, **kw):
            calls.append((method, url, kw.get("json"), kw.get("params")))
            return FakeResponse(200, {"status": "parsed", "success": True, "papers": [], "last_checkpoint": None})
        monkeypatch.setattr("common.registry_client.requests.request", fake_request)
        c = RegistryClient("http://reg:4000")
        assert c.get_status("A B/C") == "parsed"
        assert calls[-1][1] == "http://reg:4000/v1/papers/A%20B%2FC"
        assert c.set_status("x", "error", error="boom") is True
        assert calls[-1][0] == "PUT" and calls[-1][2] == {"status": "error", "error": "boom"}
        c.list_papers(status="embedded")
        assert calls[-1][3] == {"status": "embedded"}
        c.checkpoint("Graph Neural Networks")
        assert calls[-1][1].endswith("/v1/domains/Graph%20Neural%20Networks/checkpoint")

    def test_unavailable(self, monkeypatch):
        import requests

        def down(*a, **k):
            raise requests.ConnectionError("refused")
        monkeypatch.setattr("common.registry_client.requests.request", down)
        c = RegistryClient()
        assert c.health() is False
        with pytest.raises(RegistryUnavailable):
            c.get_status("x")


# --------------------------------------------------------------- registry sqlite
@pytest.fixture
def registry(tmp_path):
    spec = importlib.util.spec_from_file_location("registry_sqlite", APP_ROOT / "registry_manager" / "registry.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FileRegistry(str(tmp_path / "r.db"))


class TestFileRegistry:
    def test_status_chain_and_timestamps(self, registry):
        assert registry.update_status("a", "downloaded", domain="d", published_at="2026-09-01T00:00:00")
        assert registry.update_status("a", "downloaded") and registry.get_status("a") == "downloaded"  # idempotent
        for s in ("parsed", "processed", "embedded"):
            assert registry.update_status("a", s)
        p = registry.get_paper("a")
        assert p["status"] == "embedded" and p["parsed_at"] and p["processed_at"] and p["embedded_at"]
        assert registry.get_last_domain_date("d").startswith("2026-09-01")

    def test_unknown_paper_cannot_advance(self, registry):
        assert registry.update_status("ghost", "parsed") is False
        assert registry.get_status("ghost") is None

    def test_error_counts_and_force_reset(self, registry):
        registry.update_status("b", "downloaded")
        assert registry.update_status("b", "error", error_msg="boom")
        assert registry.update_status("b", "error", error_msg="again")
        p = registry.get_paper("b")
        assert p["status"] == "error" and p["error_count"] == 2 and p["last_error"] == "again"
        assert registry.update_status("b", "downloaded") and registry.get_status("b") == "error"   # no-op without force
        assert registry.update_status("b", "downloaded", force=True) and registry.get_status("b") == "downloaded"
        assert registry.get_paper("b")["error_count"] == 0

    def test_stats(self, registry):
        registry.update_status("a", "downloaded")
        for s in ("parsed", "processed", "embedded"):
            registry.update_status("a", s)
        registry.update_status("b", "downloaded")
        registry.update_status("b", "error", error_msg="x")
        st = registry.stats()
        assert st["counts"]["embedded"] == 1 and st["counts"]["error"] == 1 and st["total"] == 2
        assert st["errors"][0]["filename"] == "b"
        assert st["stage_seconds"]["total"] is not None
        assert registry.list_papers(status="error")[0]["filename"] == "b"
