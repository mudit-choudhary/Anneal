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


def test_data_home_is_outside_the_repo(monkeypatch, tmp_path):
    """The code folder must stay writable-free: nothing the app writes may
    resolve inside the repository."""
    import importlib
    import common.paths as paths

    for p in (paths.DATA_DIR, paths.VECTOR_DB, paths.MODELS_DIR, paths.RUN_DIR, paths.REGISTRY_DB):
        assert paths.PROJECT_ROOT not in p.parents, f"{p} is inside the repository"
        assert paths.DATA_HOME in p.parents or p == paths.DATA_HOME

    monkeypatch.setenv("ANNEAL_HOME", str(tmp_path / "elsewhere"))
    reloaded = importlib.reload(paths)
    try:
        assert reloaded.DATA_HOME == tmp_path / "elsewhere"
        assert reloaded.PDF_DIR == tmp_path / "elsewhere" / "data" / "raw_pdfs"
    finally:
        monkeypatch.delenv("ANNEAL_HOME")
        importlib.reload(paths)


def test_xdg_data_home_is_ignored(monkeypatch, tmp_path):
    """Snap terminals set XDG_DATA_HOME to a private dir; honouring it would
    silently switch corpora depending on which terminal started the app."""
    import importlib
    import common.paths as paths

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "snap-private"))
    monkeypatch.delenv("ANNEAL_HOME", raising=False)
    try:
        assert importlib.reload(paths).DATA_HOME == Path.home() / ".local" / "share" / "anneal"
    finally:
        monkeypatch.delenv("XDG_DATA_HOME")
        importlib.reload(paths)


class TestPorts:
    def test_defaults(self, monkeypatch):
        import importlib
        import common.paths as paths
        for var in ("ANNEAL_UI_PORT", "ANNEAL_REGISTRY_PORT", "ANNEAL_EMBEDDING_PORT"):
            monkeypatch.delenv(var, raising=False)
        p = importlib.reload(paths)
        assert (p.REGISTRY_PORT, p.EMBEDDING_PORT, p.UI_PORT) == (4000, 4001, 4002)
        assert p.UI_URL == "http://127.0.0.1:4002"

    def test_environment_moves_every_url(self, monkeypatch):
        import importlib
        import common.paths as paths
        monkeypatch.setenv("ANNEAL_UI_PORT", "4102")
        monkeypatch.setenv("ANNEAL_REGISTRY_PORT", "4100")
        monkeypatch.setenv("ANNEAL_EMBEDDING_PORT", "4101")
        try:
            p = importlib.reload(paths)
            assert p.UI_URL == "http://127.0.0.1:4102"
            assert p.REGISTRY_URL == "http://127.0.0.1:4100"
            assert p.EMBEDDING_URL == "http://127.0.0.1:4101"
            assert p.SERVICE_PORTS == {"registry": 4100, "embedding": 4101, "ui": 4102}
        finally:
            for var in ("ANNEAL_UI_PORT", "ANNEAL_REGISTRY_PORT", "ANNEAL_EMBEDDING_PORT"):
                monkeypatch.delenv(var)
            importlib.reload(paths)

    @pytest.mark.parametrize("value", ["abc", "0", "70000", "-1"])
    def test_a_bad_port_fails_loudly(self, monkeypatch, value):
        """Silently falling back would leave a service listening where nothing calls it."""
        import importlib
        import common.paths as paths
        monkeypatch.setenv("ANNEAL_UI_PORT", value)
        try:
            with pytest.raises(ValueError):
                importlib.reload(paths)
        finally:
            monkeypatch.delenv("ANNEAL_UI_PORT")
            importlib.reload(paths)

    def test_cli_flags_become_environment_before_paths_loads(self):
        """ops.py strips them in argv order; everything else must survive."""
        import importlib.util
        import os
        spec = importlib.util.spec_from_file_location("ops_flags", APP_ROOT / "scripts" / "ops.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        saved = {k: os.environ.get(k) for k in mod.PORT_FLAGS.values()}
        try:
            rest = mod.take_port_flags(["--port", "4102", "status", "--registry-port=4100", "--all"])
            assert rest == ["status", "--all"]
            assert os.environ["ANNEAL_UI_PORT"] == "4102"
            assert os.environ["ANNEAL_REGISTRY_PORT"] == "4100"
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
