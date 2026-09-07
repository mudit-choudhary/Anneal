"""Smoke tests for the web UI server (no Ollama/embedding services needed —
status endpoints degrade gracefully when they're down)."""

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("httpx")  # required by fastapi.testclient
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def client():
    # UI/main.py inserts rag_setup on sys.path and imports its config; clear
    # any parse_manager `config` already cached by other tests first.
    saved = {k: sys.modules.pop(k) for k in ("config", "rag") if k in sys.modules}
    spec = importlib.util.spec_from_file_location("ui_main", REPO_ROOT / "UI" / "main.py")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        for k in ("config", "rag"):
            sys.modules.pop(k, None)
        sys.modules.update(saved)
    return TestClient(module.app)


def test_index_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Research Paper" in r.text
    assert "api/query" in r.text


def test_status_degrades_gracefully(client):
    r = client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert {"ollama", "embeddings", "model", "default_backend"} <= set(body)


def test_papers_empty_when_service_down(client):
    r = client.get("/api/papers")
    assert r.status_code == 200
    assert isinstance(r.json()["files"], list)


def test_query_reports_unreachable_embedding_service(client):
    r = client.post("/api/query", json={"query": "what is x?"})
    assert r.status_code == 200
    first = r.text.strip().split("\n")[0]
    assert '"error"' in first and "unreachable" in first
