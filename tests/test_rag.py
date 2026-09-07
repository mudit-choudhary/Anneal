"""Unit tests for the RAG prompt construction (no services required).

rag_setup has its own config.py, which collides with parse_manager's on
sys.path; the fixture below imports the module in isolation and restores
whatever `config` was loaded before.
"""

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def rag():
    saved_config = sys.modules.pop("config", None)
    sys.path.insert(0, str(REPO_ROOT / "rag_setup"))
    try:
        module = importlib.import_module("rag")
    finally:
        sys.path.pop(0)
        sys.modules.pop("config", None)
        if saved_config is not None:
            sys.modules["config"] = saved_config
    return module


def test_build_context_numbers_and_sources(rag):
    docs = ["First chunk text.", "Second chunk text."]
    metas = [{"filename": "A_Great_Paper.txt"}, {"filename": "Another_One.txt"}]
    ctx = rag.build_context(docs, metas)
    assert "[1] (from: A Great Paper)\nFirst chunk text." in ctx
    assert "[2] (from: Another One)\nSecond chunk text." in ctx


def test_build_context_handles_missing_filename(rag):
    ctx = rag.build_context(["text"], [{}])
    assert "(from: unknown)" in ctx


def test_build_user_prompt_contains_query_and_context(rag):
    prompt = rag.build_user_prompt("CTX", "What is X?")
    assert "CTX" in prompt
    assert prompt.rstrip().endswith("Question: What is X?")


def test_system_prompt_demands_citations(rag):
    assert "cite" in rag.SYSTEM_PROMPT.lower()
    assert "ONLY" in rag.SYSTEM_PROMPT
