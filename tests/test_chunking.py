"""Anneal uses Grain-Growth Chunking as a library.

The algorithm's own 36 tests live with it, in the grain-growth-chunking
repository. What matters here is the seam: the dependency is installed, the
version is the evaluated one, and a document chunks the way the pipeline
expects.
"""

import pytest

grain_growth = pytest.importorskip("grain_growth", reason="pip install -r app/requirements.txt")


def test_the_evaluated_version_is_installed():
    # A different major version would move chunk boundaries, so a vector store
    # built earlier would no longer line up with newly ingested papers.
    assert grain_growth.__version__.split(".")[0] == "1"


def test_a_document_chunks_the_way_the_pipeline_expects():
    blocks = [
        {"type": "title", "page": 0, "text": "A Paper"},
        {"type": "section", "page": 1, "text": "3 Method"},
        {"type": "paragraph", "page": 1, "text": "We train the model. " * 40},
        {"type": "table", "page": 2, "text": "| a | b |"},
        {"type": "authors", "page": 0, "text": "A. Author"},
    ]
    chunks, skipped = grain_growth.chunk_document(blocks, "A_Paper")

    assert [s["type"] for s in skipped] == ["authors"]        # never embedded
    assert all(c["text"].startswith("A Paper") for c in chunks)   # heading path prefixed
    assert any(c["metadata"]["block_types"] == "table" for c in chunks)   # tables stand alone
    for c in chunks:
        m = c["metadata"]
        assert {"filename", "title", "section", "page_start", "page_end",
                "block_types", "chunk_index", "n_chars"} <= set(m)
