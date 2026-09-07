import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils import embedding_functions

from config import (
    MODEL_NAME,
    QUERY_INSTRUCTION,
    COLLECTION_NAME,
    EMBEDDING_VECTOR_PATH,
    CHUNK_TARGET_CHARS,
    CHUNK_MAX_CHARS,
)
from chunking import chunk_file

os.makedirs(EMBEDDING_VECTOR_PATH, exist_ok=True)
client = chromadb.PersistentClient(path=str(EMBEDDING_VECTOR_PATH))


class BGEEmbeddingFunction(embedding_functions.SentenceTransformerEmbeddingFunction):
    """Documents are embedded as-is; queries get bge's retrieval instruction."""

    def embed_query(self, input):
        return self.__call__([QUERY_INSTRUCTION + q for q in input])


# Initialized once. EMBED_DEVICE=cpu keeps VRAM free for the local LLM during
# daytime querying (a single query embeds in tens of ms on CPU); use cuda —
# the default — for overnight batch embedding.
embed_fn = BGEEmbeddingFunction(
    model_name=MODEL_NAME,
    device=os.environ.get("EMBED_DEVICE", "cuda"),
    normalize_embeddings=True,
)


def get_collection():
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


def chunk_and_embed(json_path: str) -> Dict[str, Any]:
    """Chunk a processed JSON (data/processed/<name>.json) and embed it.

    Re-embedding the same paper replaces its previous chunks, so re-parsing
    a paper and re-running is safe.
    Returns: {'filename': str, 'chunk_count': int, 'success': bool}
    """
    json_path = Path(json_path)
    filename = json_path.stem
    try:
        chunks, skipped = chunk_file(json_path, CHUNK_TARGET_CHARS, CHUNK_MAX_CHARS)
        if not chunks:
            return {"filename": filename, "chunk_count": 0, "success": False,
                    "error": "no embeddable blocks"}

        collection = get_collection()
        collection.delete(where={"filename": filename})
        collection.add(
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
            ids=[f"{filename}_{c['metadata']['chunk_index']}" for c in chunks],
        )
        print(f"✅ Embedded {len(chunks)} chunks from {filename} "
              f"({len(skipped)} blocks not embedded: authors/footnotes/equation numbers)")
        return {"filename": filename, "chunk_count": len(chunks), "success": True}
    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")
        return {"filename": filename, "chunk_count": 0, "success": False, "error": str(e)}


def list_embedded_files() -> List[str]:
    """Distinct source filenames present in the collection."""
    try:
        collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)
        metadatas = collection.get(include=["metadatas"])["metadatas"]
        return sorted({m["filename"] for m in metadatas if m and "filename" in m})
    except Exception:
        return []


def query_embeddings(query: str, n_results: int = 8,
                     filename_filter: Optional[List[str]] = None) -> Dict:
    """Search across all embedded documents."""
    collection = get_collection()
    where_clause = {"filename": {"$in": filename_filter}} if filename_filter else None
    results = collection.query(query_texts=[query], n_results=n_results, where=where_clause)
    return {
        "query": query,
        "documents": results["documents"][0],
        "metadatas": results["metadatas"][0],
        "distances": results["distances"][0],
    }
