"""ChromaDB access: embedding papers and saved chats, and searching both."""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb
from chromadb.utils import embedding_functions

from chunking import chunk_file
from common.logsetup import get_logger
from config import (
    CHATS_COLLECTION_NAME,
    CHUNK_MAX_CHARS,
    CHUNK_TARGET_CHARS,
    COLLECTION_NAME,
    EMBEDDING_VECTOR_PATH,
    MODEL_NAME,
    QUERY_INSTRUCTION,
)

log = get_logger("embedding")

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


def get_collection(name: str = COLLECTION_NAME):
    return client.get_or_create_collection(name=name, embedding_function=embed_fn,
                                           metadata={"hnsw:space": "cosine"})


# --------------------------------------------------------------- papers
def chunk_and_embed(json_path: str) -> Dict[str, Any]:
    """Chunk a processed JSON (data/processed/<name>.json) and embed it.
    Re-embedding the same paper replaces its previous chunks."""
    json_path = Path(json_path)
    filename = json_path.stem
    chunks, skipped = chunk_file(json_path, CHUNK_TARGET_CHARS, CHUNK_MAX_CHARS)
    if not chunks:
        return {"filename": filename, "chunk_count": 0, "success": False, "error": "no embeddable blocks"}
    collection = get_collection()
    collection.delete(where={"filename": filename})
    collection.add(
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
        ids=[f"{filename}_{c['metadata']['chunk_index']}" for c in chunks],
    )
    log.info("embedded %d chunks from %s (%d blocks not embedded)", len(chunks), filename, len(skipped))
    return {"filename": filename, "chunk_count": len(chunks), "success": True}


def list_embedded_files() -> List[str]:
    """Distinct paper stems present in the papers collection."""
    try:
        metadatas = get_collection().get(include=["metadatas"])["metadatas"]
        return sorted({m["filename"] for m in metadatas if m and "filename" in m})
    except Exception:
        return []


# --------------------------------------------------------------- chats
def embed_chat(chat_id: str, title: str, created_at: str, pairs: Sequence[Dict[str, str]]) -> int:
    """Embed a saved conversation as one document per (question, answer) pair."""
    collection = get_collection(CHATS_COLLECTION_NAME)
    collection.delete(where={"chat_id": chat_id})
    if not pairs:
        return 0
    collection.add(
        documents=[f"Q: {p['question']}\nA: {p['answer']}" for p in pairs],
        metadatas=[{"chat_id": chat_id, "title": title, "created_at": created_at, "pair_index": i}
                   for i in range(len(pairs))],
        ids=[f"chat_{chat_id}_{i}" for i in range(len(pairs))],
    )
    log.info("embedded chat %s (%d pairs)", chat_id, len(pairs))
    return len(pairs)


def delete_chat(chat_id: str) -> None:
    get_collection(CHATS_COLLECTION_NAME).delete(where={"chat_id": chat_id})


# --------------------------------------------------------------- search
def search(query: str, n_results: int = 8, filenames: Optional[List[str]] = None,
           sources: Sequence[str] = ("papers",), n_chat_results: int = 2) -> List[Dict[str, Any]]:
    """Search papers (and optionally saved chats). Returns a flat list of
    {id, text, metadata, distance, source} sorted by distance within source."""
    results: List[Dict[str, Any]] = []
    if "papers" in sources:
        where = {"filename": {"$in": filenames}} if filenames else None
        res = get_collection().query(query_texts=[query], n_results=n_results, where=where)
        results += _flatten(res, "papers")
    if "chats" in sources and n_chat_results > 0:
        try:
            if get_collection(CHATS_COLLECTION_NAME).count():
                res = get_collection(CHATS_COLLECTION_NAME).query(query_texts=[query], n_results=n_chat_results)
                results += _flatten(res, "chats")
        except Exception as e:  # an empty/new collection must never break paper search
            log.warning("chat search failed: %s", e)
    return results


def _flatten(res, source):
    out = []
    for i, doc in enumerate(res["documents"][0]):
        out.append({"id": res["ids"][0][i], "text": doc, "metadata": res["metadatas"][0][i],
                    "distance": res["distances"][0][i], "source": source})
    return out
