"""Chunking logic, separated from embeddings.py so it can be imported (e.g.
by scripts/rag_inspect.py) without triggering the ChromaDB client or the
SentenceTransformer model load."""

from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP_PCT


def make_splitter(chunk_size: int = CHUNK_SIZE,
                  chunk_overlap_pct: float = CHUNK_OVERLAP_PCT) -> RecursiveCharacterTextSplitter:
    chunk_overlap = int(chunk_size * chunk_overlap_pct) + 1
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               chunk_overlap_pct: float = CHUNK_OVERLAP_PCT) -> List[str]:
    return make_splitter(chunk_size, chunk_overlap_pct).split_text(text)
