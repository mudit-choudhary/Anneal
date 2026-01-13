from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.utils import embedding_functions
import os
import chromadb
from typing import List, Dict, Optional, Any
import numpy as np
from pathlib import Path

from config import MODEL_NAME, EMBEDDING_VECTOR_PATH
if not os.path.exists(EMBEDDING_VECTOR_PATH):
    os.makedirs(EMBEDDING_VECTOR_PATH, exist_ok=True)
    
client = chromadb.PersistentClient(path=EMBEDDING_VECTOR_PATH)

# Initialize embedding function once (global reuse)
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=MODEL_NAME, device="cuda"
)

def chunk_and_embed(file_path: str, chunk_size: int = 512, chunk_overlap_pct: float = 0.2) -> Dict[str, Any]:
    """
    Process txt file: chunk -> embed -> store incrementally in ChromaDB.
    
    Returns: {'filename': str, 'chunk_count': int, 'success': bool}
    """
    filename = Path(file_path).name  # e.g., "doc1.txt"
    
    # Step 1: Chunk document
    chunk_overlap = int(chunk_size * chunk_overlap_pct) + 1
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]  # Better sentence boundaries [web:44]
    )
    
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
        
        chunks = splitter.split_text(text)
        if not chunks:
            return {"filename": filename, "chunk_count": 0, "success": False, "error": "Empty file"}
        
        # Step 2: Prepare metadata and IDs (unique across all docs)
        metadatas = [{"filename": filename, "chunk_id": i, "chunk_size": len(chunks[i])} 
                    for i in range(len(chunks))]
        ids = [f"{filename}_{i}" for i in range(len(chunks))]
        
        # Step 3: Get or create collection (cosine similarity, HNSW index)
        collection = client.get_or_create_collection(
            name="rag_documents",
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"}  # Best for text embeddings [web:66]
        )
        
        # Step 4: Add incrementally (no rebuild needed)
        collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✅ Embedded {len(chunks)} chunks from {filename}")
        return {"filename": filename, "chunk_count": len(chunks), "success": True}
        
    except Exception as e:
        print(f"❌ Error processing {filename}: {str(e)}")
        return {"filename": filename, "chunk_count": 0, "success": False, "error": str(e)}

# Query function for completeness
def query_embeddings(query: str, n_results: int = 5, filename_filter: Optional[List[str]] = None) -> Dict:
    """Search across all embedded documents."""
    collection = client.get_collection(name="rag_documents")
    
    where_clause = {"filename": {"$in": filename_filter}} if filename_filter else None
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_clause
    )
    
    return {
        "query": query,
        "documents": results['documents'][0],
        "metadatas": results['metadatas'][0],
        "distances": results['distances'][0]
    }

# # Example usage
# if __name__ == "__main__":
#     result = chunking_and_embedding("doc1.txt")
#     print(result)
    
#     # Query example
#     query_result = query_embeddings("What is the main topic?")
#     print(query_result['documents'][:2])  # Top 2 chunks
