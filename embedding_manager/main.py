import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, model_validator, field_validator
from typing import Optional, List
import os
import time
import threading
import requests

from config import PROCESSED_DIR, REGISTRY_URL
from embeddings import chunk_and_embed, query_embeddings, list_embedded_files

def chunks_and_embed_loop():  # ✅ Simple loop, NO uvicorn
    while True:
        try:
            file_list = os.listdir(PROCESSED_DIR)
            for file in file_list:
                if not file.endswith('.txt'):  # Skip non-txt
                    continue
                    
                filename = os.path.splitext(file)[0]
                response = requests.get(
                    f"{REGISTRY_URL}/get_status",
                    json={"filename": filename},
                    timeout=20
                )
                
                if response.status_code != 200:
                    print(f"Registry error for {filename}: {response.status_code}")
                    continue
                    
                response_data = response.json()
                status = response_data.get('status')
                
                if status == "embedded":
                    continue
                    
                file_path = os.path.join(PROCESSED_DIR, file)
                num_chunks = chunk_and_embed(file_path)
                
                # ✅ Update registry status
                update_data = {"filename": filename, "status": "embedded"}
                update_response = requests.post(
                    f"{REGISTRY_URL}/update_status",
                    json=update_data,
                    timeout=20
                )
                
                if update_response.json().get('success'):
                    print(f"✅ Embedded {filename} ({num_chunks} chunks)")
                else:
                    print(f"❌ Failed to update status for {filename}")
                    
        except Exception as e:
            print(f"Embedding loop error: {e}")
            
        time.sleep(300)

class QueryRequest(BaseModel):
    query: str
    filenames: Optional[List[str]] = None

    @field_validator('*', mode='before')
    @classmethod
    def strip_strings(cls, v):
        if isinstance(v, str):
            return v.strip()
        
        return v

    @model_validator(mode='after')
    def validate_error_status(self):
        if not self.query:
            raise ValueError("Missing Query. Send query to fetch chunks.")
        return self

embedding = FastAPI()

@embedding.get("/")
def read_root():
    return {"Welcome": " to embedding manager!"}

@embedding.post('/healthcheck')
def healthcheck():
    return {"Status": "Okay"}

@embedding.get('/get_chunks')
def fetch_relevant_chunks(request: QueryRequest):
    response = query_embeddings(query = request.query, filename_filter=request.filenames)
    return response

@embedding.get('/list_files')
def list_files():
    return {"files": list_embedded_files()}

# def start_and_run():
#     uvicorn.run("main:embedding", host="127.0.0.1", port=4001, reload=True)

if __name__ == "__main__":
    embedding_thread = threading.Thread(target=chunks_and_embed_loop, daemon=True)
    embedding_thread.start()

    uvicorn.run("main:embedding", host="127.0.0.1", port=4001, reload=True)
    
    