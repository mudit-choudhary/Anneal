from google import genai
import requests
from typing import Optional, List
# from google.genai import types
# import mimetypes
from config import *

PARSED_DIR = "/home/mudit/Desktop/PaperParsing/data/parsed"
def main(query: str, filenames: Optional[List[str]] = None):
    client = genai.Client(api_key = GEMINI_API_KEY)  # Auto-uses GEMINI_API_KEY env var
    response = requests.get(f"{EMBEDDING_URL}/get_chunks",
                                    json={"query": query,
                                          "filenames": filenames},
                                    timeout=90
                                )
    response_data = response.json()

    print("Chunks: ", response_data["documents"], "\n", "="*50, "\n")
    # Prepare multimodal content: file + query
    contents = [
        # types.Part.from_bytes(data=file_data, mime_type=mime_type),
        response_data["documents"],
        query  # Text prompt after file for best results
    ]
    
    # Generate response (add config for temperature, etc., as needed)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents
    )
    
    print(response.text)

if __name__ == "__main__":
    query = input("Please enter your query: ")
    filenames = input("Please enter your filenames separated by a comma (,): ")
    file_list = filenames.split(',') if filenames else None
    main(query, file_list)
