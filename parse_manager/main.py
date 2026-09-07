"""Parse manager service.

Two polling loops share one process:
- parser_loop:    raw_pdfs/*.pdf with registry status "downloaded" -> layout
                  JSON in parsed/ (status becomes "parsed")
- processor_loop: parsed/*.json with registry status "parsed" -> assembled
                  text in processed/ (status becomes "processed")
"""

import time
import threading

import requests

from config import PDF_DIR, PARSED_DIR, PROCESSED_DIR, REGISTRY_URL
from pdf_parser import parse
from txt_processor import process_layout_json


def get_status(filename):
    try:
        response = requests.get(
            f"{REGISTRY_URL}/get_status",
            json={"filename": filename},
            timeout=20,
        )
        return response.json().get("status")
    except requests.RequestException as e:
        print(f"[registry] unreachable: {e}")
        return None


def parser_loop(poll_interval=5):
    while True:
        for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
            if get_status(pdf_path.stem) != "downloaded":
                continue
            try:
                parse(pdf_path)
            except Exception as e:
                print(f"[parser] error on {pdf_path.name}: {e}")
        time.sleep(poll_interval)


def processor_loop(poll_interval=5):
    while True:
        for json_path in sorted(PARSED_DIR.glob("*.json")):
            if get_status(json_path.stem) != "parsed":
                continue
            try:
                process_layout_json(json_path)
            except Exception as e:
                print(f"[processor] error on {json_path.name}: {e}")
        time.sleep(poll_interval)


if __name__ == "__main__":
    for d in (PDF_DIR, PARSED_DIR, PROCESSED_DIR):
        d.mkdir(parents=True, exist_ok=True)

    threading.Thread(target=parser_loop, daemon=True).start()
    threading.Thread(target=processor_loop, daemon=True).start()

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Shutting down...")
