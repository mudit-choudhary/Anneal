import os
import time
import threading

from pdf_parser import parse
from txt_processor import process_pdf_txt
from config import PDF_DIR, PROCESSED_DIR, PARSED_DIR, REGISTRY_URL

import requests

def parser_loop(poll_interval: int = 5):
    """Continuously scan PDF_DIR and parse any PDFs not yet processed."""
    while True:
        pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
        for file in pdf_files:
            filename = file.split('.')[0]
            response = requests.get(f"{REGISTRY_URL}/get_status",
                                    json={"filename": filename},
                                    timeout=20
                                )
            response_data = response.json()
            status = response_data['status']
            if status != "downloaded":
                continue
            pdf_path = os.path.join(PDF_DIR, file)
            try:
                parse(pdf_path)
            except Exception as e:
                print(f"[parser] error on {file}: {e}")
        time.sleep(poll_interval)


def processor_loop(poll_interval: int = 5):
    """Continuously scan PARSED_DIR and process any new TXT files."""
    while True:
        txt_files = [f for f in os.listdir(PARSED_DIR) if f.lower().endswith(".txt")]
        for file in txt_files:
            filename = file.split('.')[0]
            response = requests.get(f"{REGISTRY_URL}/get_status",
                                    json={"filename": filename},
                                    timeout=20
                                )
            response_data = response.json()
            status = response_data['status']
            if status != "downloaded":
                continue
            print("Processing File: ", file)
            filename = file.rsplit('.', 1)[0]
            input_file = os.path.join(PARSED_DIR, file)
            output_txt = os.path.join(PROCESSED_DIR, file)
            output_json = os.path.join(PROCESSED_DIR, f"{filename}.json")
            try:
                process_pdf_txt(input_file, output_txt, output_json)
            except Exception as e:
                print(f"[processor] error on {file}: {e}")
        time.sleep(poll_interval)


if __name__ == "__main__":
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR, exist_ok=True)

    if not os.path.exists(PARSED_DIR):
        os.makedirs(PARSED_DIR, exist_ok=True)

    t1 = threading.Thread(target=parser_loop, daemon=True)
    t2 = threading.Thread(target=processor_loop, daemon=True)

    t1.start()
    t2.start()

    try:
        # Keep main thread alive until manually stopped
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Shutting down...")
