# Runs every 30 minutes to check if any file from any directory has moved up from it's directorial status.
# directory:::::::StatusExpected:::::::::StatusForDeletion
# ===========================================================================
# raw_pdfs::::::::::downloaded::::::::::::parsed, processed, embedded
# parsed::::::::::::::parsed::::::::::::::processed, embedded
# processed:::::::::processed:::::::::::::embedded

import os
import requests
import threading
import time
from config import PROCESSED_DIR, PARSED_DIR, PDF_DIR, REGISTRY_URL


def check_downloads_directory():
    while True:
        if os.path.exists(PDF_DIR):
            try:
                file_list = os.listdir(PDF_DIR)

                for file in file_list:
                    filename = file.split('.')[0]
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

                    if status not in ["downloaded", "error"]:
                        filepath = PDF_DIR + "/" + file
                        os.remove(filepath)
            except Exception as e:
                print(e)
                continue
        else:
             continue
        

def check_parsed_directory():
    while True:
        if os.path.exists(PARSED_DIR):
            try:
                file_list = os.listdir(PARSED_DIR)

                for file in file_list:
                    filename = file.split('.')[0]
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

                    if status not in ["downloaded", "parsed", "error"]:
                        filepath = PARSED_DIR + "/" + file
                        os.remove(filepath)
            except Exception as e:
                print(e)
                continue
        else:
             continue
            
def check_processed_directory():
    while True:
        if os.path.exists(PROCESSED_DIR):
            try:
                file_list = os.listdir(PROCESSED_DIR)

                for file in file_list:
                    filename = file.split('.')[0]
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

                    if status not in ["downloaded", "parsed", "processed", "error"]:
                        filepath = PROCESSED_DIR + "/" + file
                        os.remove(filepath)
            except Exception as e:
                print(e)
                continue
        else:
             continue
            
def deletion_loop():
    t1 = threading.Thread(target=check_downloads_directory, daemon=True)
    t2 = threading.Thread(target=check_parsed_directory, daemon=True)
    t3 = threading.Thread(target=check_processed_directory, daemon=True)

    t1.start()
    t2.start()
    t3.start()

    try:
        # Keep main thread alive until manually stopped
        while True:
            time.sleep(1800)
    except KeyboardInterrupt:
        print("Shutting down...")