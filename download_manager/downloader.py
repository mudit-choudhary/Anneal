# # import arxiv
# # import os
# # import re
# # from config import PDF_DIR, DOMAINS, REGISTRY_URL
# # from loguru import logger
# # import requests


# # def sanitize_filename(title):
# #     return re.sub(r'[^a-zA-Z0-9_\- ]', '', title).replace(' ', '_')[:100]

# # def download_papers(max_papers=5):
# #     if not os.path.exists(PDF_DIR):
# #         os.makedirs(PDF_DIR)

# #     client = arxiv.Client()

# #     for domain in DOMAINS:
# #         logger.info(f"🔍 Searching: {domain}")
# #         search = arxiv.Search(
# #             query=f'ti:"{domain}" OR abs:"{domain}"',
# #             max_results=max_papers,
# #             sort_by=arxiv.SortCriterion.SubmittedDate
# #         )

# #         for result in client.results(search):
# #             paper_id = result.entry_id.split('/')[-1] # Extract ArXiv ID
# #             file = sanitize_filename(result.title)
# #             filename = file + ".pdf"
            
# #             try:
# #                 data = {"filename": file}
                
# #                 response = requests.get(
# #                     f"{REGISTRY_URL}/get_status",
# #                     json=data,
# #                     timeout=20
# #                 )
# #                 response_data = response.json()
# #                 # print("Response Data: ", response_data)
# #                 status = response_data['status']
# #             except Exception as e:
# #                 raise e
            
# #             if status:
# #                 logger.info(f"⏭️  Skipping {paper_id} ({status})")
# #                 continue

# #             logger.info(f"⬇️  Downloading: {result.title}")
# #             try:
# #                 result.download_pdf(dirpath=PDF_DIR, filename=filename)
# #                 print("Downloaded")
# #                 data = {"filename": file,
# #                         "status": "downloaded"}
                
# #                 response = requests.post(
# #                     f"{REGISTRY_URL}/update_status",
# #                     json=data,
# #                     timeout=20
# #                 )
# #                 # print("Response: ", response)
# #                 response_data = response.json()
# #                 # print(f"Response Data: {response_data}")
# #                 if not response_data['success']:
# #                     raise
# #             except Exception as e:
# #                 logger.error(f"❌ Failed {result.title}: {e}")

# # if __name__ == "__main__":
# #     download_papers(max_papers=3)

# import arxiv
# import os
# import re
# import time
# import random
# import requests
# from datetime import datetime, timedelta, timezone
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from loguru import logger
# from config import PDF_DIR, DOMAINS, REGISTRY_URL

# # --- Configuration ---
# BACKFILL_DAYS = 1095       # ~3 years (previously YEARS=3)
# CHECK_INTERVAL = 3600      # Main loop checks every hour

# # --- Anti-Abuse / Jitter Configuration ---
# MAX_WORKERS = len(DOMAINS) # One thread per domain
# START_JITTER_RANGE = (5, 60)   # Random delay before a thread starts (seconds)
# DOWNLOAD_SLEEP_RANGE = (3, 10) # Random pause between individual downloads
# BREAK_AFTER_RANGE = (10, 20)    # Take a long break after downloading X papers
# BREAK_DURATION_RANGE = (45, 120)# How long the "long break" lasts (seconds)

# def sanitize_filename(title):
#     return re.sub(r'[^a-zA-Z0-9_\- ]', '', title).replace(' ', '_')[:100]

# def get_cutoff_date(days):
#     return datetime.now(timezone.utc) - timedelta(days=days)

# def process_domain(domain, cutoff_date):
#     """
#     Handles search & download for a domain with randomized delays.
#     """
    
#     # 1. DESYNC START: Sleep randomly so all threads don't hit the API at the exact same second
#     start_delay = random.uniform(*START_JITTER_RANGE)
#     logger.info(f"⏳ [Jitter] {domain} waiting {start_delay:.2f}s to start...")
#     time.sleep(start_delay)

#     # Initialize Client with a baseline delay
#     client = arxiv.Client(
#         page_size=100, 
#         delay_seconds=3.0, 
#         num_retries=5
#     )
    
#     logger.info(f"🚀 [Start] Scanning: {domain}")
    
#     search = arxiv.Search(
#         query=f'ti:"{domain}" OR abs:"{domain}"',
#         max_results=None,
#         sort_by=arxiv.SortCriterion.SubmittedDate
#     )

#     # Counters for the "Long Break" logic
#     session_downloads = 0
#     next_break_threshold = random.randint(*BREAK_AFTER_RANGE)
    
#     total_downloads = 0
#     checked_count = 0

#     try:
#         for result in client.results(search):
#             checked_count += 1
            
#             # --- DATE GUARDRAIL ---
#             if result.published < cutoff_date:
#                 break
            
#             paper_id = result.entry_id.split('/')[-1]
#             file_clean = sanitize_filename(result.title)
#             filename = file_clean + ".pdf"

#             # --- REGISTRY CHECK ---
#             try:
#                 check_resp = requests.get(
#                     f"{REGISTRY_URL}/get_status",
#                     json={"filename": file_clean},
#                     timeout=10
#                 )
#                 if check_resp.status_code == 200 and check_resp.json().get('status'):
#                     continue 
#             except Exception:
#                 pass 

#             # --- DOWNLOAD ---
#             logger.info(f"⬇️  Downloading [{domain}]: {result.title[:30]}...")
#             try:
#                 result.download_pdf(dirpath=PDF_DIR, filename=filename)
                
#                 requests.post(
#                     f"{REGISTRY_URL}/update_status",
#                     json={"filename": file_clean, "status": "downloaded"},
#                     timeout=10
#                 )
                
#                 total_downloads += 1
#                 session_downloads += 1
                
#                 # --- RANDOM SLEEP LOGIC ---
                
#                 # Check if we need a "Long Break"
#                 if session_downloads >= next_break_threshold:
#                     break_time = random.uniform(*BREAK_DURATION_RANGE)
#                     logger.warning(f"☕ [{domain}] Taking a break for {break_time:.1f}s after {session_downloads} downloads...")
#                     time.sleep(break_time)
                    
#                     # Reset counters for next batch
#                     session_downloads = 0
#                     next_break_threshold = random.randint(*BREAK_AFTER_RANGE)
#                 else:
#                     # Standard short sleep between files
#                     short_sleep = random.uniform(*DOWNLOAD_SLEEP_RANGE)
#                     time.sleep(short_sleep)

#             except Exception as e:
#                 logger.error(f"❌ Failed [{domain}] {result.title[:20]}: {e}")

#     except Exception as e:
#         logger.error(f"⚠️ Error in thread {domain}: {e}")
        
#     logger.success(f"✅ [Done] {domain}: Checked {checked_count}, Downloaded {total_downloads}")
#     return domain

# def main_loop():
#     if not os.path.exists(PDF_DIR):
#         os.makedirs(PDF_DIR)

#     logger.info(f"🔥 Starting Parallel Monitor (Workers: {MAX_WORKERS}, Backfill: {BACKFILL_DAYS} days)")

#     while True:
#         cycle_start = time.time()
#         cutoff_date = get_cutoff_date(BACKFILL_DAYS)
        
#         with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#             future_to_domain = {
#                 executor.submit(process_domain, domain, cutoff_date): domain 
#                 for domain in DOMAINS
#             }
            
#             for future in as_completed(future_to_domain):
#                 domain = future_to_domain[future]
#                 try:
#                     future.result() 
#                 except Exception as exc:
#                     logger.error(f"Thread for {domain} crashed: {exc}")

#         elapsed = time.time() - cycle_start
#         logger.info(f"💤 Cycle complete in {elapsed:.2f}s. Sleeping for {CHECK_INTERVAL}s...")
#         time.sleep(CHECK_INTERVAL)

# if __name__ == "__main__":
#     try:
#         main_loop()
#     except KeyboardInterrupt:
#         logger.info("🛑 Monitor stopped.")

import arxiv
import os
import re
import time
import random
import requests
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
from config import PDF_DIR, DOMAINS, REGISTRY_URL

# --- Configuration ---
BACKFILL_DAYS = 1095  # Fallback if DB is empty (3 years)
CHECK_INTERVAL = 3600
MAX_WORKERS = len(DOMAINS)

# Jitter Config
START_JITTER = (5, 60)
DOWNLOAD_SLEEP = (3, 10)
BREAK_AFTER = (5, 12)
BREAK_DURATION = (30, 90)

def sanitize_filename(title):
    return re.sub(r'[^a-zA-Z0-9_\- ]', '', title).replace(' ', '_')[:100]

def get_smart_cutoff(domain):
    """
    Asks the Registry: 'When was the last paper I downloaded for this domain?'
    """
    try:
        response = requests.post(
            f"{REGISTRY_URL}/get_last_checkpoint",
            json={"domain": domain},
            timeout=5
        )
        data = response.json()
        date_str = data.get("last_checkpoint")
        
        if date_str:
            # Convert string back to timezone-aware datetime
            # ArXiv uses UTC, so we ensure this is UTC
            checkpoint = datetime.fromisoformat(date_str)
            if checkpoint.tzinfo is None:
                checkpoint = checkpoint.replace(tzinfo=timezone.utc)
            
            logger.info(f"🧠 Smart Resume [{domain}]: Searching only after {checkpoint.date()}")
            return checkpoint
        
    except Exception as e:
        logger.warning(f"⚠️ Could not fetch checkpoint for {domain}: {e}")

    # Fallback: 3 Years ago
    fallback = datetime.now(timezone.utc) - timedelta(days=BACKFILL_DAYS)
    logger.info(f"🔙 Full Backfill [{domain}]: Starting from {fallback.date()}")
    return fallback

def process_domain(domain):
    # 1. Random Start Delay
    start_delay = random.uniform(*START_JITTER)
    time.sleep(start_delay)

    # 2. Get Dynamic Cutoff Date
    cutoff_date = get_smart_cutoff(domain)
    
    client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=5)
    
    # Search descending (Newest first)
    search = arxiv.Search(
        query=f'ti:"{domain}" OR abs:"{domain}"',
        max_results=None,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    session_downloads = 0
    next_break = random.randint(*BREAK_AFTER)
    total_downloads = 0

    try:
        for result in client.results(search):
            # --- THE SMART GUARDRAIL ---
            # If result.published < cutoff_date:  (Stops BEFORE the date)
            # This ensures we re-check papers published on the exact same second/day 
            # as the checkpoint to catch anything missed.
            if result.published < cutoff_date:
                logger.success(f"🛑 Caught up on {domain} (reached {result.published.date()})")
                break
            
            paper_id = result.entry_id.split('/')[-1]
            file_clean = sanitize_filename(result.title)
            filename = file_clean + ".pdf"

            # Check Registry for Duplicates
            # This is now CRITICAL because of the date overlap. 
            # We will encounter files we have already, so we must skip them efficiently.
            try:
                # Optimized: We send filename in body as per your server code
                status_chk = requests.get(
                    f"{REGISTRY_URL}/get_status", 
                    json={"filename": filename},
                    timeout=5
                )
                if status_chk.status_code == 200:
                    status_data = status_chk.json()
                    if status_data.get('status'):
                        # logger.debug(f"⏭️  Already have: {filename}")
                        continue
            except Exception:
                pass # If check fails, we proceed to download to be safe

            # Download
            logger.info(f"⬇️  Downloading [{domain}]: {result.title[:40]}...")
            try:
                result.download_pdf(dirpath=PDF_DIR, filename=filename)
                
                # Update Registry WITH Domain and Date
                requests.post(
                    f"{REGISTRY_URL}/update_status",
                    json={
                        "filename": filename, 
                        "status": "downloaded",
                        "domain": domain,
                        "published_at": result.published.isoformat() 
                    },
                    timeout=10
                )
                
                total_downloads += 1
                session_downloads += 1
                
                # Sleep / Break Logic
                if session_downloads >= next_break:
                    sleep_time = random.uniform(*BREAK_DURATION)
                    logger.info(f"☕ [{domain}] Break for {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
                    session_downloads = 0
                    next_break = random.randint(*BREAK_AFTER)
                else:
                    time.sleep(random.uniform(*DOWNLOAD_SLEEP))

            except Exception as e:
                logger.error(f"❌ Failed {result.title[:20]}: {e}")

    except Exception as e:
        logger.error(f"⚠️ Crash {domain}: {e}")
        
    return f"{domain}: {total_downloads} new"

def main_loop():
    if not os.path.exists(PDF_DIR):
        os.makedirs(PDF_DIR, exist_ok=True)
        
    logger.info(f"🔥 Starting Smart Monitor (Workers: {MAX_WORKERS})")

    while True:
        cycle_start = time.time()
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # We don't pass cutoff_date anymore; process_domain fetches it per thread
            future_to_domain = {executor.submit(process_domain, d): d for d in DOMAINS}
            
            for future in as_completed(future_to_domain):
                try:
                    res = future.result()
                    # logger.info(res) 
                except Exception as e:
                    logger.error(f"Thread Error: {e}")

        elapsed = time.time() - cycle_start
        logger.success(f"💤 Cycle done in {elapsed:.1f}s. Next check in {CHECK_INTERVAL}s")
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        logger.info("🛑 Monitor stopped.")