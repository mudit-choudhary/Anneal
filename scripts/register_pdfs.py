"""Register PDFs in data/raw_pdfs/ with the registry as "downloaded".

Needed after a fresh registry DB, or for PDFs you copied in by hand (the
downloader registers its own). Papers already known to the registry are left
untouched (the registry ignores duplicate "downloaded" inserts).

    python scripts/register_pdfs.py             # every PDF
    python scripts/register_pdfs.py --limit 3   # first 3 (alphabetical) — smoke tests
    python scripts/register_pdfs.py A.pdf B.pdf # specific files

Requires the registry service (port 4000) to be running.
"""

import argparse
import sys
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = REPO_ROOT / "data" / "raw_pdfs"
REGISTRY_URL = "http://127.0.0.1:4000"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="PDF filenames (default: all in data/raw_pdfs)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.files:
        pdfs = [PDF_DIR / Path(f).name for f in args.files]
    else:
        pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if args.limit:
        pdfs = pdfs[:args.limit]
    if not pdfs:
        sys.exit(f"No PDFs found in {PDF_DIR}")

    registered = skipped = failed = 0
    for pdf in pdfs:
        if not pdf.exists():
            print(f"  missing: {pdf.name}")
            failed += 1
            continue
        try:
            status = requests.get(f"{REGISTRY_URL}/get_status",
                                  json={"filename": pdf.stem}, timeout=10).json().get("status")
            if status:
                skipped += 1
                continue
            ok = requests.post(f"{REGISTRY_URL}/update_status",
                               json={"filename": pdf.stem, "status": "downloaded"},
                               timeout=10).json().get("success")
        except requests.RequestException as e:
            sys.exit(f"Registry unreachable at {REGISTRY_URL} — start it first "
                     f"(cd registry_manager && python main.py). ({e})")
        if ok:
            registered += 1
        else:
            print(f"  rejected: {pdf.name}")
            failed += 1

    print(f"registered {registered}, already known {skipped}, failed {failed} "
          f"(of {len(pdfs)} PDFs)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
