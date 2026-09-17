"""Register PDFs in data/raw_pdfs/ with the registry as "downloaded".

Needed after a fresh registry DB, or for PDFs you copied in by hand (the
downloader registers its own). Papers already known are left alone.

    python scripts/register_pdfs.py                  # every PDF
    python scripts/register_pdfs.py --limit 3        # first 3 (alphabetical) — smoke tests
    python scripts/register_pdfs.py A.pdf B.pdf      # specific files
    python scripts/register_pdfs.py --retry-errors   # papers in status "error" -> "downloaded" (re-run from scratch)
    python scripts/register_pdfs.py --force A.pdf    # re-run one paper from scratch whatever its status

Requires the registry service (port 4000) to be running.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.paths import PDF_DIR
from common.registry_client import RegistryClient, RegistryUnavailable


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="PDF filenames (default: all in data/raw_pdfs)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--retry-errors", action="store_true", help="reset every paper in status 'error' to 'downloaded'")
    ap.add_argument("--force", action="store_true", help="reset the given papers to 'downloaded' regardless of status")
    args = ap.parse_args()

    registry = RegistryClient()
    try:
        if args.retry_errors:
            errors = registry.list_papers(status="error")
            n = sum(registry.set_status(p["filename"], "downloaded", force=True) for p in errors)
            print(f"reset {n} of {len(errors)} errored papers to 'downloaded'")
            if not args.files and args.limit is None:
                return

        pdfs = [PDF_DIR / Path(f).name for f in args.files] if args.files else sorted(PDF_DIR.glob("*.pdf"))
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
            if not args.force and registry.get_status(pdf.stem):
                skipped += 1
                continue
            if registry.set_status(pdf.stem, "downloaded", force=args.force):
                registered += 1
            else:
                print(f"  rejected: {pdf.name}")
                failed += 1
        print(f"registered {registered}, already known {skipped}, failed {failed} (of {len(pdfs)} PDFs)")
        sys.exit(1 if failed else 0)
    except RegistryUnavailable as e:
        sys.exit(f"{e} — start it first (cd registry_manager && python main.py)")


if __name__ == "__main__":
    main()
