"""
Data download helper.

Downloads MedQuAD and PubMedQA from their public GitHub repositories.
Run once before the main app:

    python download_data.py

Can also be imported and called programmatically.
"""
from __future__ import annotations

import os
import subprocess
import sys
import zipfile
import shutil
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
MEDQUAD_DIR = DATA / "MedQuAD"
PUBMEDQA_DIR = DATA / "PubMedQA"

MEDQUAD_URL = "https://github.com/abachaa/MedQuAD/archive/refs/heads/master.zip"
PUBMEDQA_URL = "https://github.com/pubmedqa/pubmedqa/archive/refs/heads/master.zip"


def _run(cmd: list[str]) -> int:
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def _download_zip(url: str, dest_zip: Path) -> bool:
    """Download a zip via requests or urllib."""
    print(f"  Downloading {url} …")
    try:
        import requests
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        with open(dest_zip, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"  requests failed ({e}); trying urllib …")
    try:
        import urllib.request
        urllib.request.urlretrieve(url, dest_zip)
        return True
    except Exception as e:
        print(f"  urllib failed: {e}")
        return False


def download_medquad(force: bool = False) -> bool:
    if MEDQUAD_DIR.exists() and any(MEDQUAD_DIR.rglob("*.xml")) and not force:
        print(f"[data] MedQuAD already present at {MEDQUAD_DIR} — skipping.")
        return True
    tmp = DATA / "_medquad.zip"
    DATA.mkdir(parents=True, exist_ok=True)
    if not _download_zip(MEDQUAD_URL, tmp):
        return False
    with zipfile.ZipFile(tmp) as z:
        z.extractall(DATA / "_medquad_tmp")
    # The archive has a top-level 'MedQuAD-master/' folder
    extracted = DATA / "_medquad_tmp" / "MedQuAD-master"
    if MEDQUAD_DIR.exists():
        shutil.rmtree(MEDQUAD_DIR)
    shutil.move(str(extracted), str(MEDQUAD_DIR))
    shutil.rmtree(DATA / "_medquad_tmp", ignore_errors=True)
    tmp.unlink(missing_ok=True)
    xml_count = len(list(MEDQUAD_DIR.rglob("*.xml")))
    print(f"[data] MedQuAD downloaded — {xml_count} XML files at {MEDQUAD_DIR}")
    return xml_count > 0


def download_pubmedqa(force: bool = False) -> bool:
    target_json = PUBMEDQA_DIR / "data" / "ori_pqal.json"
    if target_json.exists() and not force:
        print(f"[data] PubMedQA already present at {PUBMEDQA_DIR} — skipping.")
        return True
    tmp = DATA / "_pubmedqa.zip"
    DATA.mkdir(parents=True, exist_ok=True)
    if not _download_zip(PUBMEDQA_URL, tmp):
        return False
    with zipfile.ZipFile(tmp) as z:
        z.extractall(DATA / "_pubmedqa_tmp")
    extracted = DATA / "_pubmedqa_tmp" / "pubmedqa-master"
    if PUBMEDQA_DIR.exists():
        shutil.rmtree(PUBMEDQA_DIR)
    shutil.move(str(extracted), str(PUBMEDQA_DIR))
    shutil.rmtree(DATA / "_pubmedqa_tmp", ignore_errors=True)
    tmp.unlink(missing_ok=True)
    print(f"[data] PubMedQA downloaded at {PUBMEDQA_DIR}")
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download MedQuAD and PubMedQA datasets")
    parser.add_argument("--force", action="store_true", help="Re-download even if data already exists")
    parser.add_argument("--medquad-only", action="store_true")
    parser.add_argument("--pubmedqa-only", action="store_true")
    args = parser.parse_args()

    ok = True
    if not args.pubmedqa_only:
        ok = download_medquad(force=args.force) and ok
    if not args.medquad_only:
        ok = download_pubmedqa(force=args.force) and ok

    if ok:
        print("\n[data] All datasets ready.")
    else:
        print("\n[data] One or more downloads failed — check network connectivity.")
        sys.exit(1)
