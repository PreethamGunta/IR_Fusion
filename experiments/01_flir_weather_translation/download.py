#!/usr/bin/env python3
"""
Download FLIR ADAS v2 from Kaggle.

Dataset: samdazel/teledyne-flir-adas-thermal-dataset-v2
URL: https://www.kaggle.com/datasets/samdazel/teledyne-flir-adas-thermal-dataset-v2/data

Prerequisites:
- Kaggle account with API token at ~/.kaggle/access_token
  (generate at https://www.kaggle.com/settings/api)
- OR KAGGLE_API_TOKEN environment variable
- OR run `kaggle auth login` for OAuth

Usage:
    python flir_adras_v2/download.py
    python flir_adras_v2/download.py --dest /path/to/data
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    print("kaggle package not installed. Install with: pip install kaggle")
    sys.exit(1)

DATASET = "samdazel/teledyne-flir-adas-thermal-dataset-v2"
DEFAULT_DEST = Path(__file__).resolve().parent.parent / "data" / "FLIR_ADAS_v2"


def main():
    ap = argparse.ArgumentParser(description="Download FLIR ADAS v2 from Kaggle")
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                    help="Download destination (default: data/FLIR_ADAS_v2)")
    ap.add_argument("--unzip", action="store_true", default=True,
                    help="Extract zip after download (default: True)")
    ap.add_argument("--no-unzip", dest="unzip", action="store_false",
                    help="Skip extraction")
    args = ap.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading {DATASET} to {args.dest}")
    api.dataset_download_files(DATASET, path=str(args.dest), unzip=args.unzip)
    print(f"Download complete. Files at: {args.dest}")

    if args.unzip:
        zip_files = list(args.dest.glob("*.zip"))
        if zip_files:
            print(f"Found {len(zip_files)} zip files. Extracting...")
            for zf in zip_files:
                with zipfile.ZipFile(zf, 'r') as z:
                    z.extractall(args.dest)
                print(f"  Extracted: {zf.name}")
                zf.unlink()
            print("Extraction complete.")


if __name__ == "__main__":
    main()
