#!/usr/bin/env python3
"""
Download DSERT-RoLL snow conditions (Heavy_Snow, Light_Snow) for CycleGAN/CycleDiff training.

Usage:
    python download_dsert_snow.py --dry-run     # discover + report, download nothing
    python download_dsert_snow.py               # discover (cached) then download
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ID = "jeongyh98/DSERT-RoLL"
REPO_TYPE = "dataset"
HERE = Path(__file__).resolve().parent
ROOT = HERE
while not (ROOT / "data" / "flir_adas_v2").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
DEST = ROOT / "data" / "dsert_roll"
MANIFEST = HERE / "dsert_files_snow.json"

CONDITIONS = ("Heavy_Snow", "Light_Snow")

MODALITIES = {
    "rectified_THERMAL_L": "thermal",
    "rectified_crop_RGB_L": "rgb",
}

EXPECTED_SEQS = {"Heavy_Snow": 7, "Light_Snow": 7}  # from repo listing


def human(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:,.1f} {unit}"
        n /= 1024.0
    return f"{n:,.1f} PB"


def _net_guard(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except Exception as e:
        print(f"  Network error: {e}", file=sys.stderr)
        time.sleep(2)
        return _net_guard(fn, *a, **kw)


def discover_files(hf_token=None):
    """List files for snow conditions only."""
    from huggingface_hub import HfApi
    
    print("Discovering snow files in DSERT-RoLL...")
    all_files = []
    api = HfApi()
    
    for condition in CONDITIONS:
        # List sequences in this condition
        seqs = list(api.list_repo_tree(REPO_ID, repo_type=REPO_TYPE, path_in_repo=condition, recursive=False))
        print(f"  {condition}: {len(seqs)} sequences")
        
        for seq in seqs:
            seq_name = seq.path.split("/")[-1]
            for modality_hub, modality_key in MODALITIES.items():
                modality_path = f"{condition}/{seq_name}/{modality_hub}"
                print(f"    Scanning {modality_path}...")
                try:
                    files = list(api.list_repo_tree(REPO_ID, repo_type=REPO_TYPE, path_in_repo=modality_path, recursive=True))
                    for item in files:
                        if hasattr(item, 'path') and item.path.lower().endswith((".png", ".jpg")):
                            all_files.append({
                                "condition": condition,
                                "modality": modality_key,
                                "path": item.path,
                                "size": getattr(item, 'size', 0)
                            })
                except Exception as e:
                    print(f"      Warning: failed to list {modality_path}: {e}")
    
    print(f"Discovered {len(all_files)} files")
    return all_files


def download_files(file_list, dest_root, max_workers=8, hf_token=None):
    """Download files with thread pool."""
    from huggingface_hub import hf_hub_download
    
    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    
    def download_one(f):
        rel = f["path"]
        local_path = Path(dest_root) / rel
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        if local_path.exists() and local_path.stat().st_size == f["size"]:
            return f"skip {rel}"
        
        try:
            _net_guard(hf_hub_download, 
                       repo_id=REPO_ID, repo_type=REPO_TYPE,
                       filename=rel, local_dir=str(dest_root),
                       local_dir_use_symlinks=False, token=hf_token)
            return f"downloaded {rel}"
        except Exception as e:
            return f"failed {rel}: {e}"
    
    print(f"Downloading {len(file_list)} files with {max_workers} workers...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_one, f): f for f in file_list}
        for future in as_completed(futures):
            result = future.result()
            print(f"  {result}")


def main():
    ap = argparse.ArgumentParser(description="Download DSERT-RoLL snow data")
    ap.add_argument("--dry-run", action="store_true", help="Discover only, don't download")
    ap.add_argument("--discover-only", action="store_true", help="Same as --dry-run")
    ap.add_argument("--workers", type=int, default=8, help="Download workers")
    ap.add_argument("--token", type=str, help="HF token (or set HF_TOKEN env)")
    args = ap.parse_args()
    
    token = args.token or os.environ.get("HF_TOKEN")
    
    # Discover
    if MANIFEST.exists():
        print(f"Loading cached manifest from {MANIFEST}")
        with open(MANIFEST) as f:
            files = json.load(f)
    else:
        files = discover_files(hf_token=token)
        with open(MANIFEST, "w") as f:
            json.dump(files, f, indent=2)
        print(f"Manifest saved to {MANIFEST}")
    
    # Stats
    by_cond = {}
    total_size = 0
    for f in files:
        cond = f["condition"]
        by_cond[cond] = by_cond.get(cond, 0) + 1
        total_size += f["size"]
    print(f"Files per condition: {by_cond}")
    print(f"Total size: {human(total_size)}")
    
    if args.dry_run or args.discover_only:
        print("Dry run complete. No files downloaded.")
        return
    
    # Download
    download_files(files, DEST, max_workers=args.workers, hf_token=token)
    print("Snow download complete!")


if __name__ == "__main__":
    main()