#!/usr/bin/env python3
"""Download Light_Snow from DSERT-RoLL dataset."""
import sys
from pathlib import Path
from huggingface_hub import list_repo_tree, hf_hub_download

REPO_ID = "jeongyh98/DSERT-RoLL"
REPO_TYPE = "dataset"
DEST = Path("data/DSERT-RoLL")
CONDITION = "Light_Snow"

print(f"Discovering {CONDITION}...", flush=True)
files = list(list_repo_tree(REPO_ID, repo_type=REPO_TYPE, path_in_repo=CONDITION, recursive=True))

to_download = []
for item in files:
    if hasattr(item, "path") and item.path.lower().endswith((".png", ".jpg", ".npy", ".npz", ".pkl", ".csv", ".bin")):
        to_download.append(item.path)

print(f"Found {len(to_download)} files", flush=True)

downloaded = 0
skipped = 0
for i, rel in enumerate(to_download):
    local_path = DEST / rel
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists():
        skipped += 1
        continue
    try:
        hf_hub_download(
            repo_id=REPO_ID, repo_type=REPO_TYPE,
            filename=rel, local_dir=str(DEST),
        )
        downloaded += 1
    except Exception as e:
        print(f"FAILED {rel}: {e}", flush=True)
    if (i + 1) % 100 == 0:
        print(f"  Progress: {i+1}/{len(to_download)} (down={downloaded}, skip={skipped})", flush=True)

print(f"Done. Downloaded: {downloaded}, Skipped: {skipped}", flush=True)
