"""Download only the DSERT-RoLL slice Method 3 needs.

The full jeongyh98/DSERT-RoLL repo is 286 GB: 190 sequences x 4k+ frames across
stereo RGB, stereo event, stereo thermal, two LiDARs and radar, in Clear, Fog,
Heavy_Rain, Heavy_Snow, Light_Rain and Light_Snow.  Method 3 needs none of that.

A CycleGAN target domain only needs *images*.  We train a generator on DSERT
and then apply it to M3FD, so we never evaluate on DSERT and need no labels
either -- label.pkl is 100 KB/sequence of 3D boxes we would never read.

What we actually pull is the left thermal and left RGB of the three weather
conditions we are synthesising: ~4 GB instead of 286 GB.

    python download_dsert.py --dry-run     # discover + report, download nothing
    python download_dsert.py               # discover (cached) then download
    python download_dsert.py --discover-only

Why this does not just call snapshot_download(allow_patterns=...)
------------------------------------------------------------------
That path is the obvious one and it hangs.  allow_patterns still makes
huggingface_hub walk the repository root recursively, and the root holds on
the order of a million files, so the listing phase alone ran past two minutes
with no output.  Measuring list_repo_tree directly shows the shape of the
problem: a *targeted* non-recursive listing of one condition takes 0.3 s, but
the recursive walk is what dies.

So this discovers by walking only the branches we want -- 3 conditions x N
sequences x 2 directories -- caches the result in dsert_files.json, and then
fetches with a thread pool.  Discovery is ~40 s and happens once; the download
itself is resumable because hf_hub_download keeps its own cache and skips
files already present.

Note the GitHub README describes rectified_THERMAL_L as .npz, but the files
actually published on the Hub are .png.  Discovery records the real extension
rather than assuming one, and inspect_dsert.py reports what it got.
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
DEST = HERE.parents[2] / "data" / "dsert_roll"
MANIFEST = HERE / "dsert_files.json"

# Snow is deliberately absent: the GATE-IR Stage B enhancer has a fog branch and
# a rain branch and no snow branch, so a snow generator has nothing to feed.
CONDITIONS = ("Fog", "Heavy_Rain", "Light_Rain")

# Directory name on the Hub -> short key used in run names and manifests.
MODALITIES = {
    "rectified_THERMAL_L": "thermal",
    "rectified_crop_RGB_L": "rgb",
}
MODALITY_EXTS = {"thermal": (".png", ".npz", ".npy", ".jpg"),
                 "rgb": (".jpg", ".png")}

# Expected frame counts from the dataset's own README, used only to warn about
# a truncated or wrong-sliced download. Not used to decide anything.
EXPECTED_SEQS = {"Fog": 32, "Heavy_Rain": 18, "Light_Rain": 17}


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
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(
            f"\nERROR: HuggingFace Hub unreachable ({type(exc).__name__}: {exc})\n"
            f"\nThis cluster's compute nodes may have no outbound route. Run this\n"
            f"script from the login node instead -- a job that tries to download\n"
            f"on-node will otherwise hang until the allocation times out."
        ) from exc


def discover(api, conditions, modalities):
    """Walk only the branches we want. Returns list of dicts: path, size."""
    from huggingface_hub.hf_api import RepoFile, RepoFolder

    files, seq_counts = [], {}

    for cond in conditions:
        seqs = _net_guard(list, api.list_repo_tree(
            REPO_ID, repo_type=REPO_TYPE, path_in_repo=cond, recursive=False))
        seq_dirs = [e.path for e in seqs if isinstance(e, RepoFolder)]
        seq_counts[cond] = len(seq_dirs)

        for seq in seq_dirs:
            for mod in modalities:
                short = MODALITIES[mod]
                try:
                    entries = _net_guard(list, api.list_repo_tree(
                        REPO_ID, repo_type=REPO_TYPE,
                        path_in_repo=f"{seq}/{mod}", recursive=False))
                except SystemExit:
                    raise
                except Exception:
                    # one missing directory should not kill a 40 s discovery
                    continue
                exts = MODALITY_EXTS[short]
                for e in entries:
                    if not isinstance(e, RepoFile):
                        continue
                    if os.path.splitext(e.path)[1].lower() not in exts:
                        continue
                    files.append({"path": e.path, "size": getattr(e, "size", 0) or 0,
                                  "condition": cond, "modality": short,
                                  "sequence": seq.split("/")[-1],
                                  "frame": os.path.splitext(os.path.basename(e.path))[0]})
        print(f"  {cond:12s} {len(seq_dirs):3d} sequences "
              f"({', '.join(m for m in modalities)})", flush=True)

    return files, seq_counts


def report(files, seq_counts, conditions):
    by = {}
    for f in files:
        by.setdefault((f["condition"], f["modality"]), [0, 0])
        by[(f["condition"], f["modality"])][0] += 1
        by[(f["condition"], f["modality"])][1] += f["size"]

    total_n = sum(f["size"] for f in files)
    print(f"\n{'condition':12s} {'modality':9s} {'files':>8s} {'size':>12s}")
    print("-" * 44)
    for (cond, mod), (n, sz) in sorted(by.items()):
        print(f"{cond:12s} {mod:9s} {n:8,d} {human(sz):>12s}")
    print("-" * 44)
    print(f"{'TOTAL':22s} {len(files):8,d} {human(total_n):>12s}")
    print(f"\nfor reference the full repo is ~286 GB / 190 sequences / 6 conditions")

    for cond in conditions:
        exp = EXPECTED_SEQS.get(cond)
        got = seq_counts.get(cond, 0)
        if exp and got != exp:
            print(f"WARNING: {cond} has {got} sequences upstream, "
                  f"README says {exp}. The Hub may have been re-cut.")
    return total_n


def download(files, dest, workers):
    from huggingface_hub import hf_hub_download

    dest.mkdir(parents=True, exist_ok=True)
    done, failed = 0, []
    t0 = time.time()
    print(f"\ndownloading {len(files):,d} files with {workers} workers ...")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(hf_hub_download, REPO_ID, f["path"],
                          repo_type=REPO_TYPE, local_dir=str(dest)): f
                for f in files}
        for fut in as_completed(futs):
            f = futs[fut]
            try:
                fut.result()
            except Exception as exc:
                failed.append((f["path"], f"{type(exc).__name__}: {exc}"))
            done += 1
            if done % 1000 == 0 or done == len(files):
                el = time.time() - t0
                rate = done / el if el else 0
                left = (len(files) - done) / rate if rate else 0
                print(f"  {done:6,d}/{len(files):,d}  {rate:6.1f} f/s  "
                      f"eta {left/60:5.1f} min", flush=True)
    if failed:
        print(f"\n{len(failed)} files FAILED:")
        for p, e in failed[:10]:
            print(f"  {p}\n    {e}")
        return 1
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, default=DEST)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS),
                    choices=list(CONDITIONS))
    ap.add_argument("--thermal-only", action="store_true")
    ap.add_argument("--rgb-only", action="store_true")
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--dry-run", action="store_true",
                    help="discover and report, download nothing")
    ap.add_argument("--discover-only", action="store_true")
    ap.add_argument("--reuse-manifest", action="store_true",
                    help="skip discovery and use the cached dsert_files.json")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    if args.thermal_only and args.rgb_only:
        ap.error("--thermal-only and --rgb-only are mutually exclusive")
    modalities = list(MODALITIES)
    if args.thermal_only:
        modalities = ["rectified_THERMAL_L"]
    elif args.rgb_only:
        modalities = ["rectified_crop_RGB_L"]

    print(f"repo       : {args.repo_id}")
    print(f"dest       : {args.dest}")
    print(f"conditions : {' '.join(args.conditions)}")
    print(f"modalities : {' '.join(MODALITIES[m] for m in modalities)}\n")

    if args.reuse_manifest and MANIFEST.exists():
        blob = json.loads(MANIFEST.read_text())
        files, seq_counts = blob["files"], blob["seq_counts"]
        print(f"reusing cached manifest ({len(files):,d} files)\n")
    else:
        from huggingface_hub import HfApi
        print("discovering files (targeted walk, ~40 s) ...")
        t0 = time.time()
        files, seq_counts = discover(HfApi(), args.conditions, modalities)
        print(f"discovery took {time.time() - t0:.1f}s")
        MANIFEST.write_text(json.dumps(
            {"repo_id": REPO_ID, "conditions": args.conditions,
             "modalities": [MODALITIES[m] for m in modalities],
             "seq_counts": seq_counts, "files": files}, indent=1))
        print(f"manifest -> {MANIFEST}")

    total = report(files, seq_counts, args.conditions)

    if args.dry_run or args.discover_only:
        print("\ndry run -- nothing downloaded.")
        return 0
    if not files:
        print("\nERROR: no files discovered. Check --conditions.")
        return 1
    print()
    return download(files, args.dest, args.workers) if total else 0


if __name__ == "__main__":
    sys.exit(main())
