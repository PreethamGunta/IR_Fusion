"""Provenance helpers shared by every M3FD train/eval/generate entry point.

Why this module exists
----------------------
An earlier version of `train_m3fd.py` and `eval_m3fd.py` each tried to find
`split_manifest.json` with the same two-element candidate list:

    for candidate in [data_path.parent / "split_manifest.json",
                      Path(args.data).parent / "split_manifest.json"]:

Both entries are byte-identical -- it was a no-op dressed up as a fallback,
not a fallback.  `split_manifest.json` only exists in `m3fd_experiments/data/split/`,
so every synthetic-weather YAML, the 3B YAML and the CycleGAN YAMLs recorded
`split_hash: null` and no downstream result could be traced back to a split.

`resolve_split_hash()` below walks *up* from the YAML's `path:` key, which is
the only place the real dataset root is recorded (YOLO resolves `train:`/`val:`/
`test:` relative to it), so it finds the manifest for both the clear split and
every generated derivative.

Everything else here is deliberately cheap and never raises: a result record
with one field missing is still useful, a crashed evaluation is not.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SPLIT_MANIFEST_NAME = "split_manifest.json"
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# ── dataset YAML introspection ────────────────────────────────────────────────

def read_yaml_dataset(yaml_path: Path) -> dict:
    """Parse a YOLO dataset YAML, tolerating both `names: {0: X}` and list form.

    Returns {} on any failure -- callers use this for provenance only and must
    not be made to fail by a malformed sidecar.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        return {}
    try:
        import yaml  # ultralytics depends on it, so it is always present
        data = yaml.safe_load(yaml_path.read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def dataset_root(yaml_path: Path) -> Path | None:
    """Absolute directory the YAML's train/val/test entries resolve against."""
    data = read_yaml_dataset(yaml_path)
    raw = data.get("path")
    if not raw:
        return None
    p = Path(str(raw))
    return p if p.is_absolute() else (Path(yaml_path).resolve().parent / p)


def dataset_entries(yaml_path: Path, split: str) -> list[Path]:
    """Resolve one split's image directories (YAML values may be lists)."""
    data = read_yaml_dataset(yaml_path)
    raw = data.get(split)
    if not raw:
        return []
    values = raw if isinstance(raw, list) else [raw]
    root = dataset_root(yaml_path)
    out = []
    for v in values:
        p = Path(str(v))
        out.append(p if p.is_absolute() else (root / p if root else p))
    return out


def count_images(yaml_path: Path, split: str) -> int:
    """Number of images actually visible to Ultralytics for `split`.

    Counts by extension on purpose: the M3FD split dirs carry sibling `.npy`
    disk-cache files next to the PNGs, and a bare `iterdir()` would double-count
    them (the same class of bug that made `os.listdir` iterate 648 val entries
    for 324 frames).
    """
    total = 0
    for d in dataset_entries(yaml_path, split):
        if not d.is_dir():
            continue
        for ext in IMG_EXTS:
            total += sum(1 for _ in d.glob(f"*{ext}"))
    return total


# ── split hash ───────────────────────────────────────────────────────────────

def split_hash_for_dataset(yaml_path: Path, max_up: int = 8) -> str | None:
    """Walk up from the dataset root looking for split_manifest.json.

    Covers both layouts:
        data/split/m3fd_clear.yaml          -> data/split/split_manifest.json
        data/synthetic/fog_heavy/m3fd_*.yaml-> .../split_manifest.json (copied
                                                 next to the YAML by the
                                                 generator) or the split root
    Returns None if no manifest is reachable; callers decide whether that is
    fatal (it is, for the primary split) or merely a provenance gap.
    """
    yaml_path = Path(yaml_path)
    seen = set()

    def _from(start: Path):
        for d in [start, *start.parents][: max_up + 1]:
            man = d / SPLIT_MANIFEST_NAME
            if man in seen:
                continue
            seen.add(man)
            if man.is_file():
                try:
                    h = json.loads(man.read_text()).get("split_hash")
                except Exception:
                    continue
                if h:
                    return h
        return None

    found = _from(yaml_path.resolve().parent)
    if found:
        return found
    root = dataset_root(yaml_path)
    if root is not None:
        return _from(root.resolve())
    return None


def stamp_split(yaml_path: Path, manifest_src: Path | None = None) -> str | None:
    """Copy the active split manifest next to a generated dataset YAML.

    Generated datasets (synthetic weather, 3B, CycleGAN) are what downstream
    results are computed on, so each one carries the manifest that produced it.
    Without this every derived artifact reports `split_hash: null` and a
    reviewer cannot tell whether two runs used the same frames.
    """
    manifest_src = manifest_src or (HERE / "data" / "split" / SPLIT_MANIFEST_NAME)
    manifest_src = Path(manifest_src)
    if not manifest_src.is_file():
        return None
    yaml_path = Path(yaml_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    dest = yaml_path.parent / SPLIT_MANIFEST_NAME
    if dest.resolve() == manifest_src.resolve():
        return split_hash_for_dataset(yaml_path)
    dest.write_text(manifest_src.read_text())
    return split_hash_for_dataset(yaml_path)


# ── environment / artifact fingerprints ──────────────────────────────────────

def _git(repo: Path, *args: str) -> tuple[int, str]:
    try:
        out = subprocess.run(["git", "-C", str(repo), *args],
                             capture_output=True, text=True, timeout=20)
        return out.returncode, out.stdout
    except Exception:
        return 1, ""


def git_commit(repo: Path | None = None) -> str | None:
    """Short SHA of HEAD, or None outside a git checkout."""
    repo = repo or REPO
    rc, stdout = _git(repo, "rev-parse", "--short", "HEAD")
    if rc == 0:
        return stdout.strip() or None
    return None


def git_dirty(repo: Path | None = None) -> bool | None:
    """True when the working tree has uncommitted changes.

    `git_commit()` alone cannot tell "ran the committed code" from "ran code that
    was still being edited". Every dataset generated in this repo during
    development was produced from an uncommitted working tree, and the recorded
    SHA named a commit whose code had never touched those pixels. Paired with the
    SHA, this flag makes that state visible instead of silently wrong.
    """
    repo = repo or REPO
    rc, stdout = _git(repo, "status", "--porcelain")
    if rc != 0:
        return None
    return bool(stdout.strip())


def git_diff_hash(repo: Path | None = None) -> str | None:
    """SHA-256 of the tracked diff, so a dirty run names its own code state.

    16 hex chars like the other fingerprints here: this identifies a working
    tree, it is not a security boundary. Returns None for a clean tree or outside
    a checkout, so `git_dirty: false` plus no hash is an unambiguous "this ran
    committed code".
    """
    repo = repo or REPO
    rc, stdout = _git(repo, "diff", "HEAD")
    if rc != 0 or not stdout.strip():
        return None
    return hashlib.sha256(stdout.encode("utf-8", "replace")).hexdigest()[:16]


def sha256_file(path, chunk: int = 1 << 20) -> str | None:
    """sha256 of a file, or None if it does not exist / cannot be read.

    16 hex chars, matching the width of `split_hash` -- this identifies a
    checkpoint, it is not a security boundary.
    """
    path = Path(path)
    if not path.is_file():
        return None
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(chunk), b""):
                h.update(block)
    except OSError:
        return None
    return h.hexdigest()[:16]


def slurm_job_id() -> str | None:
    return os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID") or None


def run_record(**fields) -> dict:
    """Standard provenance block appended to every result JSON.

    Keys with no value are dropped rather than written as null, so a missing
    field is visibly absent instead of looking like a measured zero.

    `git_dirty` is a bool and is kept even when False, because "this ran
    committed code" is a fact worth recording; only None is dropped.
    """
    rec = {
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "git_diff_hash": git_diff_hash(),
        "slurm_job_id": slurm_job_id(),
        "hostname": os.environ.get("SLURMD_NODENAME") or os.uname().nodename,
    }
    rec.update(fields)
    return {k: v for k, v in rec.items() if v is not None}