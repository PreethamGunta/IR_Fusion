"""Path resolution for experiment 08 (Graph-Structured KD).

Every path is derived from this file's location. Nothing is hardcoded.
"""
from pathlib import Path

# experiments/08_graph_kd/src/graphkd/paths.py
# parents: 0=graphkd, 1=src, 2=08_graph_kd, 3=experiments, 4=release root
REPO_ROOT = Path(__file__).resolve().parents[4]
EXP_ROOT = Path(__file__).resolve().parents[2]

DATA = EXP_ROOT / "data"
RUNS = EXP_ROOT / "runs"
RESULTS = EXP_ROOT / "results"
LOGS = EXP_ROOT / "logs"
CONFIGS = EXP_ROOT / "configs"
SCRIPTS = EXP_ROOT / "scripts"
CACHE = DATA / "cache"

# ---- aligned FLIR dataset (Zhang et al. aligned pairs) ----------------------
ALIGNED = REPO_ROOT / "data" / "flir_aligned"
RAW = ALIGNED / "raw" / "align"
YOLO = ALIGNED / "yolo"
FIXED_RGB = ALIGNED / "rgb_fixed"
PAIR_MANIFEST = ALIGNED / "pair_manifest.json"

PRETRAINED = REPO_ROOT / "models" / "pretrained"

# ---- existing release artifacts useful as references -----------------------
GATEIR = REPO_ROOT / "experiments" / "02_flir_gate_ir"

CLASS_NAMES = ["person", "bicycle", "car", "dog"]


def ensure_dirs() -> None:
    for d in (DATA, RUNS, RESULTS, LOGS, CONFIGS, CACHE):
        d.mkdir(parents=True, exist_ok=True)
