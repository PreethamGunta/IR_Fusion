"""Path resolution for the GATE-IR experiment.

ALL paths are derived from this file's location. Nothing here hardcodes an
absolute path, so the whole folder can be moved as a unit.
"""

from pathlib import Path

# experiments/02_flir_gate_ir/src/paths.py
# parents: [0]=src, [1]=02_flir_gate_ir, [2]=experiments, [3]=release root
REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = Path(__file__).resolve().parents[1]

FLIR_DATA = REPO_ROOT / "data" / "flir_adas_v2"
FLIR_CODE = REPO_ROOT / "experiments" / "01_flir_weather_translation"
PRETRAINED = REPO_ROOT / "models" / "pretrained"

DATA = EXP_ROOT / "data"
RUNS = EXP_ROOT / "runs"
RESULTS = EXP_ROOT / "results"
LOGS = EXP_ROOT / "logs"
CONFIGS = EXP_ROOT / "configs"

# ---- clear (FLIR) -----------------------------------------------------------
CLEAR_TRAIN_IMG = FLIR_DATA / "yolo" / "train_3a" / "images" / "train_clear"
CLEAR_TRAIN_LBL = FLIR_DATA / "yolo" / "train_3a" / "labels" / "train_clear"
CLEAR_VAL_IMG = FLIR_DATA / "yolo" / "train_3a" / "images" / "val"
CLEAR_VAL_LBL = FLIR_DATA / "yolo" / "train_3a" / "labels" / "val"
CLEAR_TEST_IMG = FLIR_DATA / "FLIR_ADAS_v2" / "test" / "clear"
CLEAR_TEST_LBL = FLIR_DATA / "yolo" / "thermal" / "labels_vtt_test"

# ---- synthetic fog ----------------------------------------------------------
# CycleGAN fog: the training images that actually have labels live in train_3a
# (8,189 of the 10,742 translated frames). Use that exact set so the comparison
# against the existing 3a_cyclegan_fog-3 run is like-for-like.
FOG = {
    "cyclegan": {
        "train_img": FLIR_DATA / "yolo" / "train_3a" / "images" / "train_adv",
        "train_lbl": FLIR_DATA / "yolo" / "train_3a" / "labels" / "train_adv",
        "test_img": FLIR_DATA / "synthetic" / "cyclegan" / "fog" / "images" / "test",
        "test_lbl": FLIR_DATA / "synthetic" / "cyclegan" / "fog" / "labels" / "test",
    },
    "cyclediff75": {
        "train_img": FLIR_DATA / "synthetic" / "cyclediff_75fog" / "fog" / "images" / "train",
        "train_lbl": FLIR_DATA / "synthetic" / "cyclediff_75fog" / "fog" / "labels" / "train",
        "test_img": FLIR_DATA / "synthetic" / "cyclediff_75fog" / "fog" / "images" / "test",
        "test_lbl": FLIR_DATA / "synthetic" / "cyclediff_75fog" / "fog" / "labels" / "test",
    },
}

CLASS_NAMES = [
    "person", "bike", "car", "motor", "bus", "train", "truck",
    "light", "hydrant", "sign", "face", "other_vehicle",
    "backpack", "bag", "other",
]

# Existing detector runs (reused as baselines).
BASELINE_WEIGHTS = {
    "b1_cyclegan": FLIR_CODE / "runs" / "3a_cyclegan_fog-3" / "weights" / "best.pt",
    "b1_cyclediff75": FLIR_CODE / "runs" / "3a_cyclediff_fog-8" / "weights" / "best.pt",
}


def ensure_dirs() -> None:
    for d in (DATA, RUNS, RESULTS, LOGS, CONFIGS):
        d.mkdir(parents=True, exist_ok=True)
