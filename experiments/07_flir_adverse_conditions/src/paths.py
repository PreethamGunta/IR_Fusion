"""Path resolution for workstream 07 (FLIR adverse conditions).

Every path is derived from this file's location, so the folder can be moved as
a unit. Generated images are experiment-local and regenerable; they are
gitignored via ``experiments/*/data/`` in the repository ``.gitignore``.
"""

from pathlib import Path

# experiments/07_flir_adverse_conditions/src/paths.py
# parents: [0]=src, [1]=07_flir_adverse_conditions, [2]=experiments, [3]=repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = Path(__file__).resolve().parents[1]

FLIR = REPO_ROOT / "data" / "flir_adas_v2"

# Clear (source) thermal frames and their YOLO labels.
CLEAR_IMG = {
    "train": FLIR / "yolo" / "train_3a" / "images" / "train_clear",
    "val": FLIR / "yolo" / "train_3a" / "images" / "val",
    "test": FLIR / "FLIR_ADAS_v2" / "test" / "clear",
}
CLEAR_LBL = {
    "train": FLIR / "yolo" / "train_3a" / "labels" / "train_clear",
    "val": FLIR / "yolo" / "train_3a" / "labels" / "val",
    "test": FLIR / "yolo" / "thermal" / "labels_vtt_test",
}

DATA = EXP_ROOT / "data"
GENERATED = DATA / "generated"
CLEAR_TEST_TREE = DATA / "clear_test"
CONFIGS = EXP_ROOT / "configs"
RESULTS = EXP_ROOT / "results"
RUNS = EXP_ROOT / "runs"
LOGS = EXP_ROOT / "logs"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

CLASS_NAMES = [
    "person", "bike", "car", "motor", "bus", "train", "truck",
    "light", "hydrant", "sign", "face", "other_vehicle",
    "backpack", "bag", "other",
]


def condition_dir(condition: str, severity: str, split: str | None = None) -> Path:
    """``data/generated/<condition>/<severity>[/<split>]``."""
    base = GENERATED / condition / severity
    return base / split if split else base


def ensure_dirs() -> None:
    for d in (DATA, GENERATED, CONFIGS, RESULTS, RUNS, LOGS):
        d.mkdir(parents=True, exist_ok=True)
