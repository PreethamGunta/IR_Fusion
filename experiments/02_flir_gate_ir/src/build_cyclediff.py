"""Build the CycleDiff-75 datasets: B1 (unfiltered) and G1 (winner filter)."""

from __future__ import annotations

import argparse

from .build_datasets import build_one
from .paths import CLEAR_TEST_IMG, CLEAR_TEST_LBL, CLEAR_TRAIN_IMG, CLEAR_TRAIN_LBL, DATA, FOG


def main() -> None:
    ap = argparse.ArgumentParser(description="Build CycleDiff-75 datasets")
    ap.add_argument("--filter", required=True, help="winning filter name")
    args = ap.parse_args()

    w = args.filter
    fog = FOG["cyclediff75"]
    filt = DATA / "filtered" / w / "cyclediff75"
    for p in (filt / "train", filt / "test"):
        if not p.exists():
            raise FileNotFoundError(f"missing filtered CycleDiff directory: {p}")

    build_one(
        "b1_cyclediff75", CLEAR_TRAIN_IMG, CLEAR_TRAIN_LBL,
        fog["train_img"], fog["train_lbl"],
        CLEAR_TEST_IMG, fog["test_img"], fog["test_lbl"],
    )
    build_one(
        f"g1_{w}_cyclediff75", CLEAR_TRAIN_IMG, CLEAR_TRAIN_LBL,
        filt / "train", fog["train_lbl"],
        CLEAR_TEST_IMG, filt / "test", fog["test_lbl"],
    )


if __name__ == "__main__":
    main()
