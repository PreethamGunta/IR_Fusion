"""Build one G1 (gate+filter) dataset per filter, plus the unfiltered B1 baseline."""

from __future__ import annotations

import argparse

from .build_datasets import build_one
from .paths import CLEAR_TEST_IMG, CLEAR_TEST_LBL, CLEAR_TRAIN_IMG, CLEAR_TRAIN_LBL, DATA, FOG

DEFAULT_FILTERS = ["gamma", "clahe", "dcp", "unet", "gamma_clahe"]


def build_filter(filter_name: str, condition: str = "cyclegan") -> None:
    fog = FOG[condition]
    filt = DATA / "filtered" / filter_name
    f_fog_train = filt / condition / "train"
    f_fog_test = filt / condition / "test"
    for p in (f_fog_train, f_fog_test):
        if not p.exists():
            raise FileNotFoundError(f"filtered directory missing: {p}")
    build_one(
        f"g1_{filter_name}",
        CLEAR_TRAIN_IMG, CLEAR_TRAIN_LBL,
        f_fog_train, fog["train_lbl"],
        CLEAR_TEST_IMG, f_fog_test, fog["test_lbl"],
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Build G1 datasets for the filter screen")
    ap.add_argument("--condition", default="cyclegan", choices=list(FOG))
    ap.add_argument("--filters", nargs="+", default=DEFAULT_FILTERS)
    args = ap.parse_args()

    fog = FOG[args.condition]
    # Unfiltered B1 reference.
    build_one(
        "b1", CLEAR_TRAIN_IMG, CLEAR_TRAIN_LBL,
        fog["train_img"], fog["train_lbl"],
        CLEAR_TEST_IMG, fog["test_img"], fog["test_lbl"],
    )
    for f in args.filters:
        print(f"[screen] building g1_{f}")
        build_filter(f, args.condition)


if __name__ == "__main__":
    main()
