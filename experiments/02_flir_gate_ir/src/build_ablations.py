"""Build B0 / B2 / B3 ablation datasets (CycleGAN and CycleDiff-75)."""

from __future__ import annotations

import argparse

from .build_datasets import build_one
from .paths import CLEAR_TEST_IMG, CLEAR_TEST_LBL, CLEAR_TRAIN_IMG, CLEAR_TRAIN_LBL, DATA, FOG


def _exists(*paths) -> bool:
    return all(p.exists() for p in paths)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build B0/B2/B3 ablation datasets")
    ap.add_argument("--filter", required=True, help="winning filter")
    ap.add_argument("--filters", nargs="+", default=None,
                    help="filters to build B3 for (default: winner + gamma_clahe)")
    args = ap.parse_args()

    w = args.filter
    b3_filters = args.filters or [w, "gamma_clahe"]
    cg = FOG["cyclegan"]
    cd = FOG["cyclediff75"]

    # B0: clear only, evaluated on both fog domains.
    build_one("b0", CLEAR_TRAIN_IMG, CLEAR_TRAIN_LBL, None, None,
              CLEAR_TEST_IMG, cg["test_img"], cg["test_lbl"])
    build_one("b0_cyclediff75", CLEAR_TRAIN_IMG, CLEAR_TRAIN_LBL, None, None,
              CLEAR_TEST_IMG, cd["test_img"], cd["test_lbl"])

    # B2: fog only, unfiltered.
    build_one("b2", None, None, cg["train_img"], cg["train_lbl"],
              CLEAR_TEST_IMG, cg["test_img"], cg["test_lbl"])
    build_one("b2_cyclediff75", None, None, cd["train_img"], cd["train_lbl"],
              CLEAR_TEST_IMG, cd["test_img"], cd["test_lbl"])

    # B3: filtered clear + filtered fog (always-on).
    built = []
    for f in b3_filters:
        ct = DATA / "filtered" / f / "clear" / "train"
        cv = DATA / "filtered" / f / "clear" / "val"
        cte = DATA / "filtered" / f / "clear" / "test"
        ft = DATA / "filtered" / f / "cyclegan" / "train"
        fte = DATA / "filtered" / f / "cyclegan" / "test"
        if not _exists(ct, cv, cte, ft, fte):
            print(f"[ablations] skip B3 {f} (missing filtered clear/fog)")
            continue
        build_one(f"b3_{f}", ct, CLEAR_TRAIN_LBL, ft, cg["train_lbl"],
                  cte, fte, cg["test_lbl"])
        built.append(f)

    # B3 for CycleDiff with the winning filter.
    ct = DATA / "filtered" / w / "clear" / "train"
    cte = DATA / "filtered" / w / "clear" / "test"
    ft = DATA / "filtered" / w / "cyclediff75" / "train"
    fte = DATA / "filtered" / w / "cyclediff75" / "test"
    if _exists(ct, cte, ft, fte):
        build_one(f"b3_{w}_cyclediff75", ct, CLEAR_TRAIN_LBL, ft, cd["train_lbl"],
                  cte, fte, cd["test_lbl"])
        built.append(f"{w}_cyclediff75")
    else:
        print(f"[ablations] skip B3 {w} on CycleDiff (missing filtered dirs)")

    print(f"[ablations] built B3 for: {built}")


if __name__ == "__main__":
    main()
