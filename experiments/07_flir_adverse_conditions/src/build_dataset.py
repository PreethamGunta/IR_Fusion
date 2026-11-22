"""Build YOLO dataset YAMLs for one generated condition.

Naming follows workstream 02's B* convention:

* ``b0_<cond>_<sev>_train``  clear-only, test on the condition
* ``b1_<cond>_<sev>_train``  clear + adverse, test on the condition
* ``b2_<cond>_<sev>_train``  adverse-only, test on the condition
* ``b1_<cond>_<sev>_eval_clear`` / ``..._eval_<cond>``  matched evaluations

All entries are absolute because Ultralytics resolves them that way; the code
derives them from ``src/paths.py`` and never hardcodes a machine path.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from . import paths


def _link(src, dst) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    dst.symlink_to(os.path.relpath(Path(src), dst.parent))


def ensure_clear_test_tree() -> str:
    """Shared symlink tree so the clear test split has adjacent labels."""
    img_dst = paths.CLEAR_TEST_TREE / "images" / "test"
    lbl_dst = paths.CLEAR_TEST_TREE / "labels" / "test"
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)
    for img in paths.CLEAR_IMG["test"].iterdir():
        if img.suffix.lower() not in paths.IMG_EXTS:
            continue
        _link(img, img_dst / img.name)
        lbl = paths.CLEAR_LBL["test"] / f"{img.stem}.txt"
        if lbl.exists():
            _link(lbl, lbl_dst / f"{img.stem}.txt")
    return str(img_dst)


def _write(name: str, train: list[str], val: str, test: str) -> str:
    path = paths.CONFIGS / f"{name}.yaml"
    content = {
        "path": str(paths.REPO_ROOT),
        "train": train,
        "val": val,
        "test": test,
        "nc": len(paths.CLASS_NAMES),
        "names": list(paths.CLASS_NAMES),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(content, sort_keys=False))
    print(f"[build_dataset] {path}")
    return str(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build dataset YAMLs for a generated condition")
    ap.add_argument("--condition", required=True,
                    help="condition name, or 'clear' for the clear-only baseline YAMLs")
    ap.add_argument("--severity", default="medium")
    args = ap.parse_args()

    paths.ensure_dirs()
    cond, sev = args.condition, args.severity
    clear_train = str(paths.CLEAR_IMG["train"])
    clear_val = str(paths.CLEAR_IMG["val"])
    clear_test = ensure_clear_test_tree()

    if cond == "clear":
        _write("b0_clear_train", [clear_train], clear_val, clear_test)
        _write("b0_clear_eval_clear", [clear_train], clear_val, clear_test)
        return

    gen_test = paths.condition_dir(cond, sev, "test") / "images"
    gen_train = paths.condition_dir(cond, sev, "train") / "images"
    if not gen_test.is_dir():
        raise SystemExit(f"missing generated test split: {gen_test} (run src.generate first)")

    has_train = gen_train.is_dir() and any(gen_train.iterdir())
    if not has_train:
        print(f"[build_dataset] WARNING: no train split at {gen_train}; "
              f"writing eval YAMLs only")

    written = []
    written.append(_write(f"b0_{cond}_{sev}_train", [clear_train], clear_val, str(gen_test)))
    if has_train:
        b1_train = [clear_train, str(gen_train)]
        written.append(_write(f"b1_{cond}_{sev}_train", b1_train, clear_val, str(gen_test)))
        written.append(_write(f"b2_{cond}_{sev}_train", [str(gen_train)], clear_val, str(gen_test)))
        written.append(_write(f"b1_{cond}_{sev}_eval_clear", b1_train, clear_val, clear_test))
        written.append(_write(f"b1_{cond}_{sev}_eval_{cond}", b1_train, clear_val, str(gen_test)))
        written.append(_write(f"b2_{cond}_{sev}_eval_clear", [str(gen_train)], clear_val, clear_test))
        written.append(_write(f"b2_{cond}_{sev}_eval_{cond}", [str(gen_train)], clear_val, str(gen_test)))

    manifest = {
        "condition": cond,
        "severity": sev,
        "generated_test": str(gen_test),
        "generated_train": str(gen_train) if has_train else None,
        "clear_test": clear_test,
        "yaml": written,
    }
    out = paths.CONFIGS / f"manifest_{cond}_{sev}.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"[build_dataset] -> {out}")


if __name__ == "__main__":
    main()
