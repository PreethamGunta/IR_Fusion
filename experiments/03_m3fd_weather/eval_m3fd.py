#!/usr/bin/env python3
"""Evaluate a trained YOLO model on a specific dataset YAML.

Writes a JSON with mAP50, mAP50-95, precision, recall and a provenance block
(weights path + sha256, dataset path, split, split hash, image count, git
commit, SLURM job id).

Provenance is not decoration here. The previous version resolved split_hash
from a two-element candidate list whose entries were byte-identical -- a no-op
dressed up as a fallback -- so every synthetic-weather, 3B and CycleGAN result
recorded `null` and could not be traced back to a split. It also never passed
`imgsz`, relying on the 640 default matching training by coincidence, never
recorded how many images were actually scored, and never checked the test set
was non-empty. An eval that scores zero images is not an error to Ultralytics;
it is a result. So all four are now explicit here.
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"

import sys
sys.path.insert(0, str(HERE))
import provenance

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="test", choices=("val", "test"))
    ap.add_argument("--imgsz", type=int, default=640,
                    help="must match training. Left to Ultralytics it defaults "
                         "to 640, which only matches by coincidence.")
    ap.add_argument("--out", default=None,
                    help="Output JSON path. Default: <experiment>/results_<name>.json")
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Dataset YAML not found: {data_path}")

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"Weights not found: {weights}")

    split_hash = provenance.split_hash_for_dataset(data_path)
    if split_hash is None:
        print(f"WARNING: no split_manifest.json reachable from {data_path}; "
              f"this result cannot be traced to a split.")

    n_images = provenance.count_images(data_path, args.split)
    if n_images == 0:
        raise SystemExit(
            f"{data_path}: 0 images visible for split '{args.split}'. Refusing to "
            f"report an mAP computed over nothing -- check the dataset YAML's "
            f"`path:` and `{args.split}:` entries, and that the images exist.")

    # A generated dataset that was built without a `test` split still gets a YAML,
    # and its `test:` entry then aliases whichever split it does have. Counting
    # images then succeeds and the val mAP gets reported under a test name, with
    # nothing to distinguish the two. This fired for real: Stage 5 regenerated
    # fog_heavy with `--splits train val` over the 3A dataset, which rewrote its
    # `test:` to val, and the 3B gap was then scored on 240 val frames against a
    # 600-frame clear reference.
    eval_dir = set()
    for d in provenance.dataset_entries(data_path, args.split):
        if d.is_dir():
            eval_dir.add(d.resolve())
    for other in ("train", "val"):
        if other == args.split:
            continue
        for d in provenance.dataset_entries(data_path, other):
            if d.is_dir() and d.resolve() in eval_dir:
                raise SystemExit(
                    f"{data_path}: split '{args.split}' resolves to the same "
                    f"directory as '{other}' ({d}). Refusing to report an mAP over "
                    f"one frame set under two names -- generate this dataset with "
                    f"the split you intend to evaluate, or pass --split {other}.")

    model = YOLO(str(weights))
    m = model.val(
        data=str(data_path),
        split=args.split,
        imgsz=args.imgsz,
        project=str(RUNS),
        name=f"eval_{args.name}",
        exist_ok=True,
    )

    res = {
        "name": args.name,
        "weights": str(weights.resolve()),
        "weights_sha256": provenance.sha256_file(weights),
        "data": str(data_path.resolve()),
        "split": args.split,
        "split_hash": split_hash,
        "n_images": n_images,
        "imgsz": args.imgsz,
        "mAP50_95": round(float(m.box.map), 5),
        "mAP50": round(float(m.box.map50), 5),
        "precision": round(float(m.box.mp), 5),
        "recall": round(float(m.box.mr), 5),
        **provenance.run_record(),
    }

    out_file = Path(args.out) if args.out else HERE / f"results_{args.name}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(res, indent=2))
    print(f"Results for {args.name} ({n_images} images, split_hash={split_hash}): "
          f"mAP50-95 = {res['mAP50_95']:.4f}")
    print(f"  -> {out_file}")


if __name__ == "__main__":
    main()