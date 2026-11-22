#!/usr/bin/env python3
"""YOLO training for Protocol 3A (domain gap measurement) and 3B (gap-driven evaluation).

Modes:
  clear_baseline  : Train YOLOv12s on clear thermal images (baseline)
  3a              : Train on clear + synthetic adverse (gap training)
  3b              : Train on clear + synthetic adverse, then translate test set

Usage:
    python experiments/01_flir_weather_translation/train_detector.py --mode clear_baseline
    python experiments/01_flir_weather_translation/train_detector.py --mode 3a --condition fog --method cyclegan
    python experiments/01_flir_weather_translation/train_detector.py --mode 3b --condition snow --method cyclediff
"""

import argparse
import hashlib
import json
import yaml
from pathlib import Path

here = Path(__file__).resolve().parent
EXP = here                       # experiments/01_flir_weather_translation
REPO = here.parents[1]           # release root
DATA = REPO / "data" / "flir_adas_v2"

try:
    from ultralytics import YOLO
except ImportError:
    print("ultralytics not installed. Install with: pip install ultralytics")
    import sys; sys.exit(1)


def build_yaml(yaml_path, train_dir, val_dir, names):
    """Write a YOLO dataset YAML."""
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    content = {
        "path": str(train_dir.parent.resolve()),
        "train": f"images/train",
        "val": f"images/val",
        "test": f"images/test",
        "nc": len(names),
        "names": names,
    }
    yaml_path.write_text(yaml.dump(content, default_flow_style=False))


def make_mixed_yaml(yaml_path, clear_yaml, adv_yamls, names, split_hash):
    """Build a combined YAML for 3A: clear + synthetic adverse train, clear val/test."""
    import yaml as pyyaml
    clear_data = pyyaml.safe_load(open(clear_yaml))
    adv_data = pyyaml.safe_load(open(adv_yamls[0])) if adv_yamls else {}

    combined = {
        "path": str(REPO / "data" / "flir_adas_v2" / "yolo" / "train_3a"),
        "train": ["images/train_clear", "images/train_adv"],
        "val": "images/val",
        "test": "images/test",
        "nc": len(names),
        "names": names,
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(pyyaml.dump(combined, default_flow_style=False))


def main():
    ap = argparse.ArgumentParser(description="YOLO training for Protocol 3A/3B")
    ap.add_argument("--mode", type=str, required=True,
                    choices=["clear_baseline", "3a", "3b"],
                    help="clear_baseline: train on clear. 3a: clear + synthetic. 3b: translate test set")
    ap.add_argument("--condition", type=str, choices=["fog", "rain", "snow"],
                    help="Adverse weather condition")
    ap.add_argument("--method", type=str, choices=["cyclegan", "cyclediff"],
                    help="Synthetic weather generator")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--weights", type=str, default="yolov12s.pt")
    ap.add_argument("--resume", type=str, default=None,
                    help="Resume from checkpoint path")
    args = ap.parse_args()

    names = [
        "person", "bike", "car", "motor", "bus", "train", "truck",
        "light", "hydrant", "sign", "face", "other_vehicle",
        "backpack", "bag", "other"
    ]

    # Paths
    yolo_root = REPO / "data" / "flir_adas_v2" / "yolo"
    clear_yaml = yolo_root / "thermal.yaml"
    runs = EXP / "runs"

    # Load split manifest for hash verification
    manifest_path = yolo_root / "thermal" / "split_manifest.json"
    split_hash_ref = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        split_hash_ref = manifest["split_hash"]

    if args.mode == "clear_baseline":
        run_name = "clear_baseline"
        run_dir = runs / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Verify split hash hasn't changed
        if split_hash_ref and (run_dir / "split_hash").exists():
            stored = (run_dir / "split_hash").read_text().strip()
            if stored != split_hash_ref:
                print(f"ERROR: split_hash changed ({stored} -> {split_hash_ref}). Re-run prepare.py.")
                import sys; sys.exit(1)
        (run_dir / "split_hash").write_text(split_hash_ref or "")

        model = YOLO(args.weights) if not args.resume else YOLO(args.resume)
        model.train(
            data=str(clear_yaml),
            epochs=args.epochs,
            batch=args.batch_size,
            imgsz=args.img_size,
            name=run_name,
            project=str(runs),
            resume=bool(args.resume),
            cache=False,
        )

    elif args.mode in ("3a", "3b"):
        if not args.condition or not args.method:
            print("ERROR: --condition and --method required for 3a/3b modes")
            import sys; sys.exit(1)

        method_dir = (
            EXP / "method3_cyclegan" / "runs" / f"cyclegan_{args.condition}"
            if args.method == "cyclegan"
            else EXP / "method4_cyclediff" / "sample_results" / f"clear2{args.condition}"
        )

        # Find synthetic adverse images
        adv_root = REPO / "data" / "flir_adas_v2" / "synthetic" / args.method / args.condition
        if not adv_root.exists():
            print(f"ERROR: synthetic images not found at {adv_root}")
            print("  Run the translator first (train_cyclegan.sh / train_cyclediff.sh)")
            import sys; sys.exit(1)

        if args.mode == "3a":
            run_name = f"3a_{args.method}_{args.condition}"
            # Build combined YAML
            adv_yaml = adv_root / f"synthetic_{args.condition}.yaml"
            combined_yaml = yolo_root / f"3a_{args.method}_{args.condition}.yaml"
            make_mixed_yaml(combined_yaml, clear_yaml, [adv_yaml], names, split_hash_ref)

            run_dir = runs / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "split_hash").write_text(split_hash_ref or "")
            (run_dir / "provenance").write_text(json.dumps({
                "clear_baseline": str(clear_yaml),
                "synthetic_source": str(adv_root),
                "method": args.method,
                "condition": args.condition,
                "split_hash": split_hash_ref,
            }, indent=2))

            model = YOLO(args.weights) if not args.resume else YOLO(args.resume)
            model.train(
                data=str(combined_yaml),
                epochs=args.epochs,
                batch=args.batch_size,
                imgsz=args.img_size,
                name=run_name,
                project=str(runs),
                resume=bool(args.resume),
                cache=False,
            )

        else:  # mode == "3b"
            run_name = f"3b_{args.method}_{args.condition}"
            # For 3B: translate the TEST set to adverse weather, then evaluate
            # with the clear-trained model (measures generalization gap)
            test_adv_root = adv_root / "test"
            if not test_adv_root.exists():
                print(f"ERROR: translated test set not found at {test_adv_root}")
                import sys; sys.exit(1)

            # 3B uses clear baseline model to evaluate on translated test
            clear_run = runs / "clear_baseline"
            if not (clear_run / "weights" / "best.pt").exists():
                print("ERROR: clear baseline weights not found. Train clear_baseline first.")
                import sys; sys.exit(1)

            model = YOLO(str(clear_run / "weights" / "best.pt"))
            run_dir = runs / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            # Evaluate
            metrics = model.val(
                data=str(test_adv_root / "test.yaml"),
                batch=args.batch_size,
                imgsz=args.img_size,
                name=run_name,
                project=str(runs),
                save=True,
                save_json=True,
                cache=False,
            )

            # Save provenance
            (run_dir / "provenance").write_text(json.dumps({
                "clear_baseline": str(clear_run / "weights" / "best.pt"),
                "translated_test": str(test_adv_root),
                "method": args.method,
                "condition": args.condition,
                "split_hash": split_hash_ref,
                "metrics": metrics.results_dict if hasattr(metrics, 'results_dict') else {},
            }, indent=2))

            print(f"\n3B results saved to {run_dir}")
            print(f"mAP50: {metrics.results_dict.get('metrics/mAP50(B)', 0):.4f}")
            print(f"mAP50-95: {metrics.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")


if __name__ == "__main__":
    main()
