#!/usr/bin/env python3
"""
Inspect FLIR ADAS v2 dataset after download.

Reports:
- Directory structure
- Image counts, resolutions, formats
- Annotation format (YOLO, COCO, VOC, etc.)
- Class names and distribution
- Paired IR+RGB alignment (filenames)
- Any weather/lighting metadata
- Train/val/test splits (if present)
- Temporal redundancy (if sequences can be detected)

Usage:
    python flir_adras_v2/inspect.py
    python flir_adras_v2/inspect.py --path data/FLIR_ADAS_v2
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Pillow not installed. Install with: pip install Pillow")
    sys.exit(1)


def find_images(root, exts=("jpg", "jpeg", "png", "bmp")):
    root = Path(root)
    return [p for ext in exts for p in root.rglob(f"*.{ext}")]


def find_labels(root, exts=("txt", "xml", "json")):
    root = Path(root)
    return [p for ext in exts for p in root.rglob(f"*.{ext}")]


def inspect_images(paths, max_sample=100):
    sizes = Counter()
    formats = Counter()
    modes = Counter()
    for p in paths[:max_sample]:
        try:
            with Image.open(p) as img:
                sizes[img.size] += 1
                formats[img.format] += 1
                modes[img.mode] += 1
        except Exception as e:
            print(f"  Warning: could not open {p}: {e}")
    return sizes, formats, modes


def inspect_yolo_labels(paths, max_sample=100):
    class_counts = Counter()
    box_counts = Counter()
    for p in paths[:max_sample]:
        try:
            lines = p.read_text().strip().splitlines()
            box_counts[len(lines)] += 1
            for line in lines:
                parts = line.strip().split()
                if parts:
                    class_counts[int(parts[0])] += 1
        except Exception:
            pass
    return class_counts, box_counts


def check_pairs(paths):
    ir_paths = {p.name.replace("_thermal", "").replace("_ir", "").replace("THERMAL", "RGB"): p
                for p in paths if "_thermal" in p.name or "_ir" in p.name or "THERMAL" in p.name}
    rgb_paths = {p.name: p for p in paths if "_rgb" in p.name or "_visible" in p.name or "RGB" in p.name}
    return ir_paths, rgb_paths


def main():
    ap = argparse.ArgumentParser(description="Inspect FLIR ADAS v2 dataset")
    ap.add_argument("--path", type=Path, default=Path("data/FLIR_ADAS_v2"),
                    help="Dataset root path")
    ap.add_argument("--max-sample", type=int, default=100,
                    help="Max images/labels to sample for stats")
    args = ap.parse_args()

    if not args.path.exists():
        print(f"Dataset not found at {args.path}")
        print("Run: python flir_adras_v2/download.py")
        sys.exit(1)

    print(f"Inspecting: {args.path}")
    print("=" * 60)

    # Directory structure
    print("\n## Directory structure (top 3 levels)")
    for root, dirs, files in os.walk(args.path):
        level = root.replace(str(args.path), '').count(os.sep)
        if level > 3:
            continue
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/ ({len(files)} files)")

    # Images
    all_images = find_images(args.path)
    print(f"\n## Images: {len(all_images)} total")

    if all_images:
        sizes, formats, modes = inspect_images(all_images, args.max_sample)
        print(f"\nResolutions (sampled {min(len(all_images), args.max_sample)}):")
        for size, count in sizes.most_common(10):
            print(f"  {size}: {count}")
        print(f"\nFormats: {dict(formats)}")
        print(f"Modes: {dict(modes)}")

    # Labels
    all_labels = find_labels(args.path)
    print(f"\n## Labels: {len(all_labels)} total")

    if all_labels:
        yolo_labels = [p for p in all_labels if p.suffix == ".txt"]
        if yolo_labels:
            class_counts, box_counts = inspect_yolo_labels(yolo_labels, args.max_sample)
            print(f"\nYOLO labels found: {len(yolo_labels)}")
            print(f"\nClass distribution (sampled):")
            for cls, count in class_counts.most_common():
                print(f"  class {cls}: {count} boxes")
            print(f"\nBoxes per image distribution (sampled):")
            for n, count in box_counts.most_common(10):
                print(f"  {n} boxes: {count} images")

    # Pairs
    ir_paths, rgb_paths = check_pairs(all_images)
    print(f"\n## Paired IR+RGB")
    print(f"  IR candidates: {len(ir_paths)}")
    print(f"  RGB candidates: {len(rgb_paths)}")

    # Weather metadata
    print("\n## Weather metadata")
    weather_dirs = [d for d in args.path.rglob("*")
                    if d.is_dir() and any(w in d.name.lower() for w in
                                          ["fog", "rain", "snow", "weather", "adverse", "clear"])]
    if weather_dirs:
        print("  Possible weather-related directories:")
        for d in weather_dirs[:20]:
            print(f"    {d.relative_to(args.path)}")
    else:
        print("  No obvious weather-condition directories found.")

    # Metadata files
    print("\n## Metadata files")
    meta_files = [p for p in args.path.rglob("*")
                  if p.is_file() and p.suffix in (".csv", ".json", ".yaml", ".yml")]
    for mf in meta_files[:20]:
        print(f"  {mf.relative_to(args.path)}")

    # Summary
    print("\n## Summary")
    print(f"  Total images: {len(all_images)}")
    print(f"  Total labels: {len(all_labels)}")
    print(f"  IR candidates: {len(ir_paths)}")
    print(f"  RGB candidates: {len(rgb_paths)}")
    print(f"  Weather dirs: {len(weather_dirs)}")
    print(f"  Metadata files: {len(meta_files)}")

    # Write report
    report = {
        "total_images": len(all_images),
        "total_labels": len(all_labels),
        "ir_candidates": len(ir_paths),
        "rgb_candidates": len(rgb_paths),
        "weather_dirs": [str(d.relative_to(args.path)) for d in weather_dirs],
        "metadata_files": [str(m.relative_to(args.path)) for m in meta_files],
        "resolutions": {str(k): v for k, v in sizes.most_common(10)},
        "formats": dict(formats),
        "modes": dict(modes),
        "class_distribution": {str(k): v for k, v in class_counts.most_common()} if yolo_labels else {},
    }
    report_path = Path(__file__).parent / "inspect_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
