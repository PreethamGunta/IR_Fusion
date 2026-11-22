#!/usr/bin/env python3
"""Prepare FLIR ADAS v2 for YOLO training.

Converts COCO annotations to YOLO format and builds sequence-disjoint
train/val/test splits.  Sequence-disjointness is critical: FLIR ADAS v2
images come from video sequences, and a random frame-level shuffle of
sequential video would leak nearby frames between train and test -- the same
contamination that was found and fixed in the M3FD resplit
(docs/memory.md:196-204).

After running this, the clear thermal YAML at data/flir_adas_v2/yolo/thermal.yaml
is the reference every Protocol 3A gap is measured against, and
data/flir_adas_v2/yolo/split_manifest.json records the split hash.

Usage:
    python flir_adras_v2/prepare.py
    python flir_adras_v2/prepare.py --modality both --seed 42
"""

import argparse
import hashlib
import json
import random
import sys
import yaml
from collections import defaultdict
from pathlib import Path

here = Path(__file__).resolve().parent
REPO = here.parent  # recover-session-12 root
DATA_ROOT = REPO / "data" / "flir_adas_v2" / "FLIR_ADAS_v2"
OUTPUT_DIR = REPO / "data" / "flir_adas_v2" / "yolo"

# 15 classes from FLIR ADAS v2 COCO
ALL_CLASSES = [
    "person", "bike", "car", "motor", "bus", "train", "truck",
    "light", "hydrant", "sign", "face", "other_vehicle",
    "backpack", "bag", "other"
]


def load_coco(coco_path):
    with open(coco_path) as f:
        return json.load(f)


def coco_category_map(coco):
    """Map COCO category_id -> YOLO index (0-based), preserving the order
    declared in ALL_CLASSES which matches the standard FLIR ADAS ordering."""
    cat_id_to_name = {cat["id"]: cat["name"] for cat in coco["categories"]}
    cat_id_to_idx = {}
    for i, name in enumerate(ALL_CLASSES):
        for cat in coco["categories"]:
            if cat["name"] == name:
                cat_id_to_idx[cat["id"]] = i
    return cat_id_to_name, cat_id_to_idx


def convert_coco_to_yolo(coco, images_root, labels_dir, cat_id_to_idx):
    """Convert COCO annotations to YOLOv8 format.

    YOLO format: <class_idx> <x_center> <y_center> <width> <height>
    All coordinates normalized to [0, 1].
    """
    labels_dir.mkdir(parents=True, exist_ok=True)

    ann_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        ann_by_img[ann["image_id"]].append(ann)

    converted = 0
    for img in coco["images"]:
        img_id = img["id"]
        anns = ann_by_img.get(img_id, [])
        img_name = Path(img["file_name"]).name
        stem = Path(img_name).stem
        label_path = labels_dir / f"{stem}.txt"

        w, h = img["width"], img["height"]
        lines = []
        for ann in anns:
            cls = cat_id_to_idx.get(ann["category_id"], -1)
            if cls < 0:
                continue
            x, y, bw, bh = ann["bbox"]
            xc = (x + bw / 2) / w
            yc = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

        label_path.write_text("\n".join(lines))
        converted += 1

    print(f"  Converted {converted} labels to YOLO format in {labels_dir}")
    return converted


def extract_video_id(image_dict):
    """Extract a video/sequence ID from the image filename.

    FLIR ADAS v2 filenames look like:
      data/video-24ysbPEGoEKKDvRt6-frame-000000-4C4FHWxwNaMyohLZt.jpg

    The video ID is the part after 'video-' up to '-frame-'.
    """
    fname = Path(image_dict["file_name"]).name
    if fname.startswith("video-") and "-frame-" in fname:
        start = len("video-")
        end = fname.index("-frame-")
        return fname[start:end]
    return fname  # fallback: treat each image as its own "video"


def build_sequence_disjoint_split(coco, split_ratios=(0.8, 0.1, 0.1), seed=42):
    """Split images into train/val/test by VIDEO SEQUENCE.

    This prevents temporal contamination: frames from the same video sequence
    never appear in more than one split.
    """
    random.seed(seed)

    # Group images by video sequence
    seq_groups = defaultdict(list)
    for img in coco["images"]:
        vid_id = extract_video_id(img)
        seq_groups[vid_id].append(img)

    sequences = sorted(seq_groups.keys())
    random.shuffle(sequences)

    n_seq = len(sequences)
    n_train = int(n_seq * split_ratios[0])
    n_val = int(n_seq * split_ratios[1])

    train_seqs = set(sequences[:n_train])
    val_seqs = set(sequences[n_train:n_train + n_val])
    test_seqs = set(sequences[n_train + n_val:])

    train, val, test = [], [], []
    for seq_id in sequences[:n_train]:
        train.extend(seq_groups[seq_id])
    for seq_id in sequences[n_train:n_train + n_val]:
        val.extend(seq_groups[seq_id])
    for seq_id in sequences[n_train + n_val:]:
        test.extend(seq_groups[seq_id])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    print(f"  Sequences: {n_seq} total ({len(train_seqs)} train, {len(val_seqs)} val, {len(test_seqs)} test)")
    print(f"  Images: train={len(train)}, val={len(val)}, test={len(test)}")

    # Verify no sequence overlap between splits (not within a split)
    train_seq_ids = set(extract_video_id(img) for img in train)
    val_seq_ids = set(extract_video_id(img) for img in val)
    test_seq_ids = set(extract_video_id(img) for img in test)

    overlap = (train_seq_ids & val_seq_ids) | (train_seq_ids & test_seq_ids) | (val_seq_ids & test_seq_ids)
    if overlap:
        print(f"  WARNING: {len(overlap)} sequences appear in multiple splits!")
        for s in sorted(overlap)[:5]:
            print(f"    {s}")

    return train, val, test


def compute_split_hash(train, val, test, seed):
    """Compute a deterministic hash of the split for provenance tracking."""
    train_stems = sorted(Path(img["file_name"]).stem for img in train)
    val_stems = sorted(Path(img["file_name"]).stem for img in val)
    test_stems = sorted(Path(img["file_name"]).stem for img in test)
    hash_input = json.dumps({
        "train": train_stems,
        "val": val_stems,
        "test": test_stems,
        "seed": seed,
    }, sort_keys=True)
    return hashlib.sha256(hash_input.encode()).hexdigest()[:16]


def build_yolo_tree(data_root, labels_dir, output_dir, train_imgs, val_imgs, test_imgs,
                    modality, split_subdir, val_subdir=None):
    """Build YOLO directory tree with symlinks to source images.

    ``split_subdir`` is the source directory for split images (e.g.
    ``images_thermal_train``).  ``val_subdir`` is the source directory for
    val-origin images (``images_thermal_val``).  Because the sequence-disjoint
    split mixes images that physically live in *both* directories, every image
    is looked up in ``split_subdir`` first and then ``val_subdir`` so that val
    images -- whose COCO ``file_name`` includes the ``data/`` prefix -- are not
    silently skipped.
    """
    if val_subdir is None:
        val_subdir = split_subdir

    subdirs = [split_subdir, val_subdir]
    # Deduplicate while preserving order.
    seen = set()
    subdirs = [s for s in subdirs if not (s in seen or seen.add(s))]

    for split_name, imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
        img_out = output_dir / "images" / split_name
        lbl_out = output_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        count = 0
        for img in imgs:
            fname = img['file_name']
            src_img = None
            for sd in subdirs:
                candidate = data_root / sd / fname
                if candidate.exists():
                    src_img = candidate
                    break
            if src_img is None:
                continue
            src_lbl = labels_dir / f"{Path(img['file_name']).stem}.txt"
            dst_img = img_out / Path(img["file_name"]).name
            dst_lbl = lbl_out / f"{Path(img['file_name']).stem}.txt"
            if dst_img.exists() or dst_img.is_symlink():
                dst_img.unlink()
            if dst_lbl.exists() or dst_lbl.is_symlink():
                dst_lbl.unlink()
            dst_img.symlink_to(src_img.resolve())
            if src_lbl.exists():
                dst_lbl.symlink_to(src_lbl.resolve())
            count += 1
        print(f"  {split_name}: {count} images")


def write_yaml(path, output_dir, names, train_imgs):
    """Write a YOLO dataset YAML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(names),
        "names": names,
    }
    path.write_text(yaml.dump(content, default_flow_style=False))
    print(f"  YAML -> {path}")


def write_split_manifest(output_dir, split_hash, train, val, test, modality, seed):
    """Write split_manifest.json with provenance info."""
    manifest = {
        "split_hash": split_hash,
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "split_ratios": {"train": len(train), "val": len(val), "test": len(test)},
        "modality": modality,
        "seed": seed,
        "dataset": "FLIR ADAS v2",
        "split_type": "sequence-disjoint",
        "n_classes": len(ALL_CLASSES),
        "class_names": ALL_CLASSES,
        "provenance": {
            "git_commit": _git_commit(),
            "git_dirty": _git_dirty(),
        },
    }
    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  manifest -> {manifest_path}")


def _git_commit():
    import subprocess
    repo = Path(__file__).resolve().parent.parent.parent.parent
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=20)
        return out.stdout.strip() or None
    except Exception:
        return None


def _git_dirty():
    import subprocess
    repo = Path(__file__).resolve().parent.parent.parent.parent
    try:
        out = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                             capture_output=True, text=True, timeout=20)
        return bool(out.stdout.strip())
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="Prepare FLIR ADAS v2 for YOLO training")
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT,
                    help="FLIR ADAS v2 root directory")
    ap.add_argument("--output", type=Path, default=OUTPUT_DIR,
                    help="output directory for YOLO structure")
    ap.add_argument("--modality", type=str, default="thermal",
                    choices=["thermal", "rgb", "both"])
    ap.add_argument("--split-ratio", type=str, default="0.8:0.1:0.1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--classes", type=str, default=",".join(ALL_CLASSES))
    args = ap.parse_args()

    class_names = [c.strip() for c in args.classes.split(",")]
    class_map = {name: i for i, name in enumerate(class_names)}
    split_ratios = tuple(map(float, args.split_ratio.split(":")))

    print(f"FLIR ADAS v2 Preparation")
    print(f"  data_root: {args.data_root}")
    print(f"  output:    {args.output}")
    print(f"  classes:   {len(class_names)} ({class_names[:5]}...)")
    print(f"  seed:      {args.seed}")

    args.output.mkdir(parents=True, exist_ok=True)
    modalities = ["thermal", "rgb"] if args.modality == "both" else [args.modality]

    for modality in modalities:
        print(f"\n=== Processing {modality} ===")

        if modality == "thermal":
            train_coco_path = args.data_root / "images_thermal_train" / "coco.json"
            val_coco_path = args.data_root / "images_thermal_val" / "coco.json"
            images_subdir = "images_thermal_train"
        else:
            train_coco_path = args.data_root / "images_rgb_train" / "coco.json"
            val_coco_path = args.data_root / "images_rgb_val" / "coco.json"
            images_subdir = "images_rgb_train"

        # Load COCO annotations
        if not train_coco_path.exists():
            print(f"FATAL: {train_coco_path} not found")
            continue
        if not val_coco_path.exists():
            print(f"FATAL: {val_coco_path} not found")
            continue

        coco_train = load_coco(train_coco_path)
        coco_val = load_coco(val_coco_path)

        # Merge train and val annotations (test video comes separately)
        coco_all = {
            "images": coco_train["images"] + coco_val["images"],
            "annotations": coco_train["annotations"] + coco_val["annotations"],
            "categories": coco_train["categories"],
        }

        print(f"  Total images: {len(coco_all['images'])}")

        # Build category map
        _, cat_id_to_idx = coco_category_map(coco_all)

        # Convert annotations to YOLO format
        labels_dir = args.output / modality / "labels_all"
        labels_dir.mkdir(parents=True, exist_ok=True)

        # The images for training/val are under images_<modality>_train/data and
        # images_<modality>_val/data. Labels are written per-image by filename stem.
        convert_coco_to_yolo(coco_all, args.data_root / images_subdir / "data", labels_dir, cat_id_to_idx)
        # Also convert val images
        val_images_subdir = f"images_{modality}_val"
        convert_coco_to_yolo(coco_all, args.data_root / val_images_subdir / "data", labels_dir, cat_id_to_idx)

        # Build sequence-disjoint splits
        train_imgs, val_imgs, test_imgs = build_sequence_disjoint_split(
            coco_all, split_ratios=split_ratios, seed=args.seed)

        # Build YOLO tree for thermal modality
        output_mod = args.output / modality
        val_images_subdir = f"images_{modality}_val"
        build_yolo_tree(args.data_root, labels_dir, output_mod,
                        train_imgs, val_imgs, test_imgs, modality,
                        images_subdir, val_subdir=val_images_subdir)

        # Write clear thermal YAML
        yaml_out = args.output / f"{modality}.yaml"
        write_yaml(yaml_out, output_mod, class_names, train_imgs)

        # Write split manifest with hash
        split_hash = compute_split_hash(train_imgs, val_imgs, test_imgs, args.seed)
        write_split_manifest(output_mod, split_hash, train_imgs, val_imgs, test_imgs, modality, args.seed)

        print(f"\n  Done. Split hash: {split_hash}")
        print(f"  YAML: {yaml_out}")

    print("\nPreparation complete.")
    print(f"Use data/flir_adas_v2/yolo/thermal.yaml for clear training.")


if __name__ == "__main__":
    main()
