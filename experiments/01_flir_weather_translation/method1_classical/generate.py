#!/usr/bin/env python3
"""
Method 1 — Classical weather simulation for FLIR ADAS v2.

Applies uniform veil (fog), streak overlay (rain), and particle overlay (snow)
to clear images using Albumentations/OpenCV. No training required.

Weather physics for thermal IR:
- Fog: contrast reduction + path radiance (not bright veil)
- Rain: emissivity wash-out (wet surfaces), spatial smoothing
- Snow: emissivity wash-out + convective cooling + particle texture

Usage:
    python method1_classical/generate.py --split train --conditions fog,rain,snow --severities light,medium,heavy
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

try:
    import albumentations as A
except ImportError:
    print("albumentations not installed. Install with: pip install albumentations")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# Thermal IR specific parameters
FOG_ALPHAS = {"light": 0.8, "medium": 0.5, "heavy": 0.2}      # transmission
RAIN_INTENSITIES = {"light": 0.3, "medium": 0.6, "heavy": 0.9}  # streak density
SNOW_INTENSITIES = {"light": 0.2, "medium": 0.5, "heavy": 0.8}  # particle density


def apply_fog_thermal(image, alpha):
    """
    Thermal fog: contrast reduction + additive path radiance.
    I_fog = alpha * I + (1 - alpha) * path_radiance
    Path radiance for thermal: low-temperature background (value ~30-50 in 8-bit)
    """
    path_radiance = 40  # low-temp background radiance in 8-bit [0,255]
    foggy = (alpha * image.astype(np.float32) + (1 - alpha) * path_radiance).astype(np.uint8)
    return foggy


def apply_rain_thermal(image, intensity):
    """
    Thermal rain: emissivity wash-out + spatial smoothing.
    No bright streaks in thermal. Simulate by reducing local contrast and adding blur.
    """
    # Reduce local contrast (emissivity wash-out)
    kernel = np.ones((5, 5), np.float32) / 25
    local_mean = cv2.filter2D(image.astype(np.float32), -1, kernel)
    washed = (image.astype(np.float32) * (1 - intensity * 0.5) + local_mean * intensity * 0.5).astype(np.uint8)
    
    # Add slight blur to simulate water film
    ksize = max(3, int(intensity * 7) | 1)
    blurred = cv2.GaussianBlur(washed, (ksize, ksize), intensity * 2)
    
    # Blend original and blurred
    result = cv2.addWeighted(washed, 1 - intensity * 0.3, blurred, intensity * 0.3, 0)
    return result


def apply_snow_thermal(image, intensity):
    """
    Thermal snow: emissivity wash-out + convective cooling + particle texture.
    Snow on surfaces -> higher emissivity -> reduced contrast.
    Cold snow -> convective cooling of warm surfaces.
    """
    # Emissivity wash-out (contrast reduction)
    mean_val = np.mean(image)
    contrast_reduced = (image.astype(np.float32) - mean_val) * (1 - intensity * 0.6) + mean_val
    
    # Convective cooling (shift warm regions cooler)
    cooled = contrast_reduced - intensity * 20
    
    # Add snow particle texture (high-frequency noise on bright regions)
    noise = np.random.normal(0, intensity * 15, image.shape).astype(np.float32)
    bright_mask = (image > mean_val).astype(np.float32)
    textured = cooled + noise * bright_mask
    
    return np.clip(textured, 0, 255).astype(np.uint8)


def apply_fog_rgb(image, alpha):
    """RGB fog: bright veil using Albumentations."""
    transform = A.Compose([A.RandomFog(fog_coef_lower=1-alpha, fog_coef_upper=1-alpha, p=1.0)])
    return transform(image=image)["image"]


def apply_rain_rgb(image, intensity):
    """RGB rain: bright streaks using Albumentations."""
    slant = 15  # degrees from vertical
    transform = A.Compose([A.RandomRain(
        slant_lower=-slant, slant_upper=slant,
        drop_length=20, drop_width=1,
        drop_color=(200, 200, 200),
        blur_value=3,
        brightness_coefficient=0.9,
        rain_type="heavy" if intensity > 0.6 else "drizzle",
        p=1.0
    )])
    return transform(image=image)["image"]


def apply_snow_rgb(image, intensity):
    """RGB snow: white particles using Albumentations."""
    transform = A.Compose([A.RandomSnow(
        snow_point_lower=0.1, snow_point_upper=0.3,
        brightness_coeff=1.5,
        p=1.0
    )])
    return transform(image=image)["image"]


def main():
    ap = argparse.ArgumentParser(description="Method 1: Classical weather simulation for FLIR ADAS v2")
    ap.add_argument("--data-root", type=Path, default=Path("data/FLIR_ADAS_v2/FLIR_ADAS_v2"),
                    help="FLIR ADAS v2 root directory")
    ap.add_argument("--split", type=str, default="train", choices=["train", "val", "test"],
                    help="Dataset split to process")
    ap.add_argument("--modalities", type=str, default="thermal,rgb",
                    help="Comma-separated: thermal,rgb")
    ap.add_argument("--conditions", type=str, default="fog,rain,snow",
                    help="Comma-separated: fog,rain,snow")
    ap.add_argument("--severities", type=str, default="light,medium,heavy",
                    help="Comma-separated severity levels")
    ap.add_argument("--output-root", type=Path, default=Path("experiments/01_flir_weather_translation/method1_classical/output"),
                    help="Output root directory")
    ap.add_argument("--max-images", type=int, default=-1,
                    help="Max images per condition (debug)")
    args = ap.parse_args()

    modalities = [m.strip() for m in args.modalities.split(",")]
    conditions = [c.strip() for c in args.conditions.split(",")]
    severities = [s.strip() for s in args.severities.split(",")]

    # Load clear images from the split
    split_dir = args.data_root / f"images_thermal_{args.split}"
    if not split_dir.exists():
        print(f"Split directory not found: {split_dir}")
        sys.exit(1)

    # Load COCO annotations to filter by weather
    import json
    with open(split_dir / "coco.json") as f:
        coco = json.load(f)

    # Build image lookup
    img_by_id = {img["id"]: img for img in coco["images"]}
    
    # Filter for clear images (or all if we want to generate on all)
    clear_images = []
    for img in coco["images"]:
        weather = img.get("extra_info", {}).get("weather", "unknown")
        if weather == "clear":
            clear_images.append(img)

    print(f"Found {len(clear_images)} clear images in {args.split} split")
    if args.max_images > 0:
        clear_images = clear_images[:args.max_images]
        print(f"Limited to {args.max_images} images for debugging")

    # Output directories
    for modality in modalities:
        for condition in conditions:
            for severity in severities:
                out_dir = args.output_root / modality / f"{condition}_{severity}" / args.split
                out_dir.mkdir(parents=True, exist_ok=True)

    # Process images
    for img_info in tqdm(clear_images, desc="Generating weather"):
        img_id = img_info["id"]
        rel_path = img_info["file_name"]
        thermal_path = Path(args.data_root) / rel_path
        
        if not thermal_path.exists():
            alt_path = args.data_root / "FLIR_ADAS_v2" / rel_path
            if alt_path.exists():
                thermal_path = alt_path
            else:
                print(f"Warning: {thermal_path} not found")
                continue

        thermal_img = cv2.imread(str(thermal_path), cv2.IMREAD_GRAYSCALE)
        if thermal_img is None:
            print(f"Warning: could not read {thermal_path}")
            continue

        # Find paired RGB image
        rgb_rel = rel_path.replace("images_thermal", "images_rgb")
        rgb_path = Path(args.data_root) / rgb_rel
        if not rgb_path.exists():
            rgb_path = args.data_root / "FLIR_ADAS_v2" / rgb_rel
        has_rgb = rgb_path.exists()
        rgb_img = cv2.imread(str(rgb_path)) if has_rgb else None

        for condition in conditions:
            for severity in severities:
                # Thermal
                if condition == "fog":
                    alpha = FOG_ALPHAS[severity]
                    thermal_out = apply_fog_thermal(thermal_img, alpha)
                elif condition == "rain":
                    intensity = RAIN_INTENSITIES[severity]
                    thermal_out = apply_rain_thermal(thermal_img, intensity)
                elif condition == "snow":
                    intensity = SNOW_INTENSITIES[severity]
                    thermal_out = apply_snow_thermal(thermal_img, intensity)
                else:
                    continue

                # Save thermal
                out_name = f"{Path(rel_path).stem}_{condition}_{severity}.jpg"
                out_path = args.output_root / "thermal" / f"{condition}_{severity}" / args.split / out_name
                cv2.imwrite(str(out_path), thermal_out)

                # RGB if available
                if has_rgb and rgb_img is not None:
                    if condition == "fog":
                        alpha = FOG_ALPHAS[severity]
                        rgb_out = apply_fog_rgb(rgb_img, alpha)
                    elif condition == "rain":
                        intensity = RAIN_INTENSITIES[severity]
                        rgb_out = apply_rain_rgb(rgb_img, intensity)
                    elif condition == "snow":
                        intensity = SNOW_INTENSITIES[severity]
                        rgb_out = apply_snow_rgb(rgb_img, intensity)
                    else:
                        continue

                    out_path_rgb = args.output_root / "rgb" / f"{condition}_{severity}" / args.split / out_name
                    cv2.imwrite(str(out_path_rgb), rgb_out)

    print(f"Done. Output in: {args.output_root}")


if __name__ == "__main__":
    main()