#!/usr/bin/env python3
"""
Method 2 — Depth-based weather simulation for FLIR ADAS v2.

Uses monocular depth estimation (Depth Anything V2) to create spatially-varying
transmission maps for fog, and depth-modulated rain/snow.

Weather physics for thermal IR:
- Fog: Koschmieder transmission t(x,y) = exp(-beta_IR * D(x,y)), beta_IR < beta_RGB
- Rain: Depth-attenuated streaks (farther = lighter)
- Snow: Depth-modulated accumulation (horizontal surfaces, closer = more)

Usage:
    python method2_depth/generate.py --split train --conditions fog,rain,snow --severities light,medium,heavy
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# Depth-based parameters
FOG_BETAS = {"light": 1.5, "medium": 3.0, "heavy": 5.0}      # optical depth
FOG_BETA_IR_RATIO = 0.3  # IR penetrates fog better than RGB
RAIN_DEPTH_ATTEN = {"light": 0.3, "medium": 0.6, "heavy": 0.9}
SNOW_DEPTH_ATTEN = {"light": 0.2, "medium": 0.5, "heavy": 0.8}


class DepthEstimator:
    """Wrapper for Depth Anything V2."""
    
    def __init__(self, encoder="vitl", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Loading Depth Anything V2 ({encoder}) on {self.device}...")
        
        # Use the official Depth Anything V2
        self.model = torch.hub.load("yvanyin/metric3d", "metric3d_vit_giant2_800k", pretrained=True)
        self.model.to(self.device).eval()
        
    @torch.no_grad()
    def estimate(self, image_rgb):
        """Estimate depth from RGB image. Returns depth map [H, W] in meters."""
        # Image should be RGB, 0-255, HWC
        h, w = image_rgb.shape[:2]
        # Resize to model input size (keep aspect ratio)
        input_size = 518
        scale = input_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img_resized = cv2.resize(image_rgb, (new_w, new_h))
        
        # Normalize
        img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Pad to multiple of 14
        pad_h = (14 - new_h % 14) % 14
        pad_w = (14 - new_w % 14) % 14
        img_tensor = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h))
        
        # Predict
        with torch.no_grad():
            depth = self.model.infer(img_tensor)
        
        # Remove padding and resize back
        depth = depth[:, :new_h, :new_w]
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False
        ).squeeze()
        
        return depth.cpu().numpy()


def estimate_depth_metric3d(rgb_image):
    """Alternative: Use metric3d directly if available."""
    try:
        from metric3d import metric3d_predict
        depth = metric3d_predict(rgb_image)
        return depth
    except ImportError:
        return None


def apply_fog_depth(rgb_image, depth_map, beta, atmospheric_light=None):
    """
    Koschmieder fog model with depth-based transmission.
    I_fog = I * t + A * (1 - t)
    t(x,y) = exp(-beta * D(x,y))
    """
    if atmospheric_light is None:
        # Estimate as top 1% brightest pixels
        flat = rgb_image.reshape(-1, 3)
        atmospheric_light = np.percentile(flat, 99, axis=0).astype(np.float32)
    
    # Transmission map
    transmission = np.exp(-beta * depth_map)
    transmission = np.clip(transmission, 0.01, 1.0)
    transmission = transmission[:, :, np.newaxis]  # HWC
    
    # Apply fog
    foggy = rgb_image.astype(np.float32) * transmission + atmospheric_light * (1 - transmission)
    return np.clip(foggy, 0, 255).astype(np.uint8)


def apply_fog_thermal_depth(thermal_image, depth_map, beta_ir):
    """
    Thermal fog: contrast loss + path radiance.
    t_IR = exp(-beta_IR * D)
    I_fog = I * t_IR + path_radiance * (1 - t_IR)
    Path radiance for thermal: low temperature background (~40 in 8-bit)
    """
    path_radiance = 40.0
    transmission = np.exp(-beta_ir * depth_map)
    transmission = np.clip(transmission, 0.01, 1.0)
    
    foggy = thermal_image.astype(np.float32) * transmission + path_radiance * (1 - transmission)
    return np.clip(foggy, 0, 255).astype(np.uint8)


def apply_rain_depth(rgb_image, depth_map, intensity):
    """Depth-attenuated rain streaks using Albumentations."""
    import albumentations as A
    
    # Normalize depth to [0,1] for attenuation
    depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
    # Farther = less rain
    rain_mask = 1.0 - depth_norm * intensity * 0.5
    rain_mask = np.clip(rain_mask, 0.1, 1.0)
    
    # Apply base rain
    slant = 15
    transform = A.Compose([A.RandomRain(
        slant_lower=-slant, slant_upper=slant,
        drop_length=20, drop_width=1,
        drop_color=(200, 200, 200),
        blur_value=3,
        brightness_coefficient=0.9,
        rain_type="heavy",
        p=1.0
    )])
    rainy = transform(image=rgb_image)["image"]
    
    # Attenuate by depth
    rain_mask = rain_mask[:, :, np.newaxis]
    result = (rgb_image.astype(np.float32) * (1 - rain_mask) + rainy.astype(np.float32) * rain_mask)
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_snow_depth(rgb_image, depth_map, intensity):
    """Depth-modulated snow accumulation."""
    import albumentations as A
    
    # Snow accumulates more on horizontal surfaces (sky-facing)
    # Approximate: closer surfaces get more snow
    depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
    snow_mask = 1.0 - depth_norm * intensity * 0.3
    snow_mask = np.clip(snow_mask, 0.2, 1.0)
    
    transform = A.Compose([A.RandomSnow(
        snow_point_lower=0.1, snow_point_upper=0.3,
        brightness_coeff=1.5,
        p=1.0
    )])
    snowy = transform(image=rgb_image)["image"]
    
    snow_mask = snow_mask[:, :, np.newaxis]
    result = (rgb_image.astype(np.float32) * (1 - snow_mask) + snowy.astype(np.float32) * snow_mask)
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_rain_thermal_depth(thermal_image, depth_map, intensity):
    """Thermal rain with depth attenuation."""
    # Similar to Method 1 but depth-modulated
    depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
    washout_mask = 1.0 - depth_norm * intensity * 0.4
    washout_mask = np.clip(washout_mask, 0.2, 1.0)
    
    # Emissivity washout
    mean_val = np.mean(thermal_image)
    washed = (thermal_image.astype(np.float32) - mean_val) * washout_mask + mean_val
    
    # Spatial smoothing
    ksize = max(3, int(intensity * 7) | 1)
    blurred = cv2.GaussianBlur(washed, (ksize, ksize), intensity * 2)
    
    result = cv2.addWeighted(washed.astype(np.uint8), 1 - intensity * 0.3, 
                            blurred, intensity * 0.3, 0)
    return result


def apply_snow_thermal_depth(thermal_image, depth_map, intensity):
    """Thermal snow with depth modulation."""
    depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
    snow_mask = 1.0 - depth_norm * intensity * 0.5
    snow_mask = np.clip(snow_mask, 0.2, 1.0)
    
    # Emissivity washout
    mean_val = np.mean(thermal_image)
    contrast_reduced = (thermal_image.astype(np.float32) - mean_val) * (1 - intensity * 0.6) + mean_val
    
    # Convective cooling
    cooled = contrast_reduced - intensity * 15
    
    # Particle texture on horizontal surfaces (approximated by closer depth)
    noise = np.random.normal(0, intensity * 10, thermal_image.shape).astype(np.float32)
    horizontal_mask = (1.0 - depth_norm).astype(np.float32)
    textured = cooled + noise * horizontal_mask * snow_mask
    
    return np.clip(textured, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description="Method 2: Depth-based weather simulation for FLIR ADAS v2")
    ap.add_argument("--data-root", type=Path, default=Path("data/FLIR_ADAS_v2/FLIR_ADAS_v2"))
    ap.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    ap.add_argument("--conditions", type=str, default="fog,rain,snow")
    ap.add_argument("--severities", type=str, default="light,medium,heavy")
    ap.add_argument("--output-root", type=Path, default=Path("flir_adras_v2/method2_depth/output"))
    ap.add_argument("--max-images", type=int, default=-1)
    ap.add_argument("--encoder", type=str, default="vitl", choices=["vits", "vitb", "vitl"])
    args = ap.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",")]
    severities = [s.strip() for s in args.severities.split(",")]

    # Load clear RGB images
    split_dir = args.data_root / f"images_rgb_{args.split}"
    if not split_dir.exists():
        print(f"Split directory not found: {split_dir}")
        sys.exit(1)

    import json
    with open(split_dir / "coco.json") as f:
        coco = json.load(f)

    clear_images = []
    for img in coco["images"]:
        weather = img.get("extra_info", {}).get("weather", "unknown")
        if weather == "clear":
            clear_images.append(img)

    print(f"Found {len(clear_images)} clear RGB images in {args.split}")
    if args.max_images > 0:
        clear_images = clear_images[:args.max_images]

    # Output dirs
    for condition in conditions:
        for severity in severities:
            for mod in ["thermal", "rgb"]:
                out_dir = args.output_root / mod / f"{condition}_{severity}" / args.split
                out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize depth estimator
    try:
        depth_estimator = DepthEstimator(encoder=args.encoder)
    except Exception as e:
        print(f"Failed to load depth estimator: {e}")
        print("Trying metric3d...")
        return

    # Process
    for img_info in tqdm(clear_images, desc="Generating depth-based weather"):
        img_id = img_info["id"]
        rel_path = img_info["file_name"]
        rgb_path = Path(args.data_root) / rel_path
        
        if not rgb_path.exists():
            continue

        rgb_img = cv2.imread(str(rgb_path))
        if rgb_img is None:
            continue
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        # Find paired thermal
        thermal_rel = rel_path.replace("images_rgb", "images_thermal")
        thermal_path = Path(args.data_root) / thermal_rel
        has_thermal = thermal_path.exists()
        thermal_img = cv2.imread(str(thermal_path), cv2.IMREAD_GRAYSCALE) if has_thermal else None

        # Estimate depth from RGB
        depth_map = depth_estimator.estimate(rgb_img)

        for condition in conditions:
            for severity in severities:
                if condition == "fog":
                    beta = FOG_BETAS[severity]
                    beta_ir = beta * FOG_BETA_IR_RATIO
                    
                    rgb_out = apply_fog_depth(rgb_img, depth_map, beta)
                    thermal_out = apply_fog_thermal_depth(thermal_img, depth_map, beta_ir) if has_thermal else None
                    
                elif condition == "rain":
                    intensity = RAIN_DEPTH_ATTEN[severity]
                    rgb_out = apply_rain_depth(rgb_img, depth_map, intensity)
                    thermal_out = apply_rain_thermal_depth(thermal_img, depth_map, intensity) if has_thermal else None
                    
                elif condition == "snow":
                    intensity = SNOW_DEPTH_ATTEN[severity]
                    rgb_out = apply_snow_depth(rgb_img, depth_map, intensity)
                    thermal_out = apply_snow_thermal_depth(thermal_img, depth_map, intensity) if has_thermal else None
                else:
                    continue

                # Save
                out_name = f"{Path(img_info['file_name']).stem}_{condition}_{severity}.jpg"
                
                if has_thermal:
                    out_path_t = args.output_root / "thermal" / f"{condition}_{severity}" / args.split / out_name
                    cv2.imwrite(str(out_path_t), thermal_out)
                
                out_path_r = args.output_root / "rgb" / f"{condition}_{severity}" / args.split / Path(img_info["file_name"]).name
                cv2.imwrite(str(out_path_r), cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR))

    print(f"Done. Output in: {args.output_root}")


if __name__ == "__main__":
    main()