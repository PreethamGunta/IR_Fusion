#!/usr/bin/env python3
"""
Method 1: Classical weather simulation for FLIR ADAS v2.
Applies uniform fog veil, rain streaks, and snow particles to thermal and RGB images.
"""

import os
import json
import cv2
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
import argparse
import shutil

SEVERITIES = {
    'fog': {'light': 0.7, 'medium': 0.4, 'heavy': 0.15},
    'rain': {'light': 100, 'medium': 300, 'heavy': 600},
    'snow': {'light': 500, 'medium': 1500, 'heavy': 3000},
}

def apply_fog_thermal(image, alpha):
    """Apply fog to thermal (IR) image: reduce contrast, add path radiance."""
    mean_val = np.mean(image)
    fogged = image * alpha + mean_val * (1 - alpha)
    return np.clip(fogged, 0, 255).astype(np.uint8)

def apply_fog_rgb(image, alpha, A=200):
    """Apply fog to RGB image: Koschmieder model with atmospheric light A."""
    fogged = image * alpha + (1 - alpha) * A
    return np.clip(fogged, 0, 255).astype(np.uint8)

def apply_rain_thermal(image, num_drops):
    """Apply rain to thermal: spatial smoothing (emissivity wash-out)."""
    h, w = image.shape[:2]
    rain_layer = np.zeros_like(image, dtype=np.float32)
    for _ in range(num_drops):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        length = random.randint(10, 30)
        angle = random.uniform(-np.pi/6, np.pi/6)
        x2 = int(x + length * np.sin(angle))
        y2 = int(y + length * np.cos(angle))
        cv2.line(rain_layer, (x, y), (x2, y2), (255, 255, 255), 1)
    rain_layer = cv2.GaussianBlur(rain_layer, (3, 3), 0)
    result = cv2.addWeighted(image.astype(np.float32), 0.8, rain_layer, 0.2, 0)
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_rain_rgb(image, num_drops):
    """Apply rain to RGB: bright streaks."""
    h, w = image.shape[:2]
    rain_layer = np.zeros_like(image, dtype=np.float32)
    for _ in range(num_drops):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        length = random.randint(15, 40)
        angle = random.uniform(-np.pi/6, np.pi/6)
        x2 = int(x + length * np.sin(angle))
        y2 = int(y + length * np.cos(angle))
        color = (random.randint(180, 255), random.randint(180, 255), random.randint(180, 255))
        cv2.line(rain_layer, (x, y), (x2, y2), color, 1)
    rain_layer = cv2.GaussianBlur(rain_layer, (3, 3), 0)
    result = cv2.addWeighted(image.astype(np.float32), 1.0, rain_layer, 0.6, 0)
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_snow_thermal(image, num_flakes):
    """Apply snow to thermal: contrast reduction + high-freq noise."""
    h, w = image.shape[:2]
    snow_mask = np.zeros((h, w), dtype=np.float32)
    for _ in range(num_flakes):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        r = random.randint(1, 3)
        cv2.circle(snow_mask, (x, y), r, 1.0, -1)
    snow_mask = cv2.GaussianBlur(snow_mask, (5, 5), 0)
    mean_val = np.mean(image)
    result = image * (1 - 0.5 * snow_mask) + mean_val * 0.5 * snow_mask
    noise = np.random.normal(0, 5, image.shape)
    result = result + noise * snow_mask
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_snow_rgb(image, num_flakes):
    """Apply snow to RGB: white particles + contrast reduction."""
    h, w = image.shape[:2]
    snow_layer = np.zeros_like(image, dtype=np.float32)
    for _ in range(num_flakes):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        r = random.randint(1, 4)
        cv2.circle(snow_layer, (x, y), r, (255, 255, 255), -1)
    snow_layer = cv2.GaussianBlur(snow_layer, (5, 5), 0)
    result = cv2.addWeighted(image.astype(np.float32), 0.7, snow_layer, 0.3, 0)
    return np.clip(result, 0, 255).astype(np.uint8)

def process_split(split_name, coco_path, thermal_src, rgb_src, thermal_dst, rgb_dst, weather_type, severity):
    """Process a single split with given weather."""
    with open(coco_path) as f:
        coco = json.load(f)
    
    thermal_dst = Path(thermal_dst)
    rgb_dst = Path(rgb_dst)
    thermal_dst.mkdir(parents=True, exist_ok=True)
    rgb_dst.mkdir(parents=True, exist_ok=True)
    
    # Copy annotations
    new_coco = coco.copy()
    
    alpha = SEVERITIES['fog'][severity] if weather_type == 'fog' else None
    num_drops = SEVERITIES['rain'][severity] if weather_type == 'rain' else None
    num_flakes = SEVERITIES['snow'][severity] if weather_type == 'snow' else None
    
    print(f"Processing {split_name} - {weather_type} {severity}")
    
    for img_info in tqdm(coco['images']):
        rel_path = img_info['file_name']
        thermal_src_path = Path(thermal_src) / rel_path
        rgb_src_path = Path(rgb_src) / rel_path.replace('thermal', 'rgb') if 'thermal' in rel_path else Path(rgb_src) / rel_path
        
        # Handle thermal
        if thermal_src_path.exists():
            thermal_img = cv2.imread(str(thermal_src_path), cv2.IMREAD_GRAYSCALE)
            if thermal_img is not None:
                if weather_type == 'fog':
                    thermal_out = apply_fog_thermal(thermal_img, alpha)
                elif weather_type == 'rain':
                    thermal_out = apply_rain_thermal(thermal_img, num_drops)
                elif weather_type == 'snow':
                    thermal_out = apply_snow_thermal(thermal_img, num_flakes)
                else:
                    thermal_out = thermal_img
                
                thermal_out_path = thermal_dst / rel_path
                thermal_out_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(thermal_out_path), thermal_out)
        
        # Handle RGB
        if rgb_src_path.exists():
            rgb_img = cv2.imread(str(rgb_src_path))
            if rgb_img is not None:
                if weather_type == 'fog':
                    rgb_out = apply_fog_rgb(rgb_img, alpha)
                elif weather_type == 'rain':
                    rgb_out = apply_rain_rgb(rgb_img, num_drops)
                elif weather_type == 'snow':
                    rgb_out = apply_snow_rgb(rgb_img, num_flakes)
                else:
                    rgb_out = rgb_img
                
                rgb_out_path = rgb_dst / rel_path.replace('thermal', 'rgb') if 'thermal' in rel_path else rgb_dst / rel_path
                rgb_out_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(rgb_out_path), rgb_out)
    
    # Save new COCO
    new_coco_path = thermal_dst.parent / f"coco_{weather_type}_{severity}.json"
    with open(new_coco_path, 'w') as f:
        json.dump(new_coco, f)
    print(f"Saved annotations to {new_coco_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weather', choices=['fog', 'rain', 'snow'], required=True)
    parser.add_argument('--severity', choices=['light', 'medium', 'heavy'], required=True)
    parser.add_argument('--splits', nargs='+', default=['train', 'val'])
    args = parser.parse_args()
    
    base = Path('data/flir_adas_v2/FLIR_ADAS_v2')
    
    for split in args.splits:
        thermal_src = base / f'images_thermal_{split}' / 'data'
        rgb_src = base / f'images_rgb_{split}' / 'data'
        coco_path = base / f'images_thermal_{split}' / 'coco.json'
        
        thermal_dst = base / f'images_thermal_{split}_{args.weather}_{args.severity}' / 'data'
        rgb_dst = base / f'images_rgb_{split}_{args.weather}_{args.severity}' / 'data'
        
        process_split(split, coco_path, thermal_src, rgb_src, thermal_dst, rgb_dst, args.weather, args.severity)

if __name__ == '__main__':
    main()