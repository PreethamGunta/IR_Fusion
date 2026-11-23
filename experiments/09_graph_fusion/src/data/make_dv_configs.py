#!/usr/bin/env python3
"""Write YOLO data YAMLs for the built DroneVehicle v2 tree."""
from pathlib import Path

EXP = Path(__file__).resolve().parents[2]
ROOT = EXP / "data" / "dronevehicle_v2" / "yolo"
NAMES = ["car", "truck", "bus", "van", "freight_car"]
CONFIGS = EXP / "configs"

for modality in ("ir", "rgb"):
    (CONFIGS / f"dronevehicle_v2_{modality}.yaml").write_text(
        f"path: {ROOT / modality}\ntrain: images/train\nval: images/val\ntest: images/test\n\n"
        f"nc: {len(NAMES)}\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES))
    )
(CONFIGS / "dronevehicle_v2_paired.yaml").write_text(
    f"path: {ROOT / 'ir'}\ntrain: images/train\nval: images/val\ntest: images/test\n\n"
    f"nc: {len(NAMES)}\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES))
    + "\nchannels: 6\n"
)
print("wrote", [p.name for p in CONFIGS.glob("dronevehicle_v2_*.yaml")])
