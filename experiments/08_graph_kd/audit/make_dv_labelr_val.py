"""Build an audit-only DroneVehicle IR val split using vallabelr labels.

Quantifies the train/val annotation-policy mismatch without touching the
experiment dataset. Writes under audit/ only.
"""
from __future__ import annotations

import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from graphkd.data.build_dronevehicle import CLASS_NAMES, _parse

REPO = Path('/home2/home/preetham_g/gate_ir_release')
SRC = REPO / 'data' / 'dronevehicle' / 'raw' / 'extracted' / 'val' / 'val'
OUT = REPO / 'experiments' / '08_graph_kd' / 'audit' / 'dv_ir_labelr_val'
IMG_SRC = REPO / 'data' / 'dronevehicle' / 'yolo' / 'ir' / 'images' / 'val'
LAB_SRC = SRC / 'vallabelr'

OUT.mkdir(parents=True, exist_ok=True)
labels = OUT / 'labels'
labels.mkdir(exist_ok=True)
imgs = OUT / 'images'
if imgs.is_symlink() or imgs.exists():
    if imgs.is_symlink():
        imgs.unlink()
    else:
        shutil.rmtree(imgs)
imgs.mkdir(parents=True)
# Hardlink each image so its path stays under audit/ (a directory symlink is
# resolved by Ultralytics back to the original folder, which silently reuses
# the visible-label cache).
for ip in sorted(IMG_SRC.glob('*.jpg')):
    os.link(ip, imgs / ip.name)
for stale in (OUT / 'labels').glob('*.cache'):
    stale.unlink()

n = 0
for ip in sorted(IMG_SRC.glob('*.jpg')):
    lp = LAB_SRC / f'{ip.stem}.xml'
    objs = _parse(lp.read_bytes()) if lp.exists() else []
    lines = [f"{c} {(x1+x2)/2/640:.6f} {(y1+y2)/2/512:.6f} {(x2-x1)/640:.6f} {(y2-y1)/512:.6f}"
             for c, x1, y1, x2, y2 in objs]
    (labels / f'{ip.stem}.txt').write_text('\n'.join(lines) + ('\n' if lines else ''))
    n += 1
print('wrote', n, 'labels')

yaml = OUT.parent / 'dv_ir_labelr_val.yaml'
yaml.write_text(
    f"path: {OUT}\ntrain: images\nval: images\ntest: images\n\n"
    f"nc: {len(CLASS_NAMES)}\nnames:\n" + ''.join(f"  {i}: {c}\n" for i, c in enumerate(CLASS_NAMES))
)
print(yaml.read_text())
