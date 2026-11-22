# GATE-IR on FLIR — results report

## Phase 1 — CycleGAN 60-epoch filter screen (YOLOv8s)

| arm | clear mAP50-95 | fog mAP50-95 | gap | n clr/fog |
|---|---:|---:|---:|---:|
| B1 (no filter) | 0.1646 | 0.1413 | 0.0233 | 3749/3749 |
| G1 gamma | 0.1588 | 0.1369 | 0.0219 | 3749/3749 |
| G1 clahe | 0.1636 | 0.1335 | 0.0300 | 3749/3749 |
| G1 dcp | 0.1651 | 0.1394 | 0.0257 | 3749/3749 |
| G1 unet | 0.1676 | 0.1524 | 0.0152 | 3749/3749 |
| G1 gamma_clahe | 0.1610 | 0.1384 | 0.0227 | 3749/3749 |

## Phase 1 — main comparison, 40-epoch YOLOv12s (winner: unet)

| arm | clear mAP50-95 | fog mAP50-95 | gap | n clr/fog |
|---|---:|---:|---:|---:|
| B1 (no filter) | 0.1464 | 0.1234 | 0.0230 | 3749/3749 |
| G1 unet | 0.1518 | 0.1433 | 0.0084 | 3749/3749 |

## Phase 1 — 100-epoch YOLOv12s confirmation

| arm | clear mAP50-95 | fog mAP50-95 | gap | n clr/fog |
|---|---:|---:|---:|---:|
| B1 (no filter) | 0.1525 | 0.1266 | 0.0258 | 3749/3749 |
| G1 U-Net | 0.1683 | 0.1535 | 0.0148 | 3749/3749 |

## Phase 2 — CycleDiff-75 (filter: unet)

| arm | clear mAP50-95 | fog mAP50-95 | gap | n clr/fog |
|---|---:|---:|---:|---:|
| B1 (no filter) | 0.1664 | 0.1619 | 0.0046 | 3749/3749 |
| G1 unet (CycleGAN-trained) | 0.1593 | 0.1511 | 0.0082 | 3749/3749 |
| G1 U-Net trained on CycleDiff-75 | 0.1638 | 0.1560 | 0.0078 | 3749/3749 |

## Phase 2 — 100-epoch YOLOv12s confirmation on CycleDiff-75

| arm | clear mAP50-95 | fog mAP50-95 | gap | n clr/fog |
|---|---:|---:|---:|---:|
| B1 (no filter) | 0.1491 | 0.1495 | -0.0004 | 3749/3749 |
| G1 U-Net trained on CycleDiff-75 (stopped at epoch 27, resumable) | — | — | — | —/— |

## Phase 3 — ablations

| arm | clear mAP50-95 | fog mAP50-95 | gap |
|---|---:|---:|---:|
| B0 clear-only (CycleGAN fog) | 0.1540 | 0.0925 | 0.0615 | 3749/3749 |
| B0 clear-only (CycleDiff fog) | 0.1540 | 0.1294 | 0.0246 | 3749/3749 |
| B2 fog-only (CycleGAN) | — | 0.1313 | — | —/3749 |
| B2 fog-only (CycleDiff) | — | 0.1616 | — | —/3749 |
| B3 always-on gamma | — | — | — | —/— |
| B3 always-on clahe | — | — | — | —/— |
| B3 always-on dcp | — | — | — | —/— |
| B3 always-on unet | 0.1592 | 0.1510 | 0.0082 | 3749/3749 |
| B3 always-on gamma_clahe | — | — | — | —/— |
| B3 always-on unet (CycleDiff) | — | — | — | —/— |

## Gate metrics

### Gate cyclegan

| test | acc | F1 | FNR fog→clear | FPR clear→fog |
|---|---:|---:|---:|---:|
| vtt_clear | 0.9141 | 0.0000 | 0.0000 | 0.0859 |
| vtt_fog | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| vtt_fog_cyclediff75 | 0.6612 | 0.7961 | 0.3388 | 0.0000 |

