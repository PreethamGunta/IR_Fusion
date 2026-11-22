# Graph-KD results — aligned FLIR paired protocol

Test split: 609 paired frames. AP is COCO mAP50-95. Deltas are relative to the GT-only YOLO12n student (`a0_gt`).

| Arm | AP | AP50 | AP75 | APs | APm | APl | ΔAP | ECE | Brier | Params | ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| a0_gt | 0.2518 | 0.4955 | 0.2206 | 0.1662 | 0.2986 | 0.4269 |  | 0.1257 | 0.0902 | 2.57M | 65.89 |
| a2_ir_control | 0.2498 | 0.4911 | 0.2190 | 0.1679 | 0.2928 | 0.4528 | -0.0020 | 0.1004 | 0.0984 | 2.57M | 72.40 |
| a1_pred_ir | 0.2409 | 0.4821 | 0.1990 | 0.1551 | 0.2858 | 0.4508 | -0.0110 | 0.1160 | 0.1095 | 2.57M | 64.31 |
| a3_node_edge_ir | 0.2586 | 0.5085 | 0.2223 | 0.1691 | 0.3031 | 0.5205 | +0.0068 | 0.0995 | 0.1027 | 2.57M | 70.11 |
| a5_fgw_ir | 0.2569 | 0.4990 | 0.2228 | 0.1705 | 0.3051 | 0.4484 | +0.0051 | 0.1020 | 0.1062 | 2.57M | 65.66 |

## Best student: `a3_node_edge_ir`

| Class | AP |
|---|---:|
| car | 0.5087 |
| person | 0.3183 |
| bicycle | 0.2025 |
| dog | 0.0050 |

## Per-arm KD loss terms (last epoch)

| Arm | last kd terms |
|---|---|
| a2_ir_control | kd_obj=0.027, kd_cls=2.309, kd_box=0.288, kd_feat=0.000 |
| a1_pred_ir | kd_pred=2.041 |
| a3_node_edge_ir | kd_box=0.293, kd_feat=0.000, kd_cls=2.307, kd_edge=0.035, kd_obj=0.029 |
| a5_fgw_ir | kd_struct=0.012, kd_feat=0.000, kd_obj=0.040, kd_box=0.384, kd_edge=0.037, kd_cls=2.223 |

## Provenance

- **a0_gt**: weights `<repo-root>/experiments/08_graph_kd/runs/a0_gt/weights/best.pt` sha256 `cf3236e20cbb92db…`, split `5c8e38f1a18b4d95`, n=609
- **a2_ir_control**: weights `<repo-root>/experiments/08_graph_kd/runs/a2_ir_control/weights/best.pt` sha256 `1ef6ec2bbb67f4e3…`, split `5c8e38f1a18b4d95`, n=609
- **a1_pred_ir**: weights `<repo-root>/experiments/08_graph_kd/runs/a1_pred_ir/weights/best.pt` sha256 `4a84659e8db43505…`, split `5c8e38f1a18b4d95`, n=609
- **a3_node_edge_ir**: weights `<repo-root>/experiments/08_graph_kd/runs/a3_node_edge_ir/weights/best.pt` sha256 `fe676c755f7ed530…`, split `5c8e38f1a18b4d95`, n=609
- **a5_fgw_ir**: weights `<repo-root>/experiments/08_graph_kd/runs/a5_fgw_ir/weights/best.pt` sha256 `3405068e2c386cf3…`, split `5c8e38f1a18b4d95`, n=609
