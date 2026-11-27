# Assets that are intentionally not in git

`AGENTS.md` forbids committing `*.pt`, `runs/`, `results/` and `data/`, so the
following files travel out-of-band. They are required before stage `N0`; the
queue's `preflight` aborts with a clear message if any is missing.

| File | Size | Why |
|---|---:|---|
| `models/pretrained/yolo12s.pt` | 19 MB | student initialisation (N1, N4, N6) |
| `models/pretrained/yolo12n.pt` | 5.4 MB | student initialisation (N2, N3) |
| `experiments/09_graph_fusion/runs/e30_v2_100ep/weights/best.pt` | 30 MB | frozen E30 teacher |
| `experiments/09_graph_fusion/results/gt/dv_v2_ir_test.json` | 22 MB | locked 8,980-image test GT |
| `experiments/09_graph_fusion/results/preds/e00_ir_test.json` | 37 MB | E00 reference predictions (bootstrap) |
| `experiments/09_graph_fusion/results/preds/e30_v2_100ep_test.json` | 27 MB | E30 reference predictions (bootstrap) |

Dataset: DroneVehicle v2 YOLO trees (`ir/`, `rgb/`) at

```
experiments/09_graph_fusion/data/dronevehicle_v2/yolo
```

or wherever `DV_ROOT` points. `scripts/make_data_configs.py` derives the three
train/val/test yamls from that root at queue start-up, so no absolute path is
baked into the repository.

## One command from the development node

```bash
bash experiments/10_ir_distillation/scripts/sync_assets.sh --check
bash experiments/10_ir_distillation/scripts/sync_assets.sh \
  --to yash_gupta@<node>:/abs/path/to/IR_Fusion
```

`rsync -R` preserves the repo-relative paths. Nothing above is committed; the
repository itself stays free of multi-hundred-MB binaries.

## Start on the target node

```bash
git pull
cd experiments/10_ir_distillation
PY=/path/to/env/bin/python DV_ROOT=/path/to/dronevehicle_v2/yolo \
  bash scripts/run_node2_queue.sh
```
