# Assets that are intentionally not in git

`AGENTS.md` forbids committing `*.pt`, `runs/`, `results/` and `data/`, so the
following files travel out-of-band. They are required before stage `N0`; the
queue's `preflight` aborts with a clear message if any is missing.

| File | Size | Why |
|---|---:|---|
| `models/pretrained/yolo12s.pt` | 19 MB | student initialisation (N1, N4, N6) |
| `models/pretrained/yolo12n.pt` | 5.4 MB | student initialisation (N2, N3) |
| `experiments/09_graph_fusion/runs/e30_v2_100ep/weights/best.pt` | 30 MB | frozen E30 teacher |
| `experiments/09_graph_fusion/runs/e00_ir/weights/best.pt` | 19 MB | E00 IR-only latency/AP reference in profiling |
| `experiments/09_graph_fusion/results/gt/dv_v2_ir_test.json` | 22 MB | locked 8,980-image test GT |
| `experiments/09_graph_fusion/results/preds/e00_ir_test.json` | 37 MB | E00 reference predictions (bootstrap) |
| `experiments/09_graph_fusion/results/preds/e30_v2_100ep_test.json` | 27 MB | E30 reference predictions (bootstrap) |
| `experiments/09_graph_fusion/results/e00_ir_test_metrics.json` | 3 KB | E00 test AP for the figures |
| `experiments/09_graph_fusion/results/e30_v2_100ep_test_metrics.json` | 3 KB | E30 test AP for the figures |
| `experiments/09_graph_fusion/results/bootstrap_flagships_vs_e00.json` | 5 KB | E30 CI for the forest plot |
| `experiments/09_graph_fusion/runs/e00_ir/results.csv` | 30 KB | E00 val curve |
| `experiments/09_graph_fusion/runs/e30_v2_100ep/results.csv` | 30 KB | E30 val curve |

## Dataset — build it on the target node (recommended)

If `data/dronevehicle/raw` is present (the 14 GB release), do **not** copy the
5.5 GB YOLO tree. The queue builds it automatically:

```bash
PY=... DV_ROOT=<repo>/experiments/09_graph_fusion/data/dronevehicle_v2/yolo \
RAW_DV=<repo>/data/dronevehicle/raw bash scripts/run_node2_queue.sh
```

`scripts/build_dv_from_raw.py` uses the committed split manifest
(`assets/dv_v2_splits.json`, exact 17,962 / 1,469 / 8,980 stems), the 09 XML
parser (`*labelr` / `*label`) and the verified crop + PIL JPEG quality 92. On
the development node it reproduces the 09 tree **bit-exactly** (240/240 sampled
images array-equal, labels byte-equal).

To build manually:

```bash
python scripts/build_dv_from_raw.py --raw <repo>/data/dronevehicle/raw \
  --out <repo>/experiments/09_graph_fusion/data/dronevehicle_v2
```

Alternatively copy the prebuilt `experiments/09_graph_fusion/data/dronevehicle_v2/yolo`
(5.5 GB) and set `DV_ROOT` to it.

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
