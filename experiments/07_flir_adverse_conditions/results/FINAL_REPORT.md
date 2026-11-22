# FLIR adverse-condition screen and retraining — final results

Generated 2026-11-12T00:19:15.245158+00:00.

## Protocol

- Detector: YOLOv12s (`<repo-root>/models/pretrained/yolo12s.pt`), 40 epochs max, patience 10, batch 16, imgsz 640, seed 0.
- Baseline **CM**: trained on clear train **C_Tr** (<repo-root>/data/flir_adas_v2/yolo/train_3a/images/train_clear → 9,051 images).
- Clear test **C_T**: the matched 3,749-frame VTT split.
- Adverse test **A_T**: the same 3,749 frames transformed at **medium** severity.
- Selection: relative mAP50-95 drop `(C_T − A_T) / C_T` > 5 %.
- Mixed model **Mix_M**: same recipe trained on **C_Tr + A_Tr** (18,102 images). The remaining gap is `Mix(C_T) − Mix(A_T)`.

Baseline CM on C_T: **mAP50-95 = 0.14966**, **mAP50 = 0.27546** (3749 images).

## 1. Screen: is the clear model hurt?

| condition | family | CM C_T (50-95) | CM A_T (50-95) | abs drop | rel drop | selected |
|---|---|---:|---:|---:|---:|---|
| fog | cyclegan | 0.14966 | 0.14645 | 0.00321 | 2.14% | False |
| rain | cyclegan | 0.14966 | 0.14521 | 0.00445 | 2.97% | False |
| snow | cyclegan | 0.14966 | 0.12652 | 0.02314 | 15.46% | True |
| crossover | cyclegan | 0.14966 | 0.14301 | 0.00665 | 4.44% | False |
| sensor | cyclegan | 0.14966 | 0.09470 | 0.05496 | 36.72% | True |
| small_target | physics-only | 0.14966 | 0.12675 | 0.02291 | 15.31% | True |
| defocus | physics-only | 0.14966 | 0.05373 | 0.09593 | 64.10% | True |
| occlusion | physics-only | 0.14966 | 0.11775 | 0.03191 | 21.32% | True |

The same comparison under mAP50 (all 3,749 frames):

| condition | CM C_T | CM A_T | rel drop |
|---|---:|---:|---:|
| fog | 0.27546 | 0.27096 | 1.63% |
| rain | 0.27546 | 0.26998 | 1.99% |
| snow | 0.27546 | 0.23578 | 14.40% |
| crossover | 0.27546 | 0.26366 | 4.28% |
| sensor | 0.27546 | 0.18255 | 33.73% |
| small_target | 0.27546 | 0.23551 | 14.50% |
| defocus | 0.27546 | 0.11944 | 56.64% |
| occlusion | 0.27546 | 0.22428 | 18.58% |

Three conditions fall below the 5 % screen and are **not** retrained under this protocol: fog (2.14 %), rain (2.97 %) and crossover (4.44 %). They are reported as `skipped_below_threshold`, not as "harmless" — a mild physical simulation is not evidence that the real weather is safe.

## 2. Retraining selected conditions

| condition | selected | Mix C_T | Mix A_T | Mix gap | CM gap | gap reduction | Δ clear | Δ adverse |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fog | False | — | — | — | — | — | — | — |
| rain | False | — | — | — | — | — | — | — |
| snow | True | 0.16306 | 0.15806 | 0.0050 | 0.0231 | 78.4 % | 0.0134 | 0.0315 |
| crossover | False | — | — | — | — | — | — | — |
| sensor | True | 0.15057 | 0.12825 | 0.0223 | 0.0550 | 59.4 % | 0.0009 | 0.0335 |
| small_target | True | 0.14323 | 0.37717 | -0.2339 | 0.0229 | n/a (negative gap) | -0.0064 | 0.2504 |
| defocus | True | 0.14706 | 0.10070 | 0.0464 | 0.0959 | 51.7 % | -0.0026 | 0.0470 |
| occlusion | True | 0.14796 | 0.13521 | 0.0128 | 0.0319 | 60.0 % | -0.0017 | 0.0175 |

## 3. What happened

- **Snow**: CM loses 15.5 %; the mixed model cuts the gap from 0.0231 to 0.0050 (−78 %) and improves both clear and snow accuracy.
- **Sensor degradation**: the largest selected drop (36.7 %); the mixed model cuts the gap from 0.0550 to 0.0223 (−59 %) while preserving clear accuracy.
- **Defocus**: CM loses 64.1 %; the mixed model recovers 0.0470 of adverse mAP and cuts the gap from 0.0959 to 0.0464 (−52 %).
- **Static obstruction**: CM loses 21.3 %; the mixed model cuts the gap from 0.0319 to 0.0128 (−60 %), preserves clear accuracy (−0.0017) and improves adverse accuracy by 0.0175.
- **Small targets**: CM loses 15.3 %, but the mixed model is *better* on A_T (0.3772) than on C_T (0.1432), giving a negative gap (−0.2339). The 8×8 pixelation creates a strong, learnable block cue; the model specializes to it and gives up 0.0064 clear mAP. This is a real domain-cue effect, not a label or split error: the clear and small-target test labels are byte-identical (61,596 instances, same 9 classes). It should not be read as "small targets are easy".

## 4. Incidents and blockers

- The first `Mix_sensor_yolo12s` attempt was SIGKILLed (rc = −9) after ~2 minutes while RAM-caching 18,102 images under the 48 GB allocation. It was retried successfully with `cache=false` and 12 workers; the final sensor row is from that completed run. The failure remains in `protocol_state.json` as a historical entry, but is not an unresolved blocker.
- No unresolved blockers. All eval files report 3,749 images.

## 5. Caveats

- Single seed (0), single medium severity per condition. No variance across seeds or severities is measured.
- All conditions are simulations on 8-bit thermal JPEGs. FLIR has no real adverse weather, no depth and no radiometric temperature; fog uses a uniform transmission by default, crossover uses the local box-ring background as a proxy, and obstruction is procedural.
- Fog, rain and crossover were screened at medium severity only. The physical fog here is a uniform-transmission veil and is milder than the project's CycleGAN fog, whose B1 gap is about 17 % in mAP50-95; the below-threshold fog result is specific to this severity and implementation, not evidence that real fog is harmless.
- Training stopped by early stopping on some arms (snow 33, occlusion 30, small-target 38, sensor 39, defocus 39 of 40).
- The clear baseline is trained on 9,051 frames while the mixed arms see 18,102; both use the same recipe and epoch budget.

## 6. Artefacts

- Per-arm eval JSONs: `results/eval_*.json` (weights hash, image count, mAP50-95, mAP50, precision, recall, host, SLURM job id).
- Runs: `runs/CM_clear_yolo12s`, `runs/Mix_<condition>_yolo12s`.
- Generation provenance: `data/generated/<condition>/medium/generation.json`.
- Machine-readable state: `results/protocol_state.json`.
- Tabular export: `results/protocol_summary.csv`.
