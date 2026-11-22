# FLIR adverse-condition screen and retraining

Generated 2026-11-12T00:15:55.660402+00:00

Selection rule: relative drop of CM on A_T vs C_T in mAP50-95 > 5 %.

| condition | family | CM C_T | CM A_T | drop | selected | Mix C_T | Mix A_T | Mix gap | status |
|---|---|---:|---:|---:|---|---:|---:|---:|---|
| fog | cyclegan | 0.14966 | 0.14645 | 2.14% | False |  |  |  | skipped_below_threshold |
| rain | cyclegan | 0.14966 | 0.14521 | 2.97% | False |  |  |  | skipped_below_threshold |
| snow | cyclegan | 0.14966 | 0.12652 | 15.46% | True | 0.16306 | 0.15806 | 0.0050 | done |
| crossover | cyclegan | 0.14966 | 0.14301 | 4.44% | False |  |  |  | skipped_below_threshold |
| sensor | cyclegan | 0.14966 | 0.0947 | 36.72% | True | 0.15057 | 0.12825 | 0.0223 | done |
| small_target | physics-only | 0.14966 | 0.12675 | 15.31% | True | 0.14323 | 0.37717 | -0.2339 | done |
| defocus | physics-only | 0.14966 | 0.05373 | 64.10% | True | 0.14706 | 0.1007 | 0.0464 | done |
| occlusion | physics-only | 0.14966 | 0.11775 | 21.32% | True | 0.14796 | 0.13521 | 0.0128 | done |

Baseline CM: `CM_clear_yolo12s`
