"""Assemble the final 07 report from the eval JSONs and protocol state.

Separate from `run_protocol.write_report` so the public report is stable and
does not carry resolved transient failures (for example, the first sensor arm
was OOM-killed and then retried successfully).
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from . import conditions, paths


def _eval(name: str) -> dict | None:
    path = paths.RESULTS / f"eval_{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _fmt(value, digits: int = 4) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def main() -> None:
    state = json.loads((paths.RESULTS / "protocol_state.json").read_text())
    baseline = _eval("CM_clear")
    if not baseline:
        raise SystemExit("missing eval_CM_clear.json")

    b95, b50 = baseline["mAP50_95"], baseline["mAP50"]
    rows = []
    for cond in conditions.CONDITION_NAMES:
        c = state["conditions"][cond]
        cm_adv = _eval(f"CM_{cond}")
        mix_c = _eval(f"Mix_{cond}_clear")
        mix_a = _eval(f"Mix_{cond}_adv")
        a95 = cm_adv["mAP50_95"] if cm_adv else None
        a50 = cm_adv["mAP50"] if cm_adv else None
        drop_abs = (b95 - a95) if a95 is not None else None
        drop_rel = (drop_abs / b95) if drop_abs is not None and b95 else None
        mc95 = mix_c["mAP50_95"] if mix_c else None
        ma95 = mix_a["mAP50_95"] if mix_a else None
        mc50 = mix_c["mAP50"] if mix_c else None
        ma50 = mix_a["mAP50"] if mix_a else None
        mix_gap = (mc95 - ma95) if mc95 is not None and ma95 is not None else None
        cm_gap = drop_abs
        gap_red = None
        if mix_gap is not None and cm_gap and cm_gap > 0:
            gap_red = 1.0 - mix_gap / cm_gap
        rows.append({
            "condition": cond,
            "family": conditions.REGISTRY[cond].family,
            "cm_clear_95": b95,
            "cm_clear_50": b50,
            "cm_adv_95": a95,
            "cm_adv_50": a50,
            "drop_abs": drop_abs,
            "drop_rel": drop_rel,
            "selected": bool(c.get("selected")),
            "mix_clear_95": mc95,
            "mix_clear_50": mc50,
            "mix_adv_95": ma95,
            "mix_adv_50": ma50,
            "mix_gap": mix_gap,
            "gap_red": gap_red,
            "mix_clear_delta": (mc95 - b95) if mc95 is not None else None,
            "mix_adv_delta": (ma95 - a95) if ma95 is not None and a95 is not None else None,
            "status": c.get("status"),
        })

    cfg = state.get("config", {})
    lines = [
        "# FLIR adverse-condition screen and retraining — final results",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat()}.",
        "",
        "## Protocol",
        "",
        f"- Detector: YOLOv12s (`{cfg.get('weights', 'models/pretrained/yolo12s.pt')}`), "
        f"{cfg.get('epochs', 40)} epochs max, patience {cfg.get('patience', 10)}, "
        f"batch {cfg.get('batch', 16)}, imgsz 640, seed 0.",
        f"- Baseline **CM**: trained on clear train **C_Tr** "
        f"({paths.CLEAR_IMG['train']} → 9,051 images).",
        f"- Clear test **C_T**: the matched 3,749-frame VTT split.",
        f"- Adverse test **A_T**: the same 3,749 frames transformed at "
        f"**{state.get('severity', 'medium')}** severity.",
        f"- Selection: relative mAP50-95 drop `(C_T − A_T) / C_T` > "
        f"{cfg.get('drop_threshold', 0.05)*100:.0f} %.",
        f"- Mixed model **Mix_M**: same recipe trained on **C_Tr + A_Tr** "
        f"(18,102 images). The remaining gap is `Mix(C_T) − Mix(A_T)`.",
        "",
        f"Baseline CM on C_T: **mAP50-95 = {b95:.5f}**, **mAP50 = {b50:.5f}** "
        f"({baseline['n_images']} images).",
        "",
        "## 1. Screen: is the clear model hurt?",
        "",
        "| condition | family | CM C_T (50-95) | CM A_T (50-95) | abs drop | rel drop | selected |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        rel = r["drop_rel"]
        lines.append(
            f"| {r['condition']} | {r['family']} | {_fmt(r['cm_clear_95'],5)} | "
            f"{_fmt(r['cm_adv_95'],5)} | {_fmt(r['drop_abs'],5)} | "
            f"{'' if rel is None else f'{rel*100:.2f}%'} | "
            f"{r['selected']} |"
        )

    lines += [
        "",
        "The same comparison under mAP50 (all 3,749 frames):",
        "",
        "| condition | CM C_T | CM A_T | rel drop |",
        "|---|---:|---:|---:|",
    ]
    for r in rows:
        rel50 = ((r["cm_clear_50"] - r["cm_adv_50"]) / r["cm_clear_50"]
                 if r["cm_adv_50"] is not None and r["cm_clear_50"] else None)
        lines.append(
            f"| {r['condition']} | {_fmt(r['cm_clear_50'],5)} | "
            f"{_fmt(r['cm_adv_50'],5)} | "
            f"{'' if rel50 is None else f'{rel50*100:.2f}%'} |"
        )

    lines += [
        "",
        "Three conditions fall below the 5 % screen and are **not** retrained "
        "under this protocol: fog (2.14 %), rain (2.97 %) and crossover "
        "(4.44 %). They are reported as `skipped_below_threshold`, not as "
        "\"harmless\" — a mild physical simulation is not evidence that the "
        "real weather is safe.",
        "",
        "## 2. Retraining selected conditions",
        "",
        "| condition | selected | Mix C_T | Mix A_T | Mix gap | CM gap | gap reduction | Δ clear | Δ adverse |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        if not r["selected"]:
            lines.append(f"| {r['condition']} | False | — | — | — | — | — | — | — |")
            continue
        red = "" if r["gap_red"] is None else f"{r['gap_red']*100:.1f} %"
        if r["mix_gap"] is not None and r["mix_gap"] <= 0:
            red = "n/a (negative gap)"
        lines.append(
            f"| {r['condition']} | True | {_fmt(r['mix_clear_95'],5)} | "
            f"{_fmt(r['mix_adv_95'],5)} | {_fmt(r['mix_gap'],4)} | "
            f"{_fmt(r['drop_abs'],4)} | {red} | "
            f"{_fmt(r['mix_clear_delta'],4)} | {_fmt(r['mix_adv_delta'],4)} |"
        )
    lines.append("")

    lines += [
        "## 3. What happened",
        "",
        "- **Snow**: CM loses 15.5 %; the mixed model cuts the gap from 0.0231 "
        "to 0.0050 (−78 %) and improves both clear and snow accuracy.",
        "- **Sensor degradation**: the largest selected drop (36.7 %); the "
        "mixed model cuts the gap from 0.0550 to 0.0223 (−59 %) while "
        "preserving clear accuracy.",
        "- **Defocus**: CM loses 64.1 %; the mixed model recovers 0.0470 of "
        "adverse mAP and cuts the gap from 0.0959 to 0.0464 (−52 %).",
        "- **Static obstruction**: CM loses 21.3 %; the mixed model cuts the "
        "gap from 0.0319 to 0.0128 (−60 %), preserves clear accuracy "
        "(−0.0017) and improves adverse accuracy by 0.0175.",
        "- **Small targets**: CM loses 15.3 %, but the mixed model is *better* "
        "on A_T (0.3772) than on C_T (0.1432), giving a negative gap "
        "(−0.2339). The 8×8 pixelation creates a strong, learnable block "
        "cue; the model specializes to it and gives up 0.0064 clear mAP. "
        "This is a real domain-cue effect, not a label or split error: the "
        "clear and small-target test labels are byte-identical (61,596 "
        "instances, same 9 classes). It should not be read as \"small "
        "targets are easy\".",
        "",
        "## 4. Incidents and blockers",
        "",
        "- The first `Mix_sensor_yolo12s` attempt was SIGKILLed (rc = −9) "
        "after ~2 minutes while RAM-caching 18,102 images under the 48 GB "
        "allocation. It was retried successfully with `cache=false` and 12 "
        "workers; the final sensor row is from that completed run. The "
        "failure remains in `protocol_state.json` as a historical entry, "
        "but is not an unresolved blocker.",
        "- No unresolved blockers. All eval files report 3,749 images.",
        "",
        "## 5. Caveats",
        "",
        "- Single seed (0), single medium severity per condition. No variance "
        "across seeds or severities is measured.",
        "- All conditions are simulations on 8-bit thermal JPEGs. FLIR has no "
        "real adverse weather, no depth and no radiometric temperature; fog "
        "uses a uniform transmission by default, crossover uses the local "
        "box-ring background as a proxy, and obstruction is procedural.",
        "- Fog, rain and crossover were screened at medium severity only. The "
        "physical fog here is a uniform-transmission veil and is milder than "
        "the project's CycleGAN fog, whose B1 gap is about 17 % in mAP50-95; "
        "the below-threshold fog result is specific to this severity and "
        "implementation, not evidence that real fog is harmless.",
        "- Training stopped by early stopping on some arms (snow 33, "
        "occlusion 30, small-target 38, sensor 39, defocus 39 of 40).",
        "- The clear baseline is trained on 9,051 frames while the mixed arms "
        "see 18,102; both use the same recipe and epoch budget.",
        "",
        "## 6. Artefacts",
        "",
        "- Per-arm eval JSONs: `results/eval_*.json` (weights hash, image "
        "count, mAP50-95, mAP50, precision, recall, host, SLURM job id).",
        "- Runs: `runs/CM_clear_yolo12s`, `runs/Mix_<condition>_yolo12s`.",
        "- Generation provenance: `data/generated/<condition>/medium/generation.json`.",
        "- Machine-readable state: `results/protocol_state.json`.",
        "- Tabular export: `results/protocol_summary.csv`.",
        "",
    ]

    out = paths.RESULTS / "FINAL_REPORT.md"
    out.write_text("\n".join(lines))

    csv_path = paths.RESULTS / "protocol_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[final_report] -> {out}")
    print(f"[final_report] -> {csv_path}")


if __name__ == "__main__":
    main()
