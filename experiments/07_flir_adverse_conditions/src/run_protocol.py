"""Run the screen -> retrain protocol end to end, resumably.

Protocol (severity fixed by ``--severity``, default ``medium``):

1. Train ``CM`` on clear train ``C_Tr``; evaluate on clear test ``C_T``.
2. For every condition, generate the adverse test set ``A_T`` and evaluate
   ``CM`` on it.  If the relative drop in mAP50-95 exceeds ``--drop-threshold``
   (default 5 %), the condition is selected.
3. For each selected condition, generate the adverse train set ``A_Tr``, train
   ``Mix_M`` on ``C_Tr + A_Tr``, and evaluate it on both ``C_T`` and ``A_T``.
   Report the remaining clear-minus-adverse gap.

Every stage writes a state file first, so an allocation that expires mid-run is
resumed by relaunching the orchestrator; existing images, YAMLs, checkpoints
and eval JSONs are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import conditions, paths, severity

SEV = "medium"
CM_NAME = "CM_clear_yolo12s"
STATE_FILE = paths.RESULTS / "protocol_state.json"
SUMMARY_FILE = paths.RESULTS / "protocol_summary.json"
REPORT_FILE = paths.RESULTS / "protocol_report.md"
DONE_FILE = paths.RESULTS / "PIPELINE_DONE"
STOP_FILE = paths.RESULTS / "PIPELINE_STOPPED"
WAIT_FILE = paths.RESULTS / "WAIT_FOR_ALLOCATION"

TERMINAL = {"done", "skipped_below_threshold", "blocked"}


# ── state ───────────────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except json.JSONDecodeError:
            pass
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": {},
        "conditions": {c: {"status": "pending", "blocker": None} for c in conditions.CONDITION_NAMES},
        "baseline": {"status": "pending", "blocker": None},
        "blockers": [],
    }


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def save_state(state: dict) -> None:
    state["updated_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_write(STATE_FILE, json.dumps(state, indent=2))


def blocker(state: dict, scope: str, message: str) -> None:
    entry = {"scope": scope, "message": message,
             "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    state.setdefault("blockers", []).append(entry)
    if scope == "baseline":
        state["baseline"]["blocker"] = message
    elif scope in state["conditions"]:
        state["conditions"][scope]["blocker"] = message
    save_state(state)


# ── subprocess helpers ──────────────────────────────────────────────────────

def _run(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[protocol] $ {' '.join(cmd)}", flush=True)
    with open(log_path, "a") as f:
        f.write(f"\n=== {datetime.now(timezone.utc).isoformat()} ===\n")
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode


def _py(*args: str) -> list[str]:
    return [sys.executable, "-m", *args]


def _count_images(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for p in directory.iterdir() if p.suffix.lower() in paths.IMG_EXTS)


def _json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _epochs_done(run_dir: Path) -> int:
    path = run_dir / "results.csv"
    if not path.exists():
        return 0
    try:
        import csv
        with open(path) as f:
            return sum(1 for _ in csv.DictReader(f))
    except OSError:
        return 0


# ── time budget ─────────────────────────────────────────────────────────────

def _parse_seconds(value: str) -> int:
    days = 0
    if "-" in value:
        d, value = value.split("-", 1)
        days = int(d)
    parts = [int(p) for p in value.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts
    return days * 86400 + h * 3600 + m * 60 + s


def remaining_seconds() -> float:
    """Seconds left in the current SLURM allocation (inf outside SLURM)."""
    jobid = os.environ.get("SLURM_JOB_ID")
    if not jobid:
        return float("inf")
    try:
        out = subprocess.run(["scontrol", "show", "job", jobid],
                             capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return float("inf")
    limit = runtime = None
    for token in out.split():
        if token.startswith("TimeLimit="):
            limit = _parse_seconds(token.split("=", 1)[1])
        elif token.startswith("RunTime="):
            runtime = _parse_seconds(token.split("=", 1)[1])
    if limit is None or runtime is None:
        return float("inf")
    return limit - runtime


class OutOfTime(RuntimeError):
    pass


def require_time(minutes: int) -> None:
    left = remaining_seconds() / 60.0
    if left < minutes:
        raise OutOfTime(f"only {left:.1f} min left; need {minutes}")


# ── pipeline stages ─────────────────────────────────────────────────────────

def build_yaml(cond: str, state: dict) -> None:
    require_time(5)
    rc = _run(_py("src.build_dataset", "--condition", cond, "--severity", SEV),
              paths.LOGS / f"build_{cond}.log")
    if rc != 0:
        raise RuntimeError(f"build_dataset failed for {cond} (rc={rc})")
    state["stages"] = state.get("stages", {})
    state["stages"][f"yaml_{cond}"] = True
    save_state(state)


def ensure_split(cond: str, split: str, state: dict, workers: int) -> None:
    src = paths.CLEAR_IMG[split]
    dst = paths.condition_dir(cond, SEV, split) / "images"
    expected = _count_images(src)
    have = _count_images(dst)
    if have >= expected and expected > 0:
        return
    require_time(10)
    print(f"[protocol] generating {cond}/{SEV}/{split}: {have}/{expected}", flush=True)
    rc = _run(_py("src.generate", "--condition", cond, "--severity", SEV,
                  "--splits", split, "--workers", str(workers)),
              paths.LOGS / f"generate_{cond}_{split}.log")
    have = _count_images(dst)
    if rc != 0 or have < expected:
        raise RuntimeError(f"generation incomplete for {cond}/{split}: {have}/{expected} (rc={rc})")


def train_arm(name: str, data: Path, args, state: dict, scope: str) -> None:
    run_dir = paths.RUNS / name
    if (run_dir / "TRAIN_DONE").exists():
        return
    if _epochs_done(run_dir) >= args.epochs:
        (run_dir / "TRAIN_DONE").write_text("recovered\n")
        return
    require_time(args.min_minutes)
    cmd = _py("src.train_condition", "--data", str(data), "--name", name,
              "--epochs", str(args.epochs), "--patience", str(args.patience),
              "--batch", str(args.batch), "--workers", str(args.workers),
              "--cache", args.cache)
    if (run_dir / "weights" / "last.pt").exists():
        cmd.append("--resume")
    rc = _run(cmd, paths.LOGS / f"train_{name}.log")
    if rc != 0 or not (run_dir / "TRAIN_DONE").exists():
        raise RuntimeError(f"training failed: {name} (rc={rc})")
    save_state(state)


def eval_arm(weights: Path, data: Path, name: str, batch: int) -> dict:
    out = paths.RESULTS / f"eval_{name}.json"
    cached = _json(out)
    if cached and cached.get("mAP50_95") is not None:
        return cached
    require_time(10)
    rc = _run(_py("src.eval_condition", "--weights", str(weights), "--data", str(data),
                  "--name", name, "--batch", str(batch)),
              paths.LOGS / f"eval_{name}.log")
    result = _json(out)
    if rc != 0 or not result:
        raise RuntimeError(f"evaluation failed: {name} (rc={rc})")
    return result


def _pick(result: dict) -> dict:
    return {
        "mAP50_95": result.get("mAP50_95"),
        "mAP50": result.get("mAP50"),
        "precision": result.get("precision"),
        "recall": result.get("recall"),
        "n_images": result.get("n_images"),
        "json": result.get("name"),
    }


def write_report(state: dict) -> None:
    rows = []
    for cond in conditions.CONDITION_NAMES:
        c = state["conditions"].get(cond, {})
        cm_c, cm_a = c.get("cm_clear", {}), c.get("cm_adverse", {})
        mix_c, mix_a = c.get("mix_clear", {}), c.get("mix_adverse", {})
        drop = c.get("drop_rel")
        gap = c.get("mix_gap")
        rows.append(
            f"| {cond} | {conditions.REGISTRY[cond].family} | "
            f"{cm_c.get('mAP50_95', '')} | {cm_a.get('mAP50_95', '')} | "
            f"{'' if drop is None else f'{drop*100:.2f}%'} | {c.get('selected', '')} | "
            f"{mix_c.get('mAP50_95', '')} | {mix_a.get('mAP50_95', '')} | "
            f"{'' if gap is None else f'{gap:.4f}'} | {c.get('status', '')} |"
        )
    lines = [
        "# FLIR adverse-condition screen and retraining",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat()}",
        "",
        "Selection rule: relative drop of CM on A_T vs C_T in mAP50-95 > 5 %.",
        "",
        "| condition | family | CM C_T | CM A_T | drop | selected | Mix C_T | Mix A_T | Mix gap | status |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|---|",
        *rows,
        "",
        "Baseline CM: `" + str(state["baseline"].get("run", CM_NAME)) + "`",
        "",
    ]
    bad = [b for b in state.get("blockers", [])
           if (state["baseline"].get("status") == "blocked" if b["scope"] == "baseline"
               else state["conditions"].get(b["scope"], {}).get("status") == "blocked")]
    if bad:
        lines += ["## Blockers", ""]
        lines += [f"- **{b['scope']}**: {b['message']}" for b in bad]
        lines.append("")
    REPORT_FILE.write_text("\n".join(lines))
    _atomic_write(SUMMARY_FILE, json.dumps(state, indent=2))


def all_terminal(state: dict) -> bool:
    if state["baseline"]["status"] != "done":
        return False
    return all(c.get("status") in TERMINAL for c in state["conditions"].values())


# ── main ────────────────────────────────────────────────────────────────────

def main() -> int:
    global SEV
    ap = argparse.ArgumentParser(description="Run the 07 screen/retrain protocol")
    ap.add_argument("--condition", default="all", help="condition, comma list, or all")
    ap.add_argument("--severity", default=SEV, choices=list(severity.SEVERITIES))
    ap.add_argument("--drop-threshold", type=float, default=0.05,
                    help="relative mAP50-95 drop required to retrain")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--eval-batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--cache", default="ram", choices=["ram", "disk", "false"])
    ap.add_argument("--weights", default=str(paths.REPO_ROOT / "models" / "pretrained" / "yolo12s.pt"))
    ap.add_argument("--min-minutes", type=int, default=20,
                    help="do not start a training arm with less than this left")
    ap.add_argument("--no-retrain", action="store_true",
                    help="screen only; do not generate train sets or train Mix_M")
    args = ap.parse_args()

    SEV = args.severity
    paths.ensure_dirs()

    conds = conditions.CONDITION_NAMES if args.condition == "all" else [
        c.strip() for c in args.condition.split(",") if c.strip()
    ]
    bad = [c for c in conds if c not in conditions.REGISTRY]
    if bad:
        raise SystemExit(f"unknown conditions {bad}")

    state = load_state()
    state["config"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    state["severity"] = SEV
    save_state(state)

    try:
        # 0. clear YAMLs and CM training/eval
        try:
            build_yaml("clear", state)
        except OutOfTime:
            raise
        except Exception as exc:
            state["baseline"]["status"] = "blocked"
            blocker(state, "baseline", f"clear YAML: {exc}")
            write_report(state)
            STOP_FILE.write_text(f"baseline blocked: {exc}\n")
            return 1

        # Generate the adverse test split for every condition before any
        # training. A condition that fails here is recorded and skipped.
        for cond in conds:
            try:
                ensure_split(cond, "test", state, args.workers)
                build_yaml(cond, state)
            except OutOfTime:
                raise
            except Exception as exc:
                state["conditions"][cond]["status"] = "blocked"
                blocker(state, cond, f"test generation: {exc}")
                write_report(state)

        try:
            cm_weights = paths.RUNS / CM_NAME / "weights" / "best.pt"
            if not (paths.RUNS / CM_NAME / "TRAIN_DONE").exists():
                train_arm(CM_NAME, paths.CONFIGS / "b0_clear_train.yaml", args, state, "baseline")
            cm_weights = paths.RUNS / CM_NAME / "weights" / "best.pt"
            if not cm_weights.exists():
                raise RuntimeError(f"baseline weights missing: {cm_weights}")

            cm_clear = eval_arm(cm_weights, paths.CONFIGS / "b0_clear_eval_clear.yaml",
                                "CM_clear", args.eval_batch)
            base_map = cm_clear.get("mAP50_95")
            if not base_map or base_map <= 0:
                raise RuntimeError(f"baseline mAP50-95 is not usable: {base_map}")
            state["baseline"] = {"status": "done", "run": CM_NAME,
                                 "clear": _pick(cm_clear), "blocker": None}
            save_state(state)
        except OutOfTime:
            raise
        except Exception as exc:
            state["baseline"]["status"] = "blocked"
            blocker(state, "baseline", str(exc))
            write_report(state)
            STOP_FILE.write_text(f"baseline blocked: {exc}\n")
            return 1

        # 1. screen CM on every A_T
        for cond in conds:
            c = state["conditions"].setdefault(cond, {"status": "pending", "blocker": None})
            if c.get("status") in TERMINAL:
                continue
            try:
                result = eval_arm(cm_weights, paths.CONFIGS / f"b0_{cond}_{SEV}_train.yaml",
                                  f"CM_{cond}", args.eval_batch)
                adv_map = result.get("mAP50_95")
                drop = (base_map - adv_map) / base_map if adv_map is not None else None
                c.update({
                    "cm_clear": _pick(cm_clear),
                    "cm_adverse": _pick(result),
                    "drop_rel": round(drop, 4) if drop is not None else None,
                    "drop_abs": round(base_map - adv_map, 4) if adv_map is not None else None,
                    "selected": bool(drop is not None and drop > args.drop_threshold),
                    "status": "selected" if (drop is not None and drop > args.drop_threshold)
                              else "skipped_below_threshold",
                })
                save_state(state)
                print(f"[protocol] {cond}: mAP50-95 {base_map} -> {adv_map}, "
                      f"drop {drop*100 if drop is not None else '?'}%, "
                      f"selected={c['selected']}", flush=True)
            except OutOfTime:
                raise
            except Exception as exc:
                c["status"] = "blocked"
                blocker(state, cond, str(exc))
                write_report(state)

        # 2. retrain selected conditions on C_Tr + A_Tr
        if not args.no_retrain:
            for cond in conds:
                c = state["conditions"].get(cond, {})
                if not c.get("selected") or c.get("status") == "done":
                    continue
                try:
                    ensure_split(cond, "train", state, args.workers)
                    build_yaml(cond, state)
                    name = f"Mix_{cond}_yolo12s"
                    train_arm(name, paths.CONFIGS / f"b1_{cond}_{SEV}_train.yaml",
                              args, state, cond)
                    mix_weights = paths.RUNS / name / "weights" / "best.pt"
                    if not mix_weights.exists():
                        raise RuntimeError(f"mixed weights missing: {mix_weights}")
                    mix_clear = eval_arm(mix_weights,
                                         paths.CONFIGS / f"b1_{cond}_{SEV}_eval_clear.yaml",
                                         f"Mix_{cond}_clear", args.eval_batch)
                    mix_adv = eval_arm(mix_weights,
                                       paths.CONFIGS / f"b1_{cond}_{SEV}_eval_{cond}.yaml",
                                       f"Mix_{cond}_adv", args.eval_batch)
                    gap = (mix_clear.get("mAP50_95") or 0.0) - (mix_adv.get("mAP50_95") or 0.0)
                    c.update({
                        "mix_clear": _pick(mix_clear),
                        "mix_adverse": _pick(mix_adv),
                        "mix_gap": round(gap, 4),
                        "mix_gap_rel": round(gap / mix_clear["mAP50_95"], 4)
                        if mix_clear.get("mAP50_95") else None,
                        "status": "done",
                        "blocker": None,
                    })
                    save_state(state)
                    write_report(state)
                    print(f"[protocol] {cond}: Mix gap {gap:.4f} "
                          f"(clear {mix_clear.get('mAP50_95')} vs adv {mix_adv.get('mAP50_95')})",
                          flush=True)
                except OutOfTime:
                    raise
                except Exception as exc:
                    c["status"] = "blocked"
                    blocker(state, cond, str(exc))
                    write_report(state)

        write_report(state)
        if all_terminal(state):
            DONE_FILE.write_text(datetime.now(timezone.utc).isoformat() + "\n")
            print("[protocol] PIPELINE DONE", flush=True)
            return 0
        return 0

    except OutOfTime as exc:
        WAIT_FILE.write_text(f"{os.environ.get('SLURM_JOB_ID', 'none')}\n")
        print(f"[protocol] OUT OF TIME: {exc}; state saved at {STATE_FILE}", flush=True)
        write_report(state)
        return 42


if __name__ == "__main__":
    sys.exit(main())
