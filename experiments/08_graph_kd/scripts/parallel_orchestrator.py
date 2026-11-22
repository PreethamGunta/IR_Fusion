#!/usr/bin/env python3
"""Three-way parallel, resume-safe orchestrator for FLIR + DroneVehicle.

Runs inside one srun step (32 CPUs, 1 GPU).  Independent arms are launched
concurrently; each job has its own run directory and log.  DONE markers,
teacher-cache meta files and evaluation JSONs make every phase resumable.
"""
from __future__ import annotations

import collections
import os
import subprocess
import sys
import time
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
SRC = EXP / "src"
RUNS = EXP / "runs"
LOGS = EXP / "logs"
CONFIGS = EXP / "configs"
RESULTS = EXP / "results"
PRET = REPO / "models" / "pretrained"
DV = REPO / "data" / "dronevehicle"
DV_CACHE = Path(os.environ.get("DV_CACHE", "/tmp/.cache_8f2c91/dronevehicle_cache"))
DV_CONFIGS = CONFIGS / "kd_dronevehicle"
DV_IR = CONFIGS / "dronevehicle_ir.yaml"
DV_RGB = CONFIGS / "dronevehicle_rgb.yaml"
DV_PAIR = CONFIGS / "dronevehicle_paired.yaml"
DV_IR_RUN = Path("/tmp/.cache_8f2c91/dv_ir_run/ir.yaml")

ENV = dict(os.environ)
ENV.update({
    "PYTHONPATH": str(SRC),
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "NUMEXPR_NUM_THREADS": "4",
    "DV_CACHE": str(DV_CACHE),
})
# Make nested bash helpers (make_dv_configs.sh) use the same Python env.
ENV["PATH"] = str(Path(sys.executable).parent) + os.pathsep + ENV.get("PATH", "")

for d in (RUNS, LOGS, RESULTS, DV_CACHE):
    d.mkdir(parents=True, exist_ok=True)


def weights(name: str) -> Path | None:
    for w in (RUNS / name / "weights" / "best.pt", RUNS / name / "weights" / "last.pt"):
        if w.exists():
            return w
    return None


def train_job(name: str, data: Path, model: str, *, cfg: Path | None = None,
              no_aug: bool = False, fused: bool = False, cache: str | None = None,
              no_val: bool = False) -> dict:
    argv = [sys.executable, "-m", "graphkd.training.train",
            "--name", name, "--model", model, "--data", str(data),
            "--epochs", "100", "--patience", "30", "--batch", "16",
            "--workers", "8", "--exist-ok"]
    if cfg is not None:
        argv += ["--kd-config", str(cfg)]
    if cache is not None:
        argv += ["--cache", cache]
    if no_aug:
        argv += ["--no-aug"]
    if fused:
        argv += ["--fused", "--pretrained", str(PRET / "yolo12s.pt"),
                 "--paired-yaml", str(DV_PAIR), "--rgb-yaml", str(DV_RGB),
                 "--paired-rot90"]
        if no_val:
            argv += ["--no-val"]
    if (RUNS / name / "weights" / "last.pt").exists():
        argv += ["--resume"]
    return {"name": name, "argv": argv, "kind": "train",
            "done": lambda: (RUNS / name / "DONE").exists()}


def cache_job(name: str, cache_dir: Path, weights_path: Path, data: Path,
              *, paired: bool = False) -> dict:
    argv = [sys.executable, "-m", "graphkd.graph.cache_teacher",
            "--weights", str(weights_path), "--data", str(data),
            "--out", str(cache_dir), "--name", name, "--k", "40"]
    if paired:
        argv += ["--paired", "--paired-yaml", str(DV_PAIR), "--rgb-yaml", str(DV_RGB)]
    return {"name": f"cache_{name}", "argv": argv, "kind": "cache",
            "done": lambda: (cache_dir / "meta.json").exists()}


def eval_job(name: str, data: Path, *, paired: bool = False, split: str = "test") -> dict | None:
    w = (RUNS / name / "weights" / "last.pt") if paired and (RUNS / name / "weights" / "last.pt").exists() else weights(name)
    if w is None:
        return None
    argv = [sys.executable, "-m", "graphkd.training.evaluate",
            "--weights", str(w), "--name", name, "--data", str(data),
            "--split", split]
    if paired:
        argv += ["--paired", "--paired-yaml", str(DV_PAIR), "--rgb-yaml", str(DV_RGB)]
    return {"name": f"eval_{name}", "argv": argv, "kind": "eval",
            "done": lambda: (RESULTS / f"{name}_{split}.json").exists()}


def run_pool(jobs: list[dict | None], limit: int, phase: str) -> None:
    jobs = [j for j in jobs if j is not None]
    pending = collections.deque(j for j in jobs if not j["done"]())
    skipped = len(jobs) - len(pending)
    print(f"[{time.strftime('%H:%M:%S')}] {phase}: {len(pending)} to run, {skipped} skipped", flush=True)
    running: dict[int, dict] = {}
    while pending or running:
        while pending and len(running) < limit:
            job = pending.popleft()
            log = open(LOGS / f"par_{job['name']}.log", "a")
            log.write(f"\n=== {time.strftime('%Y-%m-%dT%H:%M:%S')} {phase} {job['name']} ===\n")
            log.flush()
            p = subprocess.Popen(job["argv"], cwd=SRC, env=ENV,
                                 stdout=log, stderr=subprocess.STDOUT)
            log.close()
            running[p.pid] = {"job": job, "proc": p}
            print(f"[{time.strftime('%H:%M:%S')}] start {job['name']}", flush=True)
        time.sleep(2)
        for pid in list(running):
            rec = running[pid]
            rc = rec["proc"].poll()
            if rc is None:
                continue
            job = rec["job"]
            # Training jobs create their DONE marker here; cache/eval jobs
            # verify their own artefact (meta.json / result JSON).
            ok = rc == 0 and (job["kind"] == "train" or job["done"]())
            print(f"[{time.strftime('%H:%M:%S')}] {'done' if ok else 'FAILED'} "
                  f"{job['name']} rc={rc}", flush=True)
            if ok and job["kind"] == "train":
                (RUNS / job["name"] / "DONE").touch(exist_ok=True)
            del running[pid]
    print(f"[{time.strftime('%H:%M:%S')}] {phase} finished", flush=True)


def main() -> None:
    print(f"[{time.strftime('%H:%M:%S')}] parallel orchestrator on {os.uname().nodename} python={sys.executable}", flush=True)
    print(f"CPUs={os.cpu_count()} DV_CACHE={DV_CACHE}", flush=True)

    # ---------------- FLIR core (strong IR teacher) ----------------
    flir = [
        train_job("a1_pred_ir", CONFIGS / "ir.yaml", str(PRET / "yolo12n.pt"),
                  cfg=CONFIGS / "kd" / "a1_pred_ir.json"),
        train_job("a3_node_edge_ir", CONFIGS / "ir.yaml", str(PRET / "yolo12n.pt"),
                  cfg=CONFIGS / "kd" / "a3_node_edge_ir.json"),
        train_job("a5_fgw_ir", CONFIGS / "ir.yaml", str(PRET / "yolo12n.pt"),
                  cfg=CONFIGS / "kd" / "a5_fgw_ir.json"),
    ]
    run_pool(flir, 3, "FLIR core arms")

    flir_eval = [eval_job(n, CONFIGS / "ir.yaml") for n in
                 ("a0_gt", "a2_ir_control", "a1_pred_ir", "a3_node_edge_ir", "a5_fgw_ir")]
    run_pool(flir_eval, 2, "FLIR eval")
    subprocess.run([sys.executable, "-m", "graphkd.training.summarize"],
                   cwd=SRC, env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # ---------------- DroneVehicle teachers ----------------
    teachers = [
        train_job("dv_ir_teacher", DV_IR, str(PRET / "yolo12s.pt"), cache="none"),
        train_job("dv_rgb_teacher", DV_RGB, str(PRET / "yolo12s.pt"), cache="none"),
        train_job("dv_fused_teacher", DV_PAIR, "yolo12s.yaml", fused=True, cache="none"),
    ]
    run_pool(teachers, 3, "DV teachers")

    # ---------------- DroneVehicle caches ----------------
    caches = []
    w = weights("dv_ir_teacher")
    if w:
        caches.append(cache_job("dv_ir", DV_CACHE / "ir", w, DV_IR))
    w = weights("dv_rgb_teacher")
    if w:
        caches.append(cache_job("dv_rgb", DV_CACHE / "rgb", w, DV_RGB))
    w = RUNS / "dv_fused_teacher" / "weights" / "last.pt"
    if w.exists():
        caches.append(cache_job("dv_fused", DV_CACHE / "fused", w, DV_PAIR, paired=True))
    run_pool(caches, 3, "DV caches")

    subprocess.run(["bash", str(EXP / "scripts" / "make_dv_configs.sh")],
                   cwd=EXP, env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # ---------------- DroneVehicle full student matrix ----------------
    # Trimmed early-fusion baseline.  Fused-dependent arms will be rerun
    # against the new graph/feature-fusion teacher after the plan change.
    arms = ["a0_gt", "a2_ir_control", "a1_pred", "a2_node",
            "a3_node_edge", "b_consensus"]
    dv_jobs = []
    for arm in arms:
        if arm == "a0_gt":
            dv_jobs.append(train_job("dv_a0_gt", DV_IR, str(PRET / "yolo12n.pt"), no_aug=True, cache="none"))
        else:
            dv_jobs.append(train_job(f"dv_{arm}", DV_IR_RUN, str(PRET / "yolo12n.pt"),
                                     cfg=DV_CONFIGS / f"{arm}.json", cache="disk"))
    run_pool(dv_jobs, 3, "DV student arms")

    # ---------------- DroneVehicle evaluation ----------------
    eval_jobs = [eval_job(n, DV_IR) for n in
                 ("dv_ir_teacher", "dv_rgb_teacher")] + \
                [eval_job(f"dv_{a}", DV_IR) for a in arms]
    fused = eval_job("dv_fused_teacher", DV_PAIR, paired=True)
    if fused:
        eval_jobs.append(fused)

    run_pool(eval_jobs, 2, "DV eval")
    print(f"[{time.strftime('%H:%M:%S')}] all phases complete", flush=True)


if __name__ == "__main__":
    main()
