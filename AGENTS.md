# GATE-IR — agent rules (hard constraints)

This file is binding for every agent in this repository, including GSD subagents
(`gsd-*`). `rules.md` (local, gitignored) holds the full policy; if it exists,
read it before acting. Where this file and a plan, prompt or step file disagree,
this file wins.

## Compute

- Heavy work (training, inference, evaluation, bulk dataset building) MUST run
  inside a SLURM allocation: `salloc` + `srun --jobid=<id>[.<step>]`, or `sbatch`.
  Never run it as a plain shell command on the login node.
- Allocations: `salloc --partition=gpu --gres=gpu:1 --cpus-per-task=16 --mem=64G
  --time=24:00:00`. Allocation names must be neutral lowercase acronyms that
  reveal neither the project, dataset, method nor user.
- NEVER cancel a whole allocation (`scancel <jobid>`) and never stop a running
  training job. Only `scancel <jobid>.<step>` for a step you launched is allowed.
- If a job is pending, wait. Do not resubmit, replace or cancel it.

## Irreversible actions — ask first

- Never delete or modify datasets (`data/`), checkpoints (`*.pt`, `*.ckpt`),
  logs, `runs/`, `results/`, or any training artifact.
- Never run `git clean -fd[x]`, `git reset --hard`, `git restore .`,
  `git checkout -- .`, `git rebase`, `git commit --amend`, `git revert`,
  `git push --force`, or any history rewrite.
- Never touch `.git/`, branches, tags or refs.
- Ask the user before any irreversible action; describe the consequence, then wait.

## Git

- Before every commit: run `git status` (and `git status --ignored` when staging
  directories). Stage only intended source, config and docs; never `*.pt`,
  `*.bin`, `data/`, `runs/`, `logs/`, `results/`, `__pycache__/`, `.planning/`.
- Check for untracked files that should be committed, and for tracked files that
  should be ignored.
- `.planning/` is local agent state and gitignored: never `git add -f` it. GSD's
  `commit_docs` setting must be `false` in this repo.
- Do not commit generated configs/datasets unless the repository already tracks
  that category.

## Paths and data

- All code paths are relative to the repo root or the file's own location; never
  hardcode `/home2/home/...`.
- Datasets live once under `data/`; experiments reference them, never copy or
  hardlink whole datasets into experiment directories.
- Generated YAMLs may contain absolute paths, but generating code must derive
  them from the repo root.

## Subagents

- Only `deepseek-v4.1-flash-free` (preferred) or `deepseek-v4.1-flash` (fallback)
  may be used; never any other model.
- GSD's `gsd-*` agents are allowed only when launched by a `/gsd-*` workflow, on
  the approved models. Mappers and orchestrators write only under `.planning/`;
  executors may edit repo files and commit, but only for tasks in an approved
  PLAN.md.
- Review executor commits and SUMMARY.md before accepting a phase.

## Consequential work

- Before any Tier-3 action — training, inference sweeps, comparative
  evaluations, destructive operations, or claims about results — load the
  `rigorous-engineering` skill and apply its §3.3 pre-flight and §8 self-review.
- Minimum pre-flight: a matched control inside the same pipeline, the *resolved*
  (not requested) config recorded, code SHA in `ref_meta.json`, duplicate check
  against the run ledger, and a resource profile that fits the window.
- Report numbers only from final artifacts; label single-seed or within-noise
  deltas as provisional, never as settled rankings.

## Verification

- YOLO: check training logs for "backgrounds" to confirm labels were found; a
  silent 0-label run trains a broken model.
- Run the relevant tests before claiming work is done; report the command and the
  result.
