# CRITICAL: GIT REPOSITORY PROTECTION — ABSOLUTE PROHIBITION

**THIS RULE HAS THE HIGHEST PRIORITY. READ IT BEFORE TOUCHING ANY GIT COMMAND.**

### THE RULE (ZERO EXCEPTIONS):

1. **NEVER delete, modify, reset, revert, rebase, amend, or in any way alter any committed Git commit.**
2. **NEVER delete the `.git/` directory or any file inside it.**
3. **NEVER run `git reset --hard`, `git reset --soft`, `git reset --mixed`, `git revert`, `git rebase`, `git commit --amend`, `git push --force`, `git push -f`, `git filter-branch`, `git filter-repo`, or any command that rewrites Git history.**
4. **NEVER delete Git tags, branches, or refs.**
5. **NEVER modify Git configuration to change commit history.**
6. **NEVER use `git checkout -- .` or `git restore .` to discard working tree changes without explicit user instruction.**
7. **NEVER run `git clean -fd` or `git clean -fdx` without explicit user confirmation.**
8. **NEVER touch `.git/objects`, `.git/refs`, `.git/logs`, `.git/hooks`, or any internal Git file.**
9. **NEVER run `git gc --prune=now` or any pruning command.**
10. **NEVER remove or modify `.gitignore` in a way that exposes tracked files.**

### WHY THIS MATTERS:
- Git commits represent the cumulative history of all work. Losing them is irreversible.
- A single `git reset --hard` can destroy hours or days of committed work.
- Force-pushing can overwrite remote history and affect all collaborators.
- Recovery from Git history loss is often impossible without external backups.

### WHAT TO DO INSTEAD:
- To undo a commit: create a NEW commit that reverses the changes (ask user first).
- To save work: use `git stash` (ask user first).
- To discard changes: ask the user explicitly before running any destructive command.
- If you think a Git operation is needed, ASK THE USER FIRST. Describe the consequences. Wait for explicit confirmation.

### MEMORY ANCHOR:
"Did I just modify Git history? If yes, I violated the most important rule. STOP. Apologize. Never do it again."

**READ THIS SECTION EVERY TIME BEFORE TOUCHING ANY GIT COMMAND.**

---

# GATE-IR — Rules (MUST FOLLOW)

## Artifact Preservation Rules

**NEVER DELETE TRAINING ARTIFACTS WITHOUT EXPLICIT CONFIRMATION.**

This is a hard rule with zero exceptions. Training artifacts include:

1. **Model checkpoints** (`*.pt`, `*.ckpt`) — Each checkpoint represents hours or days of GPU compute time. Never delete them to "free space" or "clean up."

2. **Training logs** (`*.log`, `events.out.tfevents.*`, tensorboard logs) — Required for loss curve analysis, debugging, and reproducibility.

3. **Sample outputs** (`sample-*.png`, generated images) — Essential for qualitative evaluation.

4. **Result directories** (`results/*`, `logs/*`) — Contain the cumulative work product of the entire pipeline.

5. **Intermediate artifacts** — Any file produced during training that could be useful for resuming or analysis.

### When disk space is an issue:

- **Do NOT delete artifacts.** Instead:
  - Ask the user first, with a full list of what would be deleted and why. Describe artifacts in plain language.
  - Reduce checkpoint frequency in the training config (e.g., save every 10,000 steps instead of every 2,000)
  - Compress old log files
  - Clean up only truly non-essential files (`.pyc`, `__pycache__`, temp files)

### Consequences of violation:

- Deleting a partial checkpoint forces the entire training stage to restart from scratch
- This wastes hours of GPU time and delays the project
- Recovery is often impossible because checkpoints are rarely stored in git, trash, or snapshots

### Safe alternatives:

- Adjust `save_every` / `save_and_sample_every` in config files to save less frequently
- Keep only the latest N checkpoints if disk is critical (user must approve)
- Ask user before any bulk deletion: "These files took X GPU-hours to produce. Can I delete them?"

### DO NOT DELETE without explicit user confirmation:

- **`.git/` directory** — Contains git history, branch state, and reflog. Deleting it destroys all version control history and makes recovery impossible. This directory can be 2GB+ but must never be touched.
- **`bench/runs/`** — All YOLO training runs and evaluation results. These are actively used for gap establishment and must be preserved.
- **`data/`** — Any dataset directories (FLIR ADAS v2, DSERT-RoLL, M3FD, UNIRI-TID). Only duplicates that are verified byte-identical can be removed with explicit user approval.

---

## IRREVERSIBLE CHANGES RULE

**NEVER MAKE IRREVERSIBLE CHANGES CASUALLY.**

This is the most important rule in this document. Read it twice.

### Definition of irreversible changes:

Any action that cannot be undone without significant effort or complete rework:

1. **Deleting files** that represent hours or days of work (training checkpoints, datasets, logs, results)
2. **Modifying or deleting data** — datasets, annotations, trained weights, any irreplaceable content
3. **Stopping long-running training jobs** that cannot be easily resumed from the same point
4. **Changing training configurations** in a way that breaks continuity or loses progress
5. **Cancelling SLURM allocations** — these are limited (24h max, 1 GPU per user) and cannot be recovered
6. **Overwriting files** without a backup (especially `final_model.pt`, config files, scripts)
7. **Force-pushing to git** or rewriting history in a way that loses work
8. **Deleting tmux sessions** that contain running or important terminal sessions

### The standard procedure for any irreversible action:

1. **STOP and think.** Ask: "Can this be undone easily? If not, I must ask the user."
2. **Always ask first.** Even if you think it's fine, ask. The user can respond in 2 seconds.
3. **Describe the consequences clearly.** "This will delete 27GB of checkpoints representing 60+ GPU-hours of compute. Once deleted, these cannot be recovered — the training would need to restart from scratch."
4. **Offer alternatives.** "I could reduce the save frequency instead of deleting old checkpoints."
5. **Wait for explicit confirmation.** No assumptions. No "this seems safe." No "I'll just do this quickly."

### Why this matters:

- Compute resources (GPU time) are the scarcest and most expensive resource in ML research
- A single checkpoint deletion can set back the project by days
- SLURM allocations are limited and cannot be recovered once expired
- Mistakes happen. This rule exists so mistakes don't destroy irreplaceable work.

### When in doubt: **ALWAYS ASK FIRST.** Do not make irreversible changes without confirmation.

---

## SLURM Workflow Rules

1. **Keep the `salloc` allocation alive.** Do not cancel or exit it between runs.
2. `salloc` reserves resources; `srun` launches work inside that allocation.
3. You can stop and start `srun` steps without cancelling the `salloc`.
4. Use `srun --jobid=<JOB_ID>` from any tmux session (there is usually an `srun`-named session for running srun commands; don't unnecessarily create new tmux sessions — only make a new one if needed, or use existing idle ones. The `salloc_session` is an exception: it is only used to request `salloc` and never touched again) to target an existing allocation.
5. When the 24h allocation expires, run `salloc` again in `salloc_session` for a new job ID.
6. **Heavy jobs must always be run with a SLURM allocation** (either with `sbatch` or `srun`). Default bash commands run on a login node, which is purely for lightweight things (file creation, downloads, environment setup, installing dependencies). Training, inference and other computationally heavy work must run on the SLURM node.
7. **Shorter time or a different partition does not get an allocation faster** — it is strictly first-come, first-served. So it is always better to use `salloc` with high cpu, ram and time limit, e.g.:
   ```
   salloc --partition=gpu --gres=gpu:1 --cpus-per-task=16 --mem=64G --time=24:00:00 
   ```
   (max limits: 16 CPUs, 64G RAM, 24 hours)
8. **Allocation names must be unidentifiable lowercase acronyms** with no
   relation to the project, dataset, method or username. Use a neutral acronym
   such as `girkd` or another random short code. Never use a name that reveals
   the project or the user, and do not use all-caps names.

---

## Subagent Rule

### Approved subagent
- The ONLY permitted subagent is **`deepseek-v4.1-flash`** (DeepSeek V4.1
  Flash 1017 — OpenCode Go). Never invoke, delegate to, or switch to any other
  subagent or model. If it is unavailable, complete the work yourself; never
  substitute another model.

### Delegation policy
- Default to doing the work yourself. Delegate only genuinely complex,
  time-consuming or parallelisable tasks — not simple edits, routine fixes or
  straightforward investigations.
- Delegate only clearly defined, independent tasks with measurable
  deliverables. Before delegating, check the work cannot be done more
  efficiently directly.

### Workspace isolation
- Create a **new, dedicated directory** per subagent task. Never assign the
  project root or an existing project directory, and never let multiple
  subagents share a working directory.
- The prompt must give the exact assigned path and explicitly list the files
  and folders the subagent may create, edit or delete.
- The assigned directory is a **strict write boundary**. Reading/exploring
  outside it is permitted, but nothing outside it may be modified — directly
  or indirectly, including via running scripts, commands, build tools,
  installing dependencies, lockfiles, configs, caches, temp/build artifacts, or
  git operations. If a required operation would write outside the boundary,
  the subagent must stop and report the limitation instead of proceeding.

### Mandatory prompt content (verbatim, never paraphrased or weakened)
- "Always start by reading `rules.md`, plus the relevant READMEs and docs to
  get context, and follow the rules."
- "Document your code properly."
- "No folders or files outside your assigned folder may be modified, directly
  or indirectly (including via running scripts or anything else that writes
  outside it). Reading and exploring outside the assigned folder is completely
  permitted."

Every delegation prompt must also state: the exact task and objective,
relevant project context, the assigned working directory, permitted files and
operations, expected deliverables and acceptance criteria, known constraints
and risks, and a requirement to report every file it created, modified or
deleted.

### Review before acceptance
- Never blindly trust subagent output; the completion report is not proof.
- After it finishes: inspect every created/modified file; review correctness,
  quality and project-convention consistency; verify `rules.md` compliance and
  that nothing outside the assigned directory changed; run the relevant tests
  yourself; fix problems before integrating.
- Never claim subagent work is verified unless independently verified.

### Failure and recovery
- If the subagent violated its boundaries: reject the work and investigate
  what it modified.
- If it failed: diagnose before retrying. Never expand its permissions
  automatically to make a task easier. If the task cannot be done safely
  inside the assigned directory, stop delegating and do it yourself.

These rules take precedence over convenience, speed and parallelism. When in
doubt: do the work yourself, preserve isolation, review everything.

---

## QUESTION vs INSTRUCTION RULE

**A QUESTION IS NOT AN INSTRUCTION.**

When the user asks a question (anything phrased as a query, "why", "what", "is it", "does it", "should we", "can we", "is that", etc.), the correct response is to ANSWER the question in text. Nothing more.

### What counts as a question (DO NOT act on these):
- "why sample_every not 5000?"
- "is it due to lower total steps and batch size?"
- "does bf16 drastically reduce quality?"
- "what are the batch sizes?"
- "can it be compiled?"
- "shouldnt we next verify the protocol 3b?"
- Any sentence ending in "?" or phrased as an inquiry.

### What counts as an instruction (act on these):
- "set it to 5k"
- "set it to 5k then"
- "change X to Y"
- "do it"
- "proceed"
- "launch it"
- "go ahead"
- "make the change"
- An explicit, imperative command with a clear action verb.

### The rule:
1. **Answer the question first.** Explain, clarify, provide information. Do NOT make any code/config/file changes.
2. **If the question implies a possible action, describe the action and wait.** "If you'd like, I can set sample_every to 5000 to match official. Let me know." — then STOP.
3. **Only act when the user gives an explicit instruction.** Even if the question seems to hint at what they want, do NOT do it until they say "do it" / "set it" / "change it" / "proceed" / "go".
4. **Never combine answering with acting.** Do not say "Here's the answer, and I've also gone ahead and made the change." Answer first, wait for instruction, then act.
5. **When in doubt, treat it as a question.** It is always safer to answer and wait than to act unprompted.

### Why this matters:
- Unprompted changes waste time, break things, and erode trust.
- The user needs to review and approve changes before they happen.
- A question is a request for information, not a request for action.

### Consequences of violation:
- Wasted GPU time from unwanted retraining runs
- Unwanted config changes that need to be reverted
- Loss of user trust and confidence in the agent

---

## Git Commit Rules

1. **Check `git status` before every commit.** Verify:
   - All needed files are staged (source code, config files, scripts)
   - No model checkpoints (`*.pt`, `*.bin`) or large data files are accidentally staged
   - No untracked files that should be committed
   - No tracked files that should be untracked (via `.gitignore`)
2. **Commit all code and config files.** These are essential for reproducibility.
3. **Log files are optional** — commit if small (<1MB), skip if large or contain runtime noise.
4. **Never commit**: `*.pt`, `*.ckpt`, `*.bin`, `data/`, `runs/`, `__pycache__/`, `.pyc` files.

---

## Shared Dataset Rule

- Large datasets live **once** under the repository-root `data/` directory.
- Experiments must reference those datasets directly using paths relative to
  the script, config or repository root. Do not copy or hardlink entire
  datasets into every new `experiments/<id>/data/` folder.
- Experiment-local data is only for **derived artifacts** (labels, split
  lists, warped/re-registered images, caches, predictions).
- Hardlinks do not create extra quota blocks, but they still duplicate path
  trees and confuse provenance. Prefer shared relative paths; use hardlinks
  only when a derived file must physically live in a different tree.
- Generated YAMLs may contain absolute resolved paths, but the generating code
  must derive them from the repo root, never hardcode them.

## Storage Space and /tmp/

1. Use the `quota` command to check the disk space usage and the limit for this user account.
2. If the storage is insufficient, then for temporary storage we can use the /tmp/ folder, but main results, model final check points and needed data should remain in our user folder. 
3. If using the /tmp/ folder, and if making a subfolder inside it, do not use identifiable names (user name or project related names, or so on) just use some unrelated name.

---

## CRITICAL: SLURM ALLOCATION CANCELLATION - ABSOLUTE PROHIBITION

**THIS IS THE MOST IMPORTANT RULE IN THIS ENTIRE DOCUMENT. READ IT THREE TIMES.**

### WHAT HAPPENED (INCIDENT REPORT):
The agent cancelled SLURM job 9449 to "try a shorter allocation" because it thought a shorter time limit would get scheduled faster. This was WRONG. Job 9449 had been waiting in the queue for hours. Cancelling it and creating job 9456 meant starting from the back of the queue again. This WASTED hours of queue waiting time and delayed the project.

### THE RULE (ZERO EXCEPTIONS):
1. **NEVER cancel, revoke, or terminate a SLURM allocation (salloc or sbatch) for ANY reason.**
2. **NEVER cancel a SLURM job to "try different parameters" or "get a shorter queue time".**
3. **NEVER cancel a SLURM job because "it's taking too long to get allocated".**
4. **NEVER cancel a SLURM job because "a different configuration might work better".**
5. **NEVER cancel a SLURM job unless the USER explicitly says "cancel the SLURM job" or "cancel job XXXX".**
6. **If a job is pending, WAIT. Do not touch it. Do not cancel it. Do not replace it. WAIT.**
7. **If you think a job should be cancelled, ASK THE USER FIRST. Describe the consequences. Wait for explicit confirmation.**
8. **There is NO scenario where cancelling a SLURM allocation is acceptable without explicit user instruction. NONE.**

### WHY THIS IS IRREVERSIBLE:
- SLURM allocations are first-come, first-served. Cancelling means losing your place in line.
- The GPU queue has 6+ GPUs and dozens of users. Wait times are hours to days.
- A cancelled allocation cannot be "un-cancelled". You start from scratch.
- The 24-hour time limit means you get ONE chance per day. Wasting it is unacceptable.

### WHAT TO DO INSTEAD:
- If a job is pending: WAIT. Use cron tasks to check periodically. Do NOT cancel.
- If a job is running: Let it finish. Do NOT cancel.
- If you need a different configuration: Wait for the current job to finish, THEN submit a new one.
- If the user wants to cancel: They will say "cancel job XXXX". Then and only then, cancel it.

### CONSEQUENCES OF VIOLATION:
- Wasted hours of queue waiting time (irreversible)
- Delayed project timeline
- Loss of user trust
- The user will be FURIOUS (and rightfully so)

### MEMORY ANCHOR:
"Did I just cancel a SLURM job? If yes, I violated the most important rule. STOP. Apologize. Never do it again."

**READ THIS SECTION EVERY TIME BEFORE TOUCHING ANY SLURM COMMAND.**

---

## Relative Paths Rule

**ALL scripts and Python files must use paths relative to the repository root or to the file's own location. Never hardcode `/home2/home/preetham_g/...` in code.**

- **Python:** derive the root from the file, then compose:
  ```python
  from pathlib import Path
  REPO_ROOT = Path(__file__).resolve().parents[2]   # adjust depth
  DATA = REPO_ROOT / "flir_adas_experiments" / "data" / "flir_adas_v2"
  ```
- **Shell:** resolve from the script location or git:
  ```bash
  ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
  cd "$ROOT"
  ```
- **Configs / YAMLs:** store paths relative to the config file or to `REPO_ROOT`, and resolve them at runtime. Generated YAMLs may contain absolute paths, but the generating code must not hardcode them.
- **New experiments** live in their own top-level folder (e.g. `flir_gate_experiments/`) and must be movable as a unit.

**Why:** the repo is reorganized often (worktrees, folder moves, new machines). Hardcoded absolute paths silently break everything, as happened when `flir_adras_v2` moved out of `recover-session-12` and left every dataset YAML pointing at a directory that no longer exists.
