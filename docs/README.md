# `docs/` — documentation index

| File | What it is |
|---|---|
| `report/main.pdf` | Compiled 16-page consolidated report (LaTeX source in `report/main.tex`). |
| `report/figures/` | Figures for the report (UNIRI-TID, M3FD, FLIR synthesis and filters). |
| `README_HISTORICAL.md` | The full working README as it stood on 2026-11-09, before packaging. Keeps the old absolute paths and the complete results/limitations narrative. |
| `KNOWN_ISSUES.md` | Every known defect, caveat and incomplete artefact in the release. |
| `REORG_PLAN.md` | The audit and move plan used to build this release. |
| `MOVE_LOG.txt` | Machine-readable log of every top-level move, with pre-move file/symlink counts. |
| `project_notes/` | Design docs, devlog, handoff, master plan, dataset-generation plan, `gap_establishment.md`, UNIRI harness notes. |
| `project_notes/sessions/` | Two recovered Kilo session transcripts about the CycleDiff evaluation. |
| `project_notes/worktree_recover-session-12/` | Newer versions (Oct 2026) of five design docs recovered from the `recover-session-12` worktree. |
| `references/` | Kristo et al. 2020 (UNIRI-TID) paper, extracted text and figures. |
| `baseline_links.md` | Survey of candidate baselines and code links. |
| `baseline_survey_and_experiment_matrix.md` | SOTA matrix and planned experiment grid. |
| `cluster_usage_instructions` | Original cluster notes. |

## Where to start

1. Root `README.md` for the project overview and the headline results.
2. `experiments/02_flir_gate_ir/README.md` for the main GATE-IR experiment.
3. `KNOWN_ISSUES.md` before reusing any number in a paper.

Historical documents in this directory intentionally keep the old absolute
paths (`/home2/home/preetham_g/Gate_IR/...`) so the record is unambiguous.
They are excluded from the "no machine-specific paths" check.
