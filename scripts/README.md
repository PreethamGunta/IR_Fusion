# `scripts/` — release maintenance tools

These are packaging/portability helpers, not experiment code. They are safe to
run on a login node and do not start training.

| Script | What it does |
|---|---|
| `rewrite_symlinks.py` | Converts absolute symlinks to relative ones using the old-root → new-root map. Already applied to this release; re-run only if you import more symlinked data. |
| `relocate_paths.py` | Rewrites machine-specific absolute paths in text files (`.py`, `.sh`, `.sbatch`, `.yaml`, `.json`, `.md`, …). Use after moving the folder. |
| `verify_release.py` | Static validation: expected dirs, symlink resolution, machine-specific paths, dataset YAML targets. No GPU, no network. |

## After moving the release

```bash
python3 scripts/relocate_paths.py --from /old/path/to/release --to /new/path/to/release
python3 scripts/verify_release.py
```

`relocate_paths.py` also upgrades the pre-2026-11-10 layout: it maps
`/home2/home/preetham_g/Gate_IR/...` to the packaged layout, and replaces the
old conda interpreter path with `python`.

## Validation only

```bash
python3 scripts/verify_release.py
```

The script reports:
- broken/absolute symlinks,
- machine-specific paths in source/config,
- dataset YAML entries whose targets are missing.

Known, expected warnings are listed in `docs/KNOWN_ISSUES.md`.
