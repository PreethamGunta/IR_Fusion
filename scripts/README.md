# `scripts/` — repository validation

`verify_release.py` is a static layout check: it verifies the expected directory
structure, that symlinks resolve, and that dataset-YAML targets exist. It does
no training, needs no GPU and does not access the network.

```bash
python3 scripts/verify_release.py
```

It reports:

- broken or absolute symlinks,
- dataset-YAML entries whose targets are missing,
- experiment directories missing their expected files.

known, expected warnings are documented in the experiment READMEs (for example,
experiments whose intermediate `data/filtered/…` trees are regenerated at run
time).
