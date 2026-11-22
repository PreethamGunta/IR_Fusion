# GATE-IR report

`main.tex` is the consolidated report; `main.pdf` is the compiled version.
Figures are in `figures/` and were generated from the original data and result
files by `scripts`/commands described below.

## Compile

```bash
# Tectonic (self-contained, used for the shipped PDF)
tectonic -X compile main.tex

# or a normal TeX installation
pdflatex main.tex && pdflatex main.tex
# or
latexmk -pdf main.tex
```

The source uses only standard packages (`geometry`, `graphicx`, `booktabs`,
`amsmath`, `caption`, `float`, `enumitem`, `hyperref`, `url`).

## Figures

| Figure | File | Panels |
|---|---|---|
| UNIRI-TID conditions | `figures/fig_uniri_conditions.png` | clear/fog/rain, three non-adjacent frames from different sequences |
| M3FD synthesis | `figures/fig_m3fd_synthesis.png` | clear, classical fog, depth-aware fog, CycleGAN fog; frames 00001, 00060, 01170 |
| FLIR synthesis | `figures/fig_flir_synthesis.png` | clear, CycleGAN fog, CycleDiff-75 fog, CycleDiff-100 fog, U-Net dehazed; frames 000938, 000896, 007909 from three videos |
| FLIR classical filters | `figures/fig_flir_filters.png` | fog, gamma, CLAHE, DCP, gamma+CLAHE; frames 000938 and 007909 |
| FLIR weather gallery | `figures/fig_flir_weather.png` | clear, fog, rain for frame 000938 |
| FLIR additional conditions | `figures/fig_07_conditions.png` | clear + fog, rain, snow, crossover, sensor, small target, defocus, obstruction (frame 000745) |
| FLIR selected pairs | `figures/fig_07_pairs.png` | source/generated pairs for sensor, obstruction, small target and defocus |
| FLIR condition results | `figures/fig_07_results.png` | screen drop and remaining clear-minus-adverse gap |

The 07 figures are produced by
`experiments/07_flir_adverse_conditions/src/make_report_figures.py`; the section
they support is Section~8 of `main.tex`, and the underlying numbers are in
`experiments/07_flir_adverse_conditions/results/FINAL_REPORT.md`.

The report also documents the CycleDiff training time (final 100k→105k translator run: 4 h 24 min) and why the 100 % diffusion translation is unusable.

Classical filter outputs in `figures/filtered/` were produced with
`experiments/02_flir_gate_ir/src/filters.py` on the CycleGAN fog frames.

## Provenance

Numbers were read from the JSON result files listed in the report appendix; the
figures were generated from the dataset and result files described in the
experiment READMEs. The additional FLIR condition numbers come from
`experiments/07_flir_adverse_conditions/results/eval_*.json`, summarized in
`.../results/FINAL_REPORT.md`. See the project root `README.md` and the
per-experiment caveat sections for the scope of each result.
