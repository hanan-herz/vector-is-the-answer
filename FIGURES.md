# What generates what

Paper figures and tables are **not** auto-built. Edit `results/*.json`,
then re-run the script in the left column. Canonical numbers live in
`paper/tables.tex`; figures are PNGs under `paper/figures/`.

## Paper figures (`paper_body.tex`)

| Figure | Script | Reads |
|---|---|---|
| `boolq_results.png` | `python plot_boolq.py` | `scripts/build_table1.py` `CELLS` (BoolQ rows) |
| `boolq_budget_kcurve.png` | `python plot_boolq_budget.py` | `results/boolq_budget_{06b,4b,8b}.json` + `*_paired.json` |
| `ruletaker_depth_strata.png` | `python plot_ruletaker_depth.py` | `results/ruletaker_{qwen06b,qwen4b,qwen8b,dsv4}_n10k.json` |
| `tasklaw_summary.png` | `python plot_tasklaw.py` | `build_table1.CELLS` + rowpreds McNemar |
| `head_to_head_three_tasks.png` | `python plot_head_to_head.py` | `build_table1.CELLS` (also writes legacy alias below) |
| `head_to_head_boolq_ruletaker.png` | same | alias of the three-task figure |
| `layersweep_placement.png` | `python plot_layersweep.py` | `results/{boolq,ruletaker,arc}_layersweep_{06b,4b,8b,mistral7b,granite8b,dsv4}.json` (+ `*_paired.json`) |
| `synthetic_multihop.png` | `python plot_synthetic.py` | hardcoded arrays in the script |

Not in the preprint (supporting / RESULTS.md):

| Figure | Script | Notes |
|---|---|---|
| `arc_results.png` | `python plot_arc.py` | Qwen + DSV4 ARC JSONs only (no Mistral/Granite) |
| `probe_results.png` (+ `_bars` / `_layers`) | `python plot_results.py` | hardcoded synthetic numbers; writes repo root **and** `paper/` (not `paper/figures/`) |
| `confidence_calibration.png` | none | orphan PNG; no generator in-tree |

## Paper tables

| Table | File | Check / rebuild |
|---|---|---|
| Tables 1–4 | `paper/tables.tex` (hand-edited) | `python scripts/verify_tables.py` |
| same, standalone | `paper/main_tables.tex` `\input{tables.tex}` | `lualatex paper/main_tables.tex` |
| preprint | `paper/paper.tex` → `paper_body.tex` + `tables.tex` | `lualatex paper/paper.tex` |

`scripts/verify_tables.py` `CELLS` is the **paper** map (BoolQ budget / RT layersweep).
`scripts/build_table1.py` `CELLS` drives the **figures** (BoolQ budget / RT **10k/4k**).
Those maps are not identical — do not assume a figure regen updates Table 1.

Other helpers: `scripts/extract_canonical.py` (mean vs vote audit),
`scripts/build_reconciled_table.py`.

## Result JSON naming (shelf)

`results/` is git-tracked. Layout: `{task}_{model}_{kind}.json`.

| Kind | Example | What it is |
|---|---|---|
| `budget_*` | `boolq_budget_8b.json` | Ext 13 loop k-curve (pad 8192) |
| `samek_*` | `boolq_samek_8b.json` | CPU readout-n curve (`budget` first-n, `budget.balanced`) |
| `layersweep_*` | `boolq_layersweep_dsv4.json` | placement + last.mlp + loop |
| `*_extract` | `boolq_dsv4_extract.json` | extract-token run; BoolQ `loop_val=1` is dummy |
| `*_n10k` | `ruletaker_dsv4_n10k.json` | RuleTaker 10k/4k (headline) |
| `*_n2k` | `ruletaker_dsv4_n2k.json` | RuleTaker 2k/1k (legacy; layersweep still this size) |
| `*_paired` | `boolq_budget_8b_paired.json` | McNemar / bootstrap from rowpreds |
| `*_rerun` | `boolq_granite8b_rerun.json` | full-val loop when layersweep loop was short |

Same-k and DSV4 extract are **not** wired into plot scripts yet.
DSV4 extract loop scores are not usable (`loop_val=1`).

## One-shot regen (paper figures only)

```bash
python plot_boolq.py
python plot_boolq_budget.py
python plot_ruletaker_depth.py
python plot_tasklaw.py
python plot_head_to_head.py
python plot_layersweep.py
python plot_synthetic.py
```
