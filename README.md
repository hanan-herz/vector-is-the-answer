# The Vector Is the Answer?

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](LICENSE)

**One-pass residual readouts vs fair loop scoring**

On closed-set classification tasks (yes/no and multiple-choice), a single
prompt forward pass is enough: a small supervised head on a **frozen**
open-weight model’s final-token residual matches the strongest closed-set use
of the autoregressive loop as a *classifier* (full context, balanced few-shot,
next-token log-prob over the same answer set). We do **not** claim this for
free-form or multi-step generation. Boundaries of the claim are measured, not
erased.

| | |
|--|--|
| **Paper** | [`paper/paper.pdf`](paper/paper.pdf) · sources in [`paper/`](paper/) |
| **Author** | Hanan Herzog · [ORCID](https://orcid.org/0009-0009-0464-7112) · hanan.herzog@gmail.com |
| **Tasks** | BoolQ (full val), RuleTaker n2k (pilot), ARC-Challenge (full test) |
| **Models** | Qwen3-0.6B/4B/8B, Mistral-7B, Granite-3.1-8B, DeepSeek-V4-Flash |
| **Cite** | [`CITATION.cff`](CITATION.cff) |

Concurrent related work (different baseline/object):
[Hazenoot et al., arXiv:2608.07208](https://arxiv.org/abs/2608.07208)
(hidden-state probes vs weak/zero-shot decoding). The “answer is in the
latent” phenomenon is not novel; this repo measures a **producer readout on
the prompt residual alone** against a **fair closed-set scoring loop** under
matched budget and paired tests.

## Install and verify (no GPU)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/verify_tables.py   # checks results/ vs paper tables
```

Canonical headline numbers live in [`paper/tables.tex`](paper/tables.tex)
(Scheme A: best-of loop arms). Rebuild summaries from JSON under
[`results/`](results/):

```bash
python scripts/build_table1.py
python scripts/verify_tables.py
```

Full experiment re-runs need a GPU or [Modal](https://modal.com); datasets load
via HuggingFace. Cloud image pins also live in Modal builders inside
`bench.py`.

### Paper build

```bash
cd paper && tectonic paper.tex
```

`paper/paper.pdf` is the built preprint; table numbers should match
`scripts/verify_tables.py` / `paper/tables.tex`.

### Full re-runs

```bash
python scripts/extract_canonical.py
# full re-runs (Modal / GPUs): bash scripts/rerun_canonical.sh
# example shape — see bench.py --help for full flags
python bench.py --task boolq --model Qwen/Qwen3-0.6B ...
python paired_test.py ...
```

Reported figures: `python plot_boolq.py`, `plot_head_to_head.py`,
`plot_layersweep.py`, etc. (read from `results/*.json`).

## Layout

| Path | Role |
|------|------|
| `paper/` | Preprint (PDF + LaTeX), figures, bib |
| `bench.py`, `common.py`, `paired_test.py` | Core experiment harness |
| `tasks/` | BoolQ / RuleTaker / ARC loaders |
| `scripts/` | Table build, verify, canonical re-runs |
| `results/` | Run JSON (headline numbers source of truth) |
| `RESULTS.md` | Narrative results index (see banner for canonical numbers) |
| `cloud/` | Optional Modal serving helpers (no secrets) |

## License

Code, data, and manuscript: CC BY 4.0 (attribution required).
See [`LICENSE`](LICENSE).

Cite the PDF once arXiv-assigned; until then cite this repository:

```bibtex
@misc{herzog2026vector,
  title={The Vector Is the Answer? One-Pass Residual Readouts vs Fair Loop Scoring},
  author={Herzog, Hanan},
  year={2026},
  howpublished={GitHub},
  url={https://github.com/hanan-herz/vector-is-the-answer}
}
```

```
Herzog, H. The Vector Is the Answer? One-Pass Residual Readouts vs Fair Loop Scoring.
```
