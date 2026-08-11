# The Vector Is the Product:
## One-Pass Readouts of Frozen Language-Model Residuals Match the Autoregressive Loop

*Working draft, 2026-08-11. Source of truth: `paper/outline.md`. Numbers:
`RESULTS.md` (Ext 8–17), `paper/main_tables.tex`. Related work:
`paper/related-work.md`. Target: arXiv preprint first.*

---

## Abstract

On tasks whose answer is representable in a single forward pass, the
autoregressive loop is not the only readout of a language model's computation
— and, we show, not a privileged one. We fit small supervised heads (logistic
regression, or a one-hidden-layer MLP) to the frozen final-token residual of
six open-weight models (Qwen3-0.6B/4B/8B, Mistral-7B, Granite-3.1-8B,
DeepSeek-V4-Flash) and run the readout **head-to-head against a
fairly-conditioned autoregressive loop** — full context, balanced few-shot,
scored by next-token log-probability over the answer set — matched on
information access, scoring protocol, and supervision budget. On BoolQ
(full validation), RuleTaker (n2k), and ARC-Challenge (full test), the one-pass
readout **matches or beats** the fair loop at nearly every scale: it wins
significantly where the model is weakest (RuleTaker +9.3pt at 0.6B, +13.3pt on
Granite-8B; BoolQ +3.8pt at 0.6B against a 4×-budget loop), ties at mid scale,
and loses decisively only in two regimes — small models on parametric
knowledge (ARC) and a frontier MoE on serial deduction. The result survives
label-shuffle nulls, random-projection selectivity, per-length deconfounds,
and paired significance testing on identical rows. Sweeping the readout tap
across depth, we find the verdict representation is fully assembled by
mid-depth: the best mid-layer tap beats the final layer significantly at 9 of
18 task×model cells and never loses. The model, in short, knows more than it
can say — on this task class, **the vector already has the answer.** We bound
the claim explicitly: it is readout-vs-loop on a frozen backbone, not
readout-vs-fine-tuned-specialist, and it does not exhibit a regime where a
frozen loop is required.

---

## 1. Introduction

Autoregressive decoding is the universal serving interface for language
models, but it is only one readout of the computation. Every generated token
is produced by applying a fixed, next-token-trained unembedding to a residual
vector; the answer a model *would* give is thus a particular linear read of a
vector that exists, in full, after a single forward pass. This paper asks a
sharp, falsifiable question about that vector: **on tasks whose answer can be
represented in one forward pass, does a small readout head fit to the frozen
residual recover the answer — and does the autoregressive loop add anything
such a readout lacks?**

The question is easy to answer sloppily. A probe that beats a zero-shot
prompt has shown little: zero-shot understates the loop, and a token-probability
decision rule is a known-weak baseline (Cho et al., 2025). A probe compared
only to the model's own self-report has shown something else entirely — that
beliefs are present, not that a readout can *replace* the loop at producing
the right answer. And any probe result without label-shuffle nulls,
random-projection selectivity, and significance testing is one confound away
from collapse. Our contribution is therefore as much a **discipline of
controls** as a result: we run the comparison the probing literature has
stopped short of — frozen open weights, a readout fit to gold labels, against
a fairly-conditioned loop, matched on supervision budget, information, and
scoring.

**Concurrent work.** The observation that activations carry content the
response does not report is now independently published: Hazenoot et al.
(2026) show a linear probe on frozen activations beats the same model's own
prompted yes/no answer on ESG concept measurement (11/12 comparisons), with
the winning layer mid-network. Their loop baseline is zero-shot prompting —
which we show understates the loop (§3, Ext 13) — and the comparison carries
no paired significance or selectivity controls. We take the convergence as
evidence the premise is ripe; the measurement we run here is the part that
was missing.

**Findings.** On BoolQ (full validation, 3270 rows), RuleTaker (n2k pilot),
and ARC-Challenge (full test), across six models from 0.6B to a frontier MoE:

1. **Parity or better almost everywhere.** The one-pass readout matches or
   beats the fair few-shot loop at nearly every scale — winning significantly
   exactly where the loop is weakest, and tying at mid scale (§3, §4).
2. **The loop wins in two bounded regimes only:** small models on parametric
   knowledge (ARC-Challenge at ≤7B scale), and a frontier MoE on formal
   serial deduction (RuleTaker at DeepSeek-V4). Both gaps close or invert
   with scale and task type (§5).
3. **The answer is assembled by mid-depth.** Sweeping the readout tap across
   the residual stream, the best mid-layer tap beats the final layer
   significantly at 9/18 cells and never loses — the final layer is a *lower
   bound* on what the residual readout can do (§4.4).
4. **The margins can be large.** Where the loop's verbal channel is weakest,
   the readout's lead is largest: +13.3pt on RuleTaker with Granite-3.1-8B
   (readout 0.828 vs loop 0.695, McNemar p=1.3e-06).

**Why it matters.** Later work on residual heads, latent taps, and efficient
inference can treat "the answer is already in the vector" as established on a
defined task class — not as a serving system we ship here, and not as a claim
about all tasks. We state the boundaries explicitly (§5, §6): fine-tuned
specialist encoders beat our readout; serial-depth tasks may require the
loop; the frozen continuous-latent loop drifts. Within those bounds, the
footnote is earned.

**Scope.** A companion position paper explores a serving architecture of
kilobyte-scale readout heads (Paper 2). This paper contains no architecture
claims and no production system.

---

## 2. Setup and Method

### 2.1 Two readouts of the same vector

**One-pass readout.** Let $h_T \in \mathbb{R}^d$ be the final-token residual
of a frozen model after one forward pass over the task prompt (question +
passage, truncated at `pad_max=384`). We fit a head $f(h_T) \to \hat{y}$:
either multinomial logistic regression or a one-hidden-layer MLP (width 128,
4-seed vote), both on standardized features. The head sees only task labels;
the backbone is never updated.

**Fair loop.** The autoregressive baseline is *not* greedy decode — it is the
strongest classifier the loop provides. The model receives the full context
(balanced few-shot exemplars, `loop_pad_max` 2048–8192) and is scored by
next-token log-probability over the closed answer set (Yes/No, or A–D). We
report `loop.0` (zero-shot) and `loop.k` (k-shot); where noted we sweep
k up to 64 with a 4× pad budget (Ext 13).

### 2.2 Matching requirements

A readout-vs-loop comparison is only meaningful if neither side is
handicapped. We match on three axes: **information access** (both see the
same question and passage; the loop additionally sees k exemplars — an
advantage we grant it), **scoring protocol** (both produce a distribution
over the same closed answer set; accuracy on identical rows), and
**supervision budget** (Ext 13 sweeps the loop's exemplar budget to 4× the
readout's training budget at 0.6B scale).

### 2.3 Controls

Every headline number carries four controls:

- **Label-shuffle null** (`mlp.shufl`): the head refit on permuted labels
  must sit at chance, ruling out leakage and spurious separability.
- **Random-projection selectivity** (`randproj`): a head on random Gaussian
  projections of the residual. If it retains accuracy, the signal is
  *diffuse* — no privileged subspace — which is what we find (perm ≈ max ≫
  noise) at every scale.
- **Final-token vs full-context** (`ctx.mlp`): a mean-pooled full-context
  readout must underperform the final-token readout, confirming the verdict
  is *computed into* the final state rather than read off the passage
  surface.
- **Paired significance**: McNemar's exact test and paired bootstrap on
  per-row predictions from identical val rows (`rowpreds_stats.py`), not
  pooled-accuracy eyeballing.

### 2.4 Models and artifacts

Six open-weight models spanning three families and a width ladder: Qwen3-0.6B
(28×1024), Qwen3-4B (36×2560), Qwen3-8B (36×4096), Mistral-7B (32×4096),
Granite-3.1-8B (40×4096), DeepSeek-V4-Flash (43×4096, FP8 MoE). All runs are
batch 8 on model-pinned GPUs (0.6B/4B H200, 8B/Mistral/Granite B200, DSV4
B300). Trained heads are persisted per layer as versioned artifacts
(`heads/head_l{L}.npz`, ~0.5–8.4 MB); per-row predictions persist for paired
tests. Harness: `bench.py`; run manifests embed the batch tag in cache keys
after the Ext 15 cache-identity incident.

---

## 3. BoolQ: one-pass readout vs the fair loop

BoolQ is binary reading comprehension with the gold answer grounded in a
passage — the cleanest case of a one-pass-representable task. We use the full
validation split (9427 train / 3270 val).

### 3.1 Headline

Across all six models, the one-pass readout **wins at 0.6B, ties at 4B/8B and
the two cross-family 8B models, and trails the 5×-context few-shot loop by
~1pt only at the frontier MoE** (Table 1). At no scale does the fairly-
conditioned loop *significantly* beat the readout on BoolQ.

| model | readout (last.mlp) | loop.0 | loop.k | Δ (readout − loop.k) | p (McNemar) |
|---|---|---|---|---|---|
| Qwen3-0.6B | **0.753** | 0.631 | 0.715 | **+0.038*** | 2.4e-05 |
| Qwen3-4B | 0.862 | 0.854 | **0.869** | −0.007 | 0.22 |
| Qwen3-8B | 0.879 | 0.862 | **0.886** | −0.007 | 0.24 |
| Mistral-7B | 0.841 | 0.798 | **0.852** | −0.011 | 0.21 |
| Granite-3.1-8B | 0.854 | 0.815 | **0.864** | −0.010 | 0.64 |
| DeepSeek-V4 | 0.896 | 0.888 | **0.906** | −0.009 | 0.39 |

*(Table 1, abridged from `main_tables.tex`; loop.k = best of k∈{0..64} for
Qwen, k=8 for the rest.)*

### 3.2 Supervision asymmetry is ruled out (Ext 13)

The natural objection: the head sees thousands of labels, the loop sees k
exemplars. We gave the loop its budget back — k swept to 64 balanced
exemplars at 4× pad (8192) — at 0.6B/4B/8B. The loop's k-curve **plateaus by
k=32** (loop.64 − loop.32 = −0.0003, p=1.00 at 0.6B) and **never passes the
readout where the readout was winning**: at 0.6B the readout's +3.8pt over
the best loop stays significant (p=2.4e-05); at 4B/8B all cells are exact
ties. Supervision starvation is not the explanation.

### 3.3 Three readings

1. **Parity or better at every scale; the edge is largest where the model is
   weakest.** The readout's only significant win is at 0.6B — the loop's
   verbal channel is the bottleneck there, and the readout channel saturates
   before it.
2. **The signal is distributed, not localized.** Random-projection heads
   retain accuracy (perm ≈ max ≫ noise) at every scale; no privileged
   subspace carries the verdict.
3. **The verdict concentrates in the final token.** Mean-pooled full-context
   readouts underperform the final-token readout everywhere (ctx.mlp ≪
   last.mlp) — the answer is computed into the final state, not read off the
   passage.

### 3.4 FLOPs, precisely

Readout and *scoring* loop are both a single context forward pass —
FLOP-equal up to context length. The readout's structural lever is against a
*decoding* loop — and here the non-self-termination is **measured**, not
assumed: a greedy decode from the "Answer:" prompt never stops (25/25 hit the
300-token cap at 0.6B; 7/8 hit the 1000-token cap at DeepSeek-V4;
`results/decode_baseline.json`). The parametric win is therefore real and
unconditional: **a head instead of a decode loop — no new tokens, no
KV-cache.** (We quantify the economics and their boundary against early-exit
serving in Table 3; the decode row is measured.)

---

## 4. RuleTaker and ARC: serial depth and parametric knowledge

BoolQ is passage-grounded. We test the two ways a task can be harder for a
one-pass readout: **serial deduction depth** (RuleTaker) and **parametric
knowledge with no passage** (ARC-Challenge).

### 4.1 RuleTaker n2k: matched loop rows, per-depth strata

RuleTaker is closed-world rule reasoning with labeled deduction depth
(0/1/2/3/5 + NatLang). Protocol: 2000 train / 1000 val, loop on 400 matched
rows, k∈{0,8}, batch 8. On Qwen, the matched one-pass MLP **beats the fair
8-shot loop at every scale** (+2 to +9pt; significant at 0.6B, p=3.7e-03).
The cross-family models sharpen the picture: Mistral-7B +8.5pt (p=0.017),
**Granite-3.1-8B +13.3pt (p=1.3e-06) — the largest readout margin in the
campaign.** The one clean loop win is DeepSeek-V4 (+7.4pt, p=1e-03), driven
by the shallow-depth strata.

| model | readout (loop rows) | loop.0 | loop.8 | Δ | p |
|---|---|---|---|---|---|
| Qwen3-0.6B | **0.638** | 0.600 | 0.545 | **+0.093** | 3.7e-03 |
| Qwen3-4B | **0.738** | 0.653 | 0.698 | +0.040 | 0.20 |
| Qwen3-8B | **0.753** | 0.675 | 0.730 | +0.023 | 0.45 |
| Mistral-7B | **0.660** | 0.515 | 0.575 | **+0.085** | 0.017 |
| Granite-3.1-8B | **0.828** | 0.558 | 0.695 | **+0.133** | 1.3e-06 |
| DeepSeek-V4 | 0.763 | 0.605 | **0.838** | −0.074 | 1e-03 |

*(Table 1, RuleTaker block; readout scored on the loop's exact 400 rows.)*

**Per-depth** (same rows): no sharp "only the loop works past depth D" cliff.
Accuracy degrades gradually with depth; d=5 remains well above chance. The
DSV4 loop win concentrates at shallow depths (d0/d1, +10–15pt); at 0.6B the
loop *collapses* with depth (d5 loop.8 0.392 ≪ readout 0.544). Thin bins
(n=21–51) are reported with n and not over-read.

### 4.2 ARC-Challenge: the loop's best case

ARC-Challenge is parametric science knowledge with no passage — the answer
must come from the weights, not the context. Here the loop wins at small
scale: **−10.6pt at 0.6B, −4.6pt at 4B** (both p<1e-05), narrowing to −0.5pt
at 8B and −0.7pt at DSV4 (both ns). The cross-family models replicate the
pattern: Mistral-7B −5.1pt (p=9.3e-05), Granite-8B −8.2pt (p=5.8e-10). The
gap **closes monotonically with scale** — few-shot exemplars teach a task
format the residual head must otherwise learn from labels, and the advantage
shrinks as the parametric answer saturates the residual.

### 4.3 The task law

Across 18 task×model cells, a single law organizes everything: **the loop
wins in exactly two regimes — small models on parametric knowledge (ARC), and
a frontier MoE on formal serial deduction (RuleTaker at DSV4). Everywhere
else the one-pass readout wins or ties.** The ARC crossover tracks
*capability*, not family (Granite/Mistral sit where Qwen did at equal
accuracy); the RuleTaker crossover tracks *serial structure at frontier
scale*.

### 4.4 Readout placement: the final layer is a lower bound

Every headline number above taps the final residual layer. We swept the tap
across depth (10 layers, paired McNemar vs the final-layer readout, trained
heads persisted at every tap). The curve is the same everywhere: monotone
rise, mid-depth plateau, slight terminal decline (Table 2).

- **The final layer never wins.** Mid-depth taps beat the final layer
  significantly at **9/18 cells and never lose significantly** — the headline
  numbers were a lower bound on the readout, not its ceiling.
- **The optimum is task-stable, not scale-driven.** BoolQ/RuleTaker peak at
  50–86% depth at every scale; RuleTaker's optimum sits at ~2/3 depth from
  0.6B through DSV4. ARC drifts to the final layer at ≥8B — once the
  parametric answer saturates the residual, late-layer specialization stops
  costing anything.
- **Placement converts parity into wins.** Where the loop catches the
  final-layer readout (BoolQ 8B: L35 ties loop), the best mid-layer tap
  re-establishes the lead (L30 +1.1pt, p=0.016; on Granite, L20 +1.8pt,
  p=1e-03).
- **Wide-shallow plateaus earlier.** Mistral (32×4096) onsets at 35–58%
  depth vs Qwen3-8B's uniform 69% at the same width — at fixed width, the
  shallower stack assembles the verdict sooner. Granite (40×4096, deep) sits
  between the two.

*Causal hedge.* The monotone-rise → plateau → terminal-decline shape is
*consistent with* late-layer specialization toward next-token surface form,
but we do not demonstrate it: we never ablated deep layers against
generation, and never probed next-token structure per layer. The direct test
(a per-layer next-token probe) is future work; "the verdict readout needs
less than full depth" and "the final third is load-bearing for decoding" are
different claims, and only the first is on the table here.

---

## 5. The boundary: where the loop could win, and why it doesn't cleanly here

A single forward pass is a bounded-depth computation (TC⁰ in the circuit-
complexity framing); the loop's advantage should appear exactly where serial
depth is irreducible. We hunted that regime and found it only weakly
exhibitable on frozen base weights:

- **RuleTaker per-depth** shows a gentle slope, not a regime switch — the
  loop edges the readout by a few points at d=5 (noisy thin bins), and the
  one decisive loop win (DSV4) concentrates at *shallow* depths, where the
  frontier model's in-context deduction is strong.
- **ARC** shows the loop's real advantage is *parametric retrieval under a
  learned task format*, not serial compute — and it closes with scale.
- **Synthetic mechanism** (multi-hop transitivity with shuffled facts;
  `RESULTS_SYNTHETIC.md`): the relation is linearly decodable from the frozen
  residual (0.93–0.97) where the zero-shot decoder sits at chance; a 2-demo
  budget surfaces the loop's inference but never exceeds the one-pass MLP.
  The measured limit is **order-locked binding**, not depth.

The remaining "loop wins" cell — a TC⁰-hard task on which a *frozen* loop
decisively beats one-pass — is largely unoccupied in our sweep; the
counter-position (COCONUT, looped-latent) buys serial depth by *training*,
which is out of scope here.

---

## 6. Discussion

**The thesis, in its conditional form.** On closed-form, one-pass-
representable tasks, a small head on the frozen residual matches a fair
autoregressive classifier — so the answer is already in the vector. This is a
measurement, not a metaphysics: we do not claim the loop never computes, and
we identify the two regimes where it does (§5).

**The model knows more than it can say.** The readout's largest margins are
exactly where the loop is weakest (Granite RuleTaker +13.3pt; 0.6B everywhere)
— the readout channel saturates before the verbal channel. This is the
internal-knowledge/external-behavior disconnect Orgad et al. (2025) document
from the interpretability side; we show it has *operational* content: the
readout is a drop-in replacement for the classifier the loop provides.

**Separability of readout from generative interface.** The unembedding is one
readout of the residual — a fixed, next-token-trained one. That a supervised
head matches it is the default expectation once stated that way; the
empirical content is *where* it fails (ARC at scale, DSV4 deduction) and that
the failure modes are orderly.

**Specialist frontier (acknowledged).** Fine-tuned encoder classifiers
(RoBERTa-large ~0.87, DeBERTa-v3 ~0.88, T5-11B 0.91 on BoolQ) **beat** our
8B one-pass readout (0.879). Our claim is readout-vs-loop on a frozen
backbone — the comparison a serving stack faces when the weights are already
loaded — not readout-vs-fine-tuned-specialist. Where a specialist can be
trained and shipped, it wins on raw accuracy; the readout's economics are
marginal-cost (the forward pass already happened).

**Methodological corrections as worked example.** This campaign corrected
itself twice in ways worth recording: a loop-score cache keyed without batch
identity silently inflated historical loop numbers (Ext 15 — found by
signature: pre-fix runs agreed with each other, post-fix runs agreed with
each other, readouts never affected); and supervision asymmetry, which we
closed by giving the loop a 4× exemplar budget (Ext 13). Both corrections
*sharpened* the headline.

**What we are not claiming.** A frozen continuous latent loop works (it
drifts — Exp 7); the loop never computes; we beat fine-tuned specialists; we
shipped residual-first inference. Paper 2 explores the serving architecture
as a proposal.

---

## 7. Limitations

- **RuleTaker is an n2k pilot** (2000/1000, natural depth mix, d3 fat), not
  the full test; loop rows are 400.
- **Paired tests cover all 18 placement cells and the BoolQ budget arm,** but
  the per-depth loop strata are thin (n=21–51).
- **Supervision asymmetry is bounded, not eliminated**: the head still sees
  more labels than the loop's exemplars at scale parity (Ext 13 closes it at
  0.6B; Arm C — a fine-tuned decoder — is future work).
- **Reproducibility**: per-layer heads are persisted for all sweep runs; the
  two oldest canonical headline runs (Ext 8/9) predate head persistence.
  The FLOP-fraction figure is order-of-magnitude (active-param counts for the
  MoE and a self-terminating decode baseline are unmeasured — Table 3).
- **Scope**: in-knowledge, closed-form tasks only. No claim transfers to
  open-ended generation.

---

## 8. Related Work

*(Full draft in `paper/related-work.md`; condensed version for the paper:)*

**Accessing what a frozen model knows** is established from three directions,
none of which runs our comparison. *Behavioral* work (Turpin et al., 2023)
shows CoT rationales can be unfaithful — text-only access, no representation.
*Latent-belief* work reads the representation directly: CCS (Burns et al.,
2022), pre-CoT belief probes (Cox et al., 2026), belief-formation timing
(Boppana et al., 2026), and the correctness/self-knowledge cluster (Kadavath
et al., 2022; Orgad et al., 2025; "No Answer Needed", 2026) — all predict a
property of the model's own output, none against gold labels, none against
the fair loop. *Decoding-suboptimality* work (Cho et al., 2025; Buckmann &
Hill, 2024; and closest, Hazenoot et al., 2026) shows hidden-state readouts
beat the token-probability decision rule or zero-shot prompting — baselines
their own analyses show are weak. *Probe-for-verification* work (ReProbe, Ni
et al., 2026) uses internal states to improve generation, not replace it.

**The empty cell** — frozen open weights, a readout fit to gold labels, run
head-to-head against a fairly-conditioned loop, matched on supervision
budget, information, and scoring, under format/shuffle/randproj controls —
is the one this paper occupies. Turpin saw behavior, Cox saw belief, Orgad
read its correctness, Cho beat a baseline they showed was weak, Hazenoot beat
zero-shot prompting, ReProbe verified steps to generate better; we compare
the readout and the loop at producing the right answer, because the weights
are open and frozen.

**Delimiting counter-position.** COCONUT (Hao et al., 2024) and looped-latent
transformers (Saunshi et al., 2025) *train* a continuous latent loop to buy
serial depth — they set the boundary we test without training. Early-exit
serving (EE-LLM, CALM, LayerSkip, layer-removal) operationalizes placement
redundancy into a serving optimization; it shares our premise and is prior
art for any "skip the tail" claim, which we do not make.

---

## 9. Conclusion

On closed-form, one-pass-representable tasks, a small readout head fit to the
frozen residual recovers the answer without generating tokens — and the
autoregressive loop adds nothing such a readout lacks. The claim is
conditional, bounded, and control-hardened: it holds across three tasks, six
models, and three families; it fails in two orderly regimes; and its margins
are largest where the loop is weakest. **The vector already has the answer.**
The construction of serving machinery on that substrate is companion work;
here, we measured the substrate.

---

## Figures

- **Fig 1** (`paper/figures/tasklaw_summary.png`): **the task law** — 3×6 heat panel,
  readout − loop.8 per cell with paired-McNemar stars. The one visual: green
  everywhere except two red corners (small-scale ARC, DSV4 RuleTaker).
- **Fig 2** (`paper/figures/boolq_results.png`): BoolQ readout vs loop, four scales.
- **Fig 3** (`paper/figures/head_to_head_three_tasks.png`): BoolQ · RuleTaker · ARC
  head-to-head across models.
- **Fig 4** (`paper/figures/ruletaker_depth_strata.png`): RuleTaker overall + per-depth
  strata, readout vs loop.
- **Fig 5** (`paper/figures/layersweep_placement.png`): placement sweeps, all 18 cells —
  readout accuracy vs tapped layer, loop references, paired annotations.
- **Fig 6** (`paper/figures/boolq_budget_kcurve.png`): Ext 13 budget k-curve — loop plateaus
  by k=32, never passes the readout.
- **Fig 7** (`paper/figures/synthetic_multihop.png`): mechanism — multi-hop transitivity,
  one-pass readout vs decoder loop across Qwen3 scale; the loop never exceeds
  a one-pass readout.

## Tables

- **Table 1** (`main_tables.tex`): readout vs loop, all 18 cells, paired
  McNemar.
- **Table 2** (`main_tables.tex`): placement matrix — best tap vs final,
  plateau onset, paired p.
- **Table 3** (`main_tables.tex`): access-cost orientation (operator-FLOP
  shape, generates?, status: measured/projected/off-axis). Decode row measured.
- **Table 4** (`main_tables.tex`): controls summary — shufl null, randproj
  (max/perm/noise), ctx vs last, all 18 cells.

## Data and artifacts

All numbers: `RESULTS.md` (Ext 8–17), `results/*.json`; heads:
`cloud_bench_cache/<model>/<task>/heads/`; per-row predictions:
`results/*_paired.json` + `rowpreds_stats.py`; synthetic mechanism:
`RESULTS_SYNTHETIC.md`.
