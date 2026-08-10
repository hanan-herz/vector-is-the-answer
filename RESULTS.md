# Results — one-pass readout vs the autoregressive loop

The headline experiment: a **one-pass readout of the frozen final residual**
(`embedding × head`, zero generated tokens) vs a **fairly-conditioned
autoregressive loop** (full context, balanced few-shot, scored by next-token
log-prob over a closed answer set — Yes/No or A–D — a real classifier read, not
greedy decode) on public closed-form tasks. Primary: **BoolQ** (full val).
Second: **RuleTaker** n2k (Ext 11) for serial-depth strata. Third: **ARC-
Challenge** full (Ext 12) for **parametric science knowledge** (no passage).
Harness: `bench.py`. Interpretation / pitch: `implications.md`.
Synthetic mechanism experiments: `RESULTS_SYNTHETIC.md`.

## Headline table

All four runs are **full-val (9427 train / 3270 val)**, loop
`pad_max=2048`, readout `pad_max=384`.

| model | scale | last.mlp (one-pass) | loop.zero | loop.8 (few-shot) | readout − loop.8 |
|---|---|---|---|---|---|
| **Qwen3-0.6B** | 0.6B | **0.7465** | 0.6287 | 0.6963 | **+0.050** |
| **Qwen3-4B** | 4B | 0.8551 | 0.8547 | **0.8581** | −0.003 |
| **Qwen3-8B** (Ext 9) | 8B | **0.8810** | 0.8618 | 0.8810 | **0.000 (tie)** |
| **DeepSeek-V4-Flash-0731** (Ext 8) | MoE | **0.8964** | 0.8881 | **0.9052** | −0.009 |

**Takeaway.** Across every scale tested, a one-pass residual readout **matches
or beats** a fairly-conditioned autoregressive loop on BoolQ — winning by +5pts
at 0.6B, tying at 4B and 8B, and ~1pt behind the 5×-context few-shot loop
(ahead of zero-shot) at frontier MoE — in a **single forward pass, no
KV-cache**. The loop never cleanly beats the readout at any scale.

![One-pass readout vs fair loop on BoolQ](boolq_results.png)

*Figure: the four full-val runs (0.6B / 4B / 8B / DeepSeek-V4).
Regenerate with `python plot_boolq.py` (reads all numbers from
`results/*.json`).*

*Plot path:* `/Users/hanan/Projects/llm-as-latent-only/boolq_results.png`

Two controls recur throughout: **randproj** (a random-Gaussian-projected linear
head ≈ the full readout ⇒ the signal is *diffuse*, no privileged subspace) and
**ctx vs last** (mean-pooled full-context < final-token ⇒ the verdict is computed
into the final state, not read off the passage). Both hold at every scale.

## FLOPs, honestly

Readout and *scoring*-loop are both a single context forward (FLOP-equal; the
head is ~0.00008% of one forward). The readout's real lever is against a
*decoding* loop: measured greedy on the "Answer:" prompt, output never
self-terminates (0.6B ≥300 tokens; raw DSV4 7/8 hit the 1000-token cap, run
`ap-xDnS2xG9Rt1AyVdHzvZxdH`). So the readout-vs-scoring-loop comparison alone
shows no FLOP win — the FLOP win is replacing generation (RESULTS_SYNTHETIC
Ext 6); the *parametric* win is the readout's: a head instead of a decode loop,
no new tokens, no KV-cache.

## Ext 8 — DeepSeek-V4-Flash-0731 (`bench.py` on Modal B200/B300)

Frontier MoE (43 layers, 167GB FP8). **n_train 9427 / n_val 3270** (full val),
`loop_pad_max=2048` — the loop gets its own fair few-shot budget. run_id
`20260808T141721_f6d7bb`.

| metric (acc) | DeepSeek-V4-Flash |
|---|---|
| **last.mlp** | **0.8964 ± .0015** |
| last.linear / last.linear.max | 0.871 / 0.889 |
| last.mlp.loop_matched (n=3270) | 0.896 |
| **loop.zero** | **0.8881** |
| **loop.k8** (lpm 2048) | **0.9052** |
| last.mlp.shufl (null) | 0.522 |
| ctx.linear / ctx.mlp | 0.732 / 0.678 |
| randproj max / perm / noise | 0.889 / 0.891 / 0.602 |
| budget 64 / 256 / 9427 | 0.848 / 0.870 / 0.896 |

`stratum`: flat across question-/passage-length terciles (0.89–0.90); only
the gold label splits (Yes 0.910 / No 0.872).

**Reading.** `last.mlp` 0.896 vs loop.zero 0.888 / loop.k8 0.905: the
one-pass readout **beats the zero-shot loop** and trails the **8-shot loop by
~1pt** — reading 384 tokens to the loop's 2048, in one forward pass. randproj
`perm` ≥ `max` and `noise` ≫ null confirm diffuse signal; ctx < last confirms
final-token concentration.

## Ext 9 — Qwen3-8B, full scale (`bench.py` on Modal)

Qwen3-8B (36 layers), **n_train 9427 / n_val 3270 / loop_val 3270**
(full val, no subsample), `loop_pad_max=2048`, layer 35. run_id
`20260808T154125_17f659`.

| metric (acc) | Qwen3-8B |
|---|---|
| **last.mlp** | **0.8810 ± .0025** |
| last.linear / last.linear.max | 0.8391 / 0.8743 |
| last.mlp.loop_matched (n=3270) | 0.8810 ± .0025 |
| **loop.zero** | **0.8618** |
| **loop.8** (lpm 2048) | **0.8810** |
| last.mlp.shufl (null) | 0.5297 |
| ctx.linear / ctx.mlp | 0.7107 / 0.7368 |
| randproj max / perm / noise | 0.8746 / 0.8753 / 0.6071 |
| budget 64 / 256 / 9427 | 0.851 / 0.858 / 0.881 |

**Reading.** At 8B the readout **ties the fair few-shot loop exactly** (0.8810 vs
0.8810) and **beats zero-shot by ~2pts** — reading 384 tokens to the loop's
2048, one forward pass. Stronger than Ext 8 (DSV4), where the readout trailed
8-shot by ~1pt: at 8B the "loop wins by a hair" caveat does not survive. randproj
`perm` ≥ `max`, `noise` ≫ null, ctx < last all hold. **Scale arc (Qwen3 BoolQ):**
0.6B beat → 4B trailed → 8B tied — no scale at which the fairly-conditioned loop
cleanly beats the one-pass readout. Artifact:
`results_20260808T154125_17f659.json`. *(Predates the head-artifact save, so
`head_file` is absent; future runs persist trained heads.)*

## Ext 10 — Qwen3-0.6B / 4B, full scale (`bench.py` on Modal)

Completes the Qwen3 scale arc at full scale. Both at **n_train 9427 / n_val 3270 / loop_val 3270** (full
val), `loop_pad_max=2048`. run_ids `20260808T162558_e83621` (0.6B, layer 27),
`20260808T162241_cb93ed` (4B, layer 35). **First runs to persist the trained
heads** (`head_file` present).

| metric (acc) | Qwen3-0.6B | Qwen3-4B |
|---|---|---|
| **last.mlp** | **0.7465 ± .0034** | **0.8551 ± .0009** |
| last.linear / last.linear.max | 0.7413 / 0.7498 | 0.8119 / 0.8566 |
| **loop.zero** | **0.6287** | **0.8547** |
| **loop.8** (lpm 2048) | **0.6963** | **0.8581** |
| last.mlp.shufl (null) | 0.5343 | 0.5424 |
| ctx.linear / ctx.mlp | 0.6361 / 0.6660 | 0.6945 / 0.7183 |
| randproj max / perm / noise | 0.734 / 0.725 / 0.601 | 0.859 / 0.860 / 0.604 |
| budget 64 / 256 / 9427 | 0.657 / 0.676 / 0.747 | 0.853 / 0.845 / 0.855 |

**Reading.** At **0.6B the readout beats the loop decisively** (+5.0pts over
loop.8, +11.8 over zero-shot) — the small model can't surface the answer
autoregressively but the latent holds it. At **4B the readout ties the loop**
(0.8551 vs 0.8581, within noise). Combined with 8B (tie) and DSV4 (~1pt behind),
the full-val arc is uniform: **the loop never cleanly beats the one-pass
readout at any scale**, and the readout's edge is largest where the model is
weakest. randproj `perm` ≈ `max`, `noise` ≫ null, ctx < last all hold at both
scales. Artifacts: `results_20260808T162558_e83621.json`,
`results_20260808T162241_cb93ed.json`.

## Ext 11 — RuleTaker n2k, multi-model, with per-depth loop (`bench.py` on Modal)

Second public closed-form task (`tasksource/ruletaker`): synthetic closed-world
rule reasoning with labeled **deduction depth** (0/1/2/3/5 + NatLang). Same
matched-supervision discipline as BoolQ.

**Protocol (n2k pilot — cite as such, not full RuleTaker test):**

| knob | value |
|---|---|
| train / val | **2000 / 1000** (shuffle seeds 0 / 1; natural depth mix, not stratified sampling) |
| loop_val | **400** (`val[:400]`; cost cap — always compare loop to `last.mlp.loop_matched`) |
| k_shots | **0, 8** |
| pad_max / loop_pad_max | 384 / 2048 |
| GPU | Qwen 0.6B/4B: T4; Qwen 8B: **L40S** (T4 OOM on loop); DSV4: **B300** |

**Artifacts (canonical):**

| model | file | run_id |
|---|---|---|
| Qwen3-0.6B | `results/ruletaker_qwen06b_n2k.json` | `20260809T070151_833647` |
| Qwen3-4B | `results/ruletaker_qwen4b_n2k.json` | `20260809T070146_e15da9` |
| Qwen3-8B | `results/ruletaker_qwen8b_n2k.json` | `20260809T081420_2a40b2` |
| DeepSeek-V4-Flash | `results/ruletaker_dsv4_n2k.json` | `20260809T071108_85201d` |

Plots: `ruletaker_depth_strata.png` (overall + depth),
`head_to_head_three_tasks.png` (BoolQ · RuleTaker · ARC). Regenerate:
`python plot_ruletaker_depth.py`, `python plot_head_to_head.py`.

### Overall (matched loop comparison)

| model | last.mlp (full val) | last.mlp.loop_matched (n=400) | loop.zero | loop.8 | matched − loop.8 |
|---|---|---|---|---|---|
| **Qwen3-0.6B** | 0.655 ± .008 | **0.645** | 0.605 | 0.625 | **+0.020** |
| **Qwen3-4B** | 0.743 ± .002 | **0.734** | 0.658 | 0.723 | **+0.012** |
| **Qwen3-8B** | 0.758 ± .004 | **0.743** | 0.668 | 0.713 | **+0.030** |
| **DeepSeek-V4** | 0.777 ± .004 | **0.778** | 0.593 | **0.798** | −0.019 |

Controls (all models): `last.mlp.shufl` ~0.48–0.51 (≈chance); `ctx.mlp` ~0.52–0.58
≪ last.mlp; randproj `noise` ~0.49–0.51, `perm` ≈ `max`.

**Reading (overall).** On Qwen, the **matched one-pass MLP ≥ fair 8-shot loop**
at every scale (+1–3pts). At DSV4, loop.8 is **slightly** ahead (~2pts) —
parity, not a blowout. Zero-shot loop is weak (especially DSV4 0.59), so
few-shot is required for a fair loop baseline. Same qualitative story as BoolQ
(Ext 8–10): **the loop does not cleanly dominate a residual readout**.

### Per-depth loop vs MLP (same rows — `stratum_depth_loop`)

Eval only on the **loop_val=400** subsample, sliced by depth (n among those 400).
MLP here is the **global full-train head** scored on that slice (fit-once; not a
depth-specific head). Full-val depth MLP alone lives in `stratum_depth` (larger n).

| depth | n | 0.6B mlp / k8 | 4B mlp / k8 | 8B mlp / k8 | DSV4 mlp / k8 |
|---|---|---|---|---|---|
| 0 | 39 | 0.763 / 0.590 | 0.821 / 0.769 | 0.846 / 0.795 | 0.878 / 0.872 |
| 1 | 51 | 0.578 / 0.549 | 0.745 / 0.745 | 0.819 / 0.686 | 0.804 / 0.804 |
| 2 | 35 | 0.643 / 0.829 | 0.807 / 0.800 | 0.750 / 0.857 | 0.814 / 0.829 |
| 3 | 203 | 0.667 / 0.611 | 0.736 / 0.724 | 0.719 / 0.704 | 0.778 / 0.793 |
| 5 | 51 | 0.564 / 0.627 | 0.623 / 0.686 | 0.745 / 0.686 | 0.676 / 0.725 |
| NatLang | 21 | 0.571 / 0.667 | 0.679 / 0.524 | 0.571 / 0.524 | 0.714 / 0.810 |

**Reading (depth).** No sharp “only the loop works past depth D” cliff: accuracy
degrades gradually; d=5 still well above chance. At shallow depths MLP ≈ or >
loop.8; at **d=5** loop.8 can edge MLP by a few points (0.6B/4B/DSV4) or MLP
stays ahead (8B). Thin bins (n=21–51) are noisy — report n, do not overfit
NatLang. Full-val `stratum_depth` shows the same gentle depth slope with larger n
(d3 n=488, d5 n=120).

### Cross-task takeaway (BoolQ + RuleTaker)

| claim | BoolQ full-val | RuleTaker n2k |
|---|---|---|
| one-pass MLP ≈ fair loop.8 | yes (tie / ±1pt; +5 at 0.6B) | yes (Qwen +1–3; DSV4 −2) |
| loop.zero understates the loop | yes | yes (esp. DSV4) |
| scale: no loop monopoly | yes | yes |
| serial-depth cliff for frozen models | n/a | **not observed** |

**Paper sentence.** On BoolQ (full val) and RuleTaker (n2k, matched loop rows), a
small MLP on the frozen last residual matches a fair 8-shot next-token
classifier across Qwen3 scales and DeepSeek-V4; zero-shot understates the loop;
depth degrades gradually without a clean loop-only regime.

**Limits (cite with the numbers).** n2k is a fixed seed subsample, not full
RuleTaker test; loop_val=400 ≠ full val (use `loop_matched`); depth mix is
natural (d3 fat); supervision asymmetry (head sees 2k labels, loop sees 8 demos);
no paired significance tests yet; 8B required L40S (T4 OOM on loop).

![RuleTaker n2k overall + depth](ruletaker_depth_strata.png)

*Plot path:* `/Users/hanan/Projects/llm-as-latent-only/ruletaker_depth_strata.png`

![BoolQ · RuleTaker · ARC head-to-head](head_to_head_three_tasks.png)

*Plot paths:*
- `/Users/hanan/Projects/llm-as-latent-only/head_to_head_three_tasks.png`
- `/Users/hanan/Projects/llm-as-latent-only/head_to_head_boolq_ruletaker.png` (same 3-panel figure; legacy name)

## Ext 12 — ARC-Challenge full, multi-model (`bench.py --task arc`)

**Why this task.** BoolQ is passage-conditioned QA; RuleTaker is formal
composition over **rules in the prompt**. ARC-Challenge (Clark et al. 2018;
`allenai/ai2_arc` config `ARC-Challenge`) is grade-school science **multiple
choice** whose answers depend on **parametric knowledge in the weights** — no
supporting passage. That is the missing “deep knowledge without AR” cell.

**Protocol (full Challenge — cite as such):**
- 4-choice A–D only (~99% of Challenge; 3/5-option rows dropped)
- train = full Challenge train **1117**; val = full Challenge test **1165**
- loop_val = full val (1165); k ∈ {0, 8}; fair loop = next-token log-prob over A/B/C/D
- readout: multi-class MLP (CrossEntropy) + multinomial linear on final residual
- chance **0.25**; majority ~**0.265**
- 0.6B: local MPS; 4B/8B: Modal L40S; DSV4: Modal B300 (weights on
  `model-weights` volume)

**Artifacts (canonical):**

| model | shelf | run_id |
|---|---|---|
| Qwen3-0.6B | `results/arc_qwen06b.json` | `20260809T162709_39b778` |
| Qwen3-4B | `results/arc_qwen4b.json` | `20260809T093854_a7be5b` |
| Qwen3-8B | `results/arc_qwen8b.json` | `20260809T093827_c0a664` |
| DeepSeek-V4-Flash | `results/arc_dsv4.json` | `20260809T094749_fc1203` |

Also under `cloud_bench_cache/{slug}/arc/runs/{run_id}.json`. Prefer `runs/` or
`results/` over `latest.json` (last-writer pointer; empty promote race hit DSV4
once — shelf was filled from the run file).

Plot: `/Users/hanan/Projects/llm-as-latent-only/arc_results.png`  
Regenerate: `python plot_arc.py`.

### Overall (full test, matched loop rows)

| model | last.mlp | last.linear | last.mlp.shufl | loop.zero | loop.8 | mlp − loop.8 |
|---|---|---|---|---|---|---|
| **Qwen3-0.6B** | 0.487 ± .002 | 0.439 | 0.285 | 0.495 | **0.596** | **−0.108** |
| **Qwen3-4B** | **0.844** ± .003 | 0.836 | 0.250 | 0.827 | 0.877 | −0.034 |
| **Qwen3-8B** | **0.910** ± .002 | 0.912 | 0.257 | 0.904 | 0.920 | −0.010 |
| **DeepSeek-V4** | **0.951** ± .001 | 0.951 | 0.218 | **0.961** | 0.958 | −0.007 |

Controls: shufl ≈ chance/majority; ctx.mlp ≪ last.mlp (0.6B 0.285; 4B 0.360;
8B 0.422; DSV4 0.466); randproj near linear (diffuse signal). Budget rises then
flattens (0.6B: n=64→0.42, 256→0.48, 1117→0.49).

**Reading.** Parametric science answers are **in the residual** without
generation (all scales ≫ 0.25). **Zero-shot fair loop ≈ one-pass MLP** at every
scale (within ~1–2pts). **Few-shot loop gap shrinks with scale:** −11pts @0.6B
→ −3pts @4B → −1pt @8B → **−0.7pt @DSV4** (near-tie / noise). At frontier MoE
the loop does **not** pull away — both channels sit ~0.95–0.96. So ARC is not
“MLP always beats loop.8”; it is “knowledge is latently present, and from ~8B
up a residual head matches a fairly-conditioned loop.” Supervision still favors
the head (1117 labels vs 8 demos) — remaining loop edges at small scale are not
from more supervision.

### Cross-task takeaway (BoolQ + RuleTaker + ARC)

| claim | BoolQ full-val | RuleTaker n2k | ARC full |
|---|---|---|---|
| answer in residual (≫ chance, shufl null) | yes | yes | yes |
| one-pass MLP ≈ loop.zero | yes | yes | yes |
| one-pass MLP ≈ loop.8 | yes (tie / +5 @0.6B) | yes (Qwen +1–3) | **scale-dependent** (−11→−1→−0.7 @DSV4) |
| knowledge source | passage (+ some) | rules in prompt | **weights** |

**Paper sentence.** On ARC-Challenge (full test, 4-way), a multi-class MLP on
the frozen last residual recovers grade-school science answers far above chance
without autoregression; it matches the zero-shot fair loop at all scales tested
and closes to within ~1pt of the 8-shot loop by 8B / DSV4 (~0.95) — showing the
BoolQ/RuleTaker readout story extends to **parametric** knowledge, not only
in-context structure.

**Limits.** No ARC-Easy control; no paired significance; historical “Challenge”
difficulty is partly eval-setup (Borchmann 2025) — we score options jointly
(fair MC); pretraining contamination possible (standard leaderboard task);
linear.max can slightly beat MLP mean (0.6B 0.507 vs 0.487) — report both;
0.6B was MPS local, 4B/8B L40S, DSV4 B300.

![ARC-Challenge full: readout vs fair loop](arc_results.png)

*Plot path:* `/Users/hanan/Projects/llm-as-latent-only/arc_results.png`
