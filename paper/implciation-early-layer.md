# Implications of the probe-placement sweep (Ext 14/14b)

*Context: the one-pass readout head's final-layer tap was never swept in the
original runs. Ext 14 (BoolQ 0.6B, full val) and Ext 14b (RuleTaker n2k, 4B)
swept the tap across the residual stream with paired per-row tests. Result:
monotone rise → mid-depth plateau → terminal decline at both scales; L18 best
at 0.6B/BoolQ (+1.0pt vs final, **ns**), L24 best at 4B/RuleTaker (+3.8pt vs
final, **p=2.8e-03 \*\***). Same-rows loop comparison at 4B: L24 readout 0.7725
vs loop.8 0.7225 = +5.0pt (final-layer tap gave only +1.1pt loop-matched).*

## 1. It turns a reviewer objection into a strength

Every headline number in the paper was measured at the final layer. "You only
probed the last layer" was an obvious attack. Now it's answered: the final
layer is a **lower bound** on readout quality. At 4B/RuleTaker, moving the tap
widens the readout's same-rows margin over the loop from +1.1pt to **+5.0pt** —
the thesis ("the residual already contains the answer") was *understated*, not
overstated. The paper can claim: *no layer ever makes the loop win, and the
default tap is the conservative one.*

## 2. Mechanistic reading (frame carefully, but the shape is informative)

Monotone rise → mid-depth plateau → terminal decline, at both scales and both
tasks. Consistent with the standard picture: mid layers do the *computing*,
late layers specialize toward next-token surface form and output calibration —
and that specialization slightly degrades the pooled verdict representation.

*Hedge (important): "late layers are the decode stage / specialize toward
surface form" is a prior from the interpretability literature that our data is*
consistent with, *not something we demonstrate. We only measured that the
readout of the verdict degrades at the final layer; we never ablated deep
layers against generation, never probed next-token structure layer-by-layer,
never measured generation quality at depth. "The final third is load-bearing
for decoding" and "the verdict readout needs less than the full depth" are two
different claims, and only the second is on the table here. The direct test of
the trade-off is a per-layer next-token probe: does next-token predictability
rise monotonically toward the final layer while the verdict readout declines?
(It also settles the "lop off / prune the last N layers" question — pruning
needs evidence the deep layers are cheap for generation, which this setup
cannot see.)*

The task contrast is the interesting part: significant gap on RuleTaker
(serial deduction — the answer must be **computed** over depth) vs ns on BoolQ
(passage-grounded extraction — the answer is **found**). Hypothesis: the
deeper the required computation, the more final-layer specialization costs.
That gives testable predictions (ARC should look like BoolQ; deeper RuleTaker
strata should show larger placement gaps).

## 3. Serving implications — this is the big one for Paper 2

If the verdict head taps at ~2/3 depth, **you can truncate the forward pass at
the tap layer**. Readout-style queries skip not just autoregressive decode but
the entire final third of the transformer — at 4B, 11 of 36 layers ≈ **30%
FLOP reduction per call** on top of zero decode tokens. For the
streamed-frontier story (K3-scale), where the whole point is minimizing
per-call compute, "tap early and stop" is a second, orthogonal saving the
paper didn't have before. And it's deployable *now*: the sweep persists
`head_l24.npz` et al. Caveat: the serving path that truncates at the tap is
untried — that's a Paper 2 experiment, not a claim.

## 4. The two 8B runs discriminate task-type vs scale

Right now we can't tell whether the significant 4B/RuleTaker gap came from
*scale* (bigger models → more specialized late layers) or *task* (serial
depth). The in-flight runs settle it cleanly:

- **8B BoolQ ns + 8B RuleTaker significant** → it's task-type; the
  "computation depth drives placement sensitivity" story holds.
- **Both significant** → it's scale; the placement argument strengthens
  monotonically toward frontier models (where it matters most for the serving
  pitch).

One more consistency nugget: the optima sit at a strikingly similar *fraction*
of depth — L18/28 ≈ 64%, L24/36 ≈ 67%. If 8B lands near L24/36 again, "tap at
~2/3 depth" becomes a clean rule of thumb worth stating.
