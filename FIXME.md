# FIXME — bench.py (Ext 8) review

Review of `bench.py` + the Ext-8 docs, 2026-08-08. Ordered by whether it changes
a number we've published in RESULTS.md.

## Changes published numbers

### 1. The loop scores the wrong token pair — `_yes_no_ids`, bench.py:279
Scores `"Yes"`/`"No"`, but prompts end with `"Answer:"` (no trailing space) and
the few-shot exemplars end with `" Yes"`/`" No"` (bench.py:309). Qwen tokenizer:

    'Yes'  [9454]     ' Yes' [7414]
    'No'   [2753]     ' No'  [2308]

The model's mass is on 7414/2308; we compare 9454/2753. So `loop.zero` /
`loop.k8` in RESULTS.md are read off-distribution — the *fair baseline the whole
experiment exists to establish* is under-measured, in the direction that flatters
our headline. Rerun before these numbers go anywhere.

Fix: score `" Yes"`/`" No"`, or strip the space from the exemplar answers so
prompt and scored token agree.

### 2. ~~BOS-adding tokenizers collapse `_yes_no_ids`~~ — DONE
Handled already. (Was: `tok("Yes")["input_ids"][0]` returns BOS for both words
when `add_bos_token=True`, so `iy == inn`, every row predicts "No", loop lands at
exactly `1 - pct_true` with no error.) Keep the `assert iy != inn` regardless.

~~**Padding side**~~ — DONE. `to_vecs` now derives the last-token index and the
mean-pool mask from `enc["attention_mask"]`, valid for left *or* right padding
(`lens = am.sum(1) - 1` — real tokens are contiguous in both). `loop_scores`
already used the padding-safe `_last_real_index`.

### 3. ~~Right-truncation deletes the question~~ — DONE
`to_vecs` left-truncates (run_bench sets `tok.truncation_side = "left"`), so on
long passages the `Question: …\nAnswer:` suffix survives and the final-token
readout always sees the question; the loop arm now truncates at the same
`PAD_MAX` with the same side, so the two arms see matched inputs. (Trade-off:
on over-long rows the loop's prepended exemplars are the first thing truncated
away — the conditioning budget, not the answer slot, is what gives.)

Cache note: this changed the input preprocessing, so the cache key now also
carries the layer index and PAD_MAX (see #6) — old cached vectors under the
previous key are simply missed and recomputed.

### 4. ~~`randproj.max` inflates the null~~ — DONE
`readout_report` now also reports `last.linear.max` — best-over-C on
`C ∈ {0.01, 0.1, 0.5, 1, 2, 10}` — so the linear readout gets the same
best-over-config selection as randproj's max-over-dims. Compare
`last.linear.max` against `last.randproj.max`, not the single un-swept
`last.linear` (kept for continuity). Reading (d) in RESULTS.md must be re-read
against the swept stat after the next run.

## Correctness, doesn't change published numbers

### 5. Loop cache never hits — bench.py:415 vs :314
Lookup builds `f"loop.{k}"` → `"loop.8"`; the writer produces `f"loop.k{k}"` →
`"loop.k8"`. Cache always misses, so the slowest stage recomputes every run —
and fixing only one side would `KeyError` at :421.

### 6. ~~Cache key is under-specified~~ — DONE
`cache_path` now includes the layer index and `PAD_MAX` in the filename, so
changing either (or the truncation side, via the layer-tagged recompute)
invalidates stale vectors instead of silently reusing them. Call sites in
`run_bench` pass `layer=n_layers_count - 1`.

## Efficiency (matters at DeepSeek scale)

- **bench.py:378** runs a full `readout_report` on `ctx` — 4 MLP seeds + 4
  shuffle seeds + 9 logistic fits — and keeps only `.linear`; then :380
  recomputes the ctx MLP sweep again.
- **Steps [3/5] and [4/5]** retrain the MLP 32× on *identical* training data,
  varying only the eval slice. Train 4 seeds once, slice the predictions.
- ~~**`loop_scores`** is one unbatched forward per row~~ — **DONE**, now batched
  (`batch` threaded from `run_bench` → `loop_report` → `loop_scores`). Rows are
  length-sorted (longest first, so OOM surfaces on batch 1 rather than at 90%)
  and un-sorted on return. Last-token gather uses `_last_real_index` off the
  attention mask, valid for left *or* right padding. Verified equal to the
  row-at-a-time predictions at batch 1/2/5, zero-shot and few-shot, on
  Qwen3-0.6B. Was ~130 rows/min zero-shot on the 04:37 DeepSeek run.
  **Still open:** prefill the (identical across all rows) exemplar block once
  and reuse its KV. Now that batching amortizes weight-streaming, prefill
  compute is the next term — worth maybe 2-3x more on the k=8 pass, but needs
  care with batch-expanding the cache and position ids, so it was not done here.

### 8. Local `modal/` directory shadows the `modal` package — bench.py:69
`import modal` from the repo root resolves to the project's own `modal/`
directory, so `python bench.py` dies instantly with
`AttributeError: module 'modal' has no attribute 'Image'` at bench.py:78. This
is why the local invocations in the docs section below cannot work at all —
independent of the missing argparse. Doesn't affect `modal run` or the remote
container (only 3 files are mounted, per `_KEEP`). Rename the directory, or
make the import absolute.

## Durability

### 7. Results only persist at the very end — bench.py:152
`results.json` is written after *every* stage completes. Steps [1/5]–[4/5] (all
the readout numbers — the expensive part) are held in memory until the loop
baseline finishes, so a timeout or crash in the loop stage throws away work that
was already done and correct. The 04:37 DeepSeek run is a live example: readouts
done by ~05:15, loop stage projected to run past the 06:07 container timeout.

Fix: append to a `results.jsonl` as we go — one record per stage (and per loop
checkpoint of N rows), each `{"stage": ..., "k": ..., ...}`, flushed on write.
Append-only means a killed container leaves a valid file up to the last complete
line, no partial-JSON repair needed, and the loop can resume from the last
checkpoint instead of restarting at row 0. Fold the lines into the final
`results.json` at the end (or in a small reader) so the existing consumers
don't change. The hidden-vector cache already survives a timeout; the derived
numbers don't.

## Docs

- **RESULTS.md:241** cites `--max-train 2400 --max-val 400`, loop on 240, but
  never says the loop numbers came from 240 rows. At n=240 the 0.672 vs 0.695
  zero-vs-8-shot gap is inside binomial noise (~±3 pts), and claim 1's
  "0.731 beats 0.672" is ~1.9σ. State the CI.
- **RESULTS.md:265** calls bench.py "a rerunnable harness" — true only via
  `modal run`; the documented local invocations don't work.
- **bench.py:24 docstring** advertises `python bench.py --size 0.6B`: no `--size`
  (it's `--model`), no argparse wiring at all (`argparse` imported, unused), and
  `__main__` calls a `local_entrypoint`-decorated function directly.
  **bench.py:68** advertises `python bench.py --local` for MPS — also nonexistent.
- **implications.md:251** cites "`bench.py`'s strata" as evidence; per #3 those
  strata are partly measuring truncation.

## Run-config note

The 2026-08-08 04:37 DeepSeek run (`ap-9NZsDpjtYrP6oDltlvKMqP`) launched with
defaults: `max_val=None` → full 3270 val, `loop_val=None` → 3270 loop rows, vs
400/240 in RESULTS.md. ~13× the recorded run's work against `timeout=90*60`
(bench.py:128), i.e. a 06:07 container deadline.

Observed: steps [1/5]–[4/5] finished by ~05:15 (faster than feared — the
forward pass and probes were not the bottleneck). The loop stage is, at
~130 rows/min: k=0 lands ~05:40, k=8 is several times slower and is the pass
expected to hit the timeout. Per #7 that would discard the completed readouts.

Pass explicit `--max-val` / `--loop-val` next time.
