# Minimum recurrence demo plan

**Status:** exploratory experiment plan, August 2026

## Purpose and positioning

The minimum synthetic recurrence demo is a **unit test for the proposed compute
mechanism**, not a novelty claim or a main research result. A recurrent model
following a randomly generated pointer chain is established territory in
recurrent networks, Universal Transformers, neural algorithmic reasoning,
recurrent graph networks, pointer-based memory systems, Pointer Graph
Networks, and CLRS-style benchmarks.

The demo should establish that the implementation can learn and execute a
shared transition repeatedly before expensive frozen-LLM residual extraction.
It should be described as a recurrence and systems smoke test, not as evidence
that frozen LLM representations support reasoning.

## Minimum task: random pointer chasing

For each example, generate a fresh table of randomly keyed links:

```text
M_i = (key_i, next_key_i)
start key z_0
answer = key reached after L pointer traversals
```

Shuffle the rows and add distractors. The model receives the start key and the
memory table. A single shared refinement block repeatedly retrieves the row
matching the current key and emits its successor:

```text
z_(t+1) = Refine_theta(z_t, M)
```

Use a terminal self-loop so every example has a well-defined answer. Randomly
generate the graph for every example; do not use a fixed table or fixed chain
layout. This prevents memorization and forces each step to use the result of
the preceding step.

### Small initial configuration

- 32-dimensional random keys;
- 16–32 memory rows, including distractors;
- one latent slot and one attention head;
- one weight-shared cross-attention/refinement block;
- train on chain lengths 1–4;
- evaluate separately at lengths 8, 16, and optionally 32;
- execute 1, 2, 4, 8, and longer recurrent steps;
- decode the endpoint by nearest-key similarity;
- use a large batch for GPU measurements.

A one-dimensional cellular automaton is smaller, but pointer chasing is the
better minimum because every recurrent step has an unambiguous computational
role: one data-dependent memory traversal.

## Required comparisons

Use strict parameter- and compute-matched comparisons:

1. The same block executed once.
2. The shared recurrent block with `T = 1, 2, 4, 8`.
3. A larger one-shot feed-forward block with approximately matched parameters.
4. An unshared stack with approximately matched parameters and FLOPs.
5. Optionally, a fixed-depth Transformer or attention baseline.

The main plot should be an endpoint-accuracy heatmap:

- horizontal axis: recurrent steps executed;
- vertical axis: required pointer-chain length;
- cell value: endpoint accuracy.

The intended pattern is near-chance performance when `T < L`, near-perfect
performance when `T >= L`, and parameter reuse across all steps. The strongest
smoke-test result is generalization from training lengths 1–4 to longer chains
at test time, with accuracy scaling according to executed latent steps.

A high score alone is insufficient. The comparison must show that a larger
one-shot head or unshared stack does not reproduce the same step-dependent
scaling at matched resources.

## Staged experiment sequence

### Stage 1: explicit synthetic memory

Run the random pointer-chasing task above. Confirm optimization, recurrence,
step-dependent accuracy, length extrapolation, and the implementation's
throughput behavior.

### Stage 2: noisy or distributed memory

Replace explicit key/value rows with noisy or distributed representations,
using a frozen random encoder or a small encoder. This guards against the
objection that Stage 1 only demonstrates differentiable dictionary lookup.

### Stage 3: frozen causal-model residual memory

Extract complete token-level residuals from an early and a task-effective layer
of a modest frozen causal language model. Treat the token residuals as
read-only memory and apply the same independently trained recurrent latent
module.

This tests the actual project hypothesis:

```text
unmodified frozen causal LLM
    -> token-level residual memory
    -> independently trained recurrent latent module
    -> direct structured output
```

The two compute axes must remain separate:

- backbone depth: which frozen layer supplies the memory;
- recurrent depth: how many shared latent refinements are executed.

### Stage 4: structured task-native output

Move to a task where correctness can be checked mechanically, such as
RuleTaker or ProofWriter with proof nodes and edges. Predict the verdict and
proof structure directly rather than regenerating reasoning as text.

## What would make the frozen-residual result meaningful?

The synthetic pointer task is already known. The potentially interesting result
is the complete frozen-residual system and its measured behavior:

- one unmodified frozen causal LLM pass supplies reusable token memory;
- independently trained task modules use different backbone depths;
- shared latent recurrence improves with additional iterations;
- direct structured outputs avoid vocabulary projection;
- new task heads can be added without retraining the backbone or existing heads;
- adaptive compute can vary both backbone depth and recurrent iterations;
- latency and throughput benefits survive real serving measurements.

The novelty, if any, must come from this specific empirical combination and its
controls, not from the existence of recurrence or pointer chasing.

## GPU implementation and measurement plan

A refinement block is normally several GPU operations rather than one kernel:
projections, attention scores, softmax, value aggregation, normalization, feed-
forward layers, activations, and residual additions. For a tiny recurrent state,
launch and framework overhead can dominate useful arithmetic.

Measure three implementations after warm-up:

1. eager PyTorch;
2. `torch.compile(mode="reduce-overhead")`;
3. a captured CUDA Graph or a fused custom kernel where practical.

Do not synchronize inside the recurrence. Avoid `.item()` and CPU-side halting
decisions per step. For adaptive inference, bucket requests by step count or
run a fixed maximum with an active mask.

As an order-of-magnitude planning assumption, hardware submission is about
`1–5 us`, lean CUDA launch overhead is generally several microseconds, and
PyTorch eager operation overhead can be tens to hundreds of microseconds when
Python and dispatch are included. These figures are hardware- and workload-
dependent and must be measured on the target system.

CUDA Graphs can replay a fixed sequence with one graph launch. NVIDIA reported
about `2.5 us + roughly 1 ns per node` CPU overhead for repeat straight-line
graph launches on Ampere with CUDA 12.6; capture and instantiation are one-time
costs. A fused kernel is stronger for this demo because it also removes
intermediate kernel boundaries and memory traffic.

Report:

- end-to-end latency per example and per batch;
- throughput;
- peak memory;
- executed recurrent steps;
- kernel count and GPU utilization where available;
- eager versus compiled versus graph/fused timing;
- warm-up and graph-capture costs separately.

## Decision gate

Proceed to frozen residual extraction only if Stage 1 demonstrates all of the
following:

- recurrence learns reliably across random graph instances;
- accuracy depends on the number of executed steps;
- longer chains require and benefit from longer execution;
- matched non-recurrent baselines do not show the same scaling;
- the implementation's overhead can be measured and controlled.

If Stage 1 succeeds but Stage 2 fails, the mechanism is likely exploiting the
explicit symbolic structure. If Stages 1–2 succeed but Stage 3 fails, the
limitation is likely the frozen representation or token-memory interface. If
Stage 3 succeeds but Stage 4 fails, the missing component is structured
supervision or output decoding rather than recurrence itself.
