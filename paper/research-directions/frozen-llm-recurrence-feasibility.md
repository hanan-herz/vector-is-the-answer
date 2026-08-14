# Will external recurrence work over a frozen LLM?

**Status:** feasibility and prior-art analysis, August 2026

## Executive assessment

A task-trained recurrent module over frozen token residuals is likely to work
on controlled tasks. It is substantially less certain that recurrence itself
will produce genuine reasoning that scales with the number of executed steps.

This analysis assumes:

- the pretrained causal LLM is unmodified and its weights never change;
- the LLM processes the prompt once;
- complete token-level residuals are retained as read-only memory;
- only the external recurrent module and output decoder are trained.

If the external module is also untrained, the approach is not expected to
work.

Subjective likelihoods under competent implementation, sufficient task
supervision, explicit positional information, and shortcut-resistant
evaluation:

| Claim | Estimated likelihood |
|---|---:|
| Frozen residuals support pointer chasing after training the external module | 85–95% |
| High in-distribution RuleTaker or ProofWriter answer accuracy | 65–80% |
| Recurrence beats strong parameter- and compute-matched one-shot heads for the intended reason | 30–50% |
| More inference iterations solve substantially deeper proofs than training exposed | 15–30% |
| Broad natural-language reasoning improves reliably without backbone adaptation | <20% |
| A general-purpose latent reasoner emerges from modest task supervision | <10% |

The distinction is central: **high answer accuracy is plausible; proof of
iterative computation is much harder**.

## Proposed system

For a prompt `x`, a frozen LLM produces a token matrix at layer `l`:

```text
H_l(x) in R^(sequence length x model width)
```

A separately trained module maintains writable latent state while treating the
LLM representations as fixed memory:

```text
Z_0 = learned or input-conditioned slots
Z_(t+1) = Refine_theta(Z_t, H_l(x))
output = Decode_theta(Z_T, H_l(x))
```

The external module can compute functions the LLM did not explicitly compute.
It is a small task-specific model operating on a pretrained encoding rather
than raw tokens.

Full token memory is materially more promising than one final-token vector
because it can preserve individual facts and rules, entity occurrences, token
positions, clause boundaries, and distributed evidence that would otherwise be
compressed prematurely.

## What freezing does and does not prevent

Freezing the backbone is not inherently fatal. A learned external module can
combine information preserved in the residual memory in new ways. Instruction
tuning primarily improves the model's native output behavior, while a direct
external decoder does not require instruction following.

The frozen pass nevertheless imposes a strict ceiling. The external reasoner
cannot:

- recover prompt information discarded by the selected residual layer;
- dynamically ask the frozen LLM about a newly derived intermediate result;
- activate parametric knowledge that the original prompt pass did not expose;
- obtain a fresh contextual representation of an intermediate proposition.

If the module derives an entity at step one and needs a fact about that entity
at step two, the fact must already be represented in the fixed prompt memory or
stored in the module's own parameters. It cannot re-query the backbone.

The best fit is therefore **open-book, structurally pointer-like reasoning**.
The fit is weaker for open-ended mathematics, search, planning, or tasks that
need unactivated world knowledge.

## Why recurrence can add computation

Pointer traversal closely matches the architecture:

1. Represent the current key in a latent state.
2. Attend to the memory row containing that key.
3. Retrieve the destination.
4. Store the destination as the next key.
5. Repeat with shared parameters.

Only constant-sized writable state and repeated access to fixed memory are
needed.

Proof search is harder. A proof reasoner may need to retain multiple open goals,
variable substitutions, derived facts, a search frontier, visited rules, and
branch scores. Since the LLM memory is read-only, all newly derived information
must fit in the latent slots. A fixed four-slot state may handle shallow proofs
but become a hard bottleneck as proof width or depth grows. Structured proofs
may require a writable slot bank whose capacity scales with maximum proof size.

## Main technical risks

### Better pooling may masquerade as recurrence

Repeated cross-attention can improve feature extraction without performing a
multi-step algorithm. This is the most likely apparent-positive outcome. A
single attention query is already stronger than last-token or mean pooling;
two or three blocks may simply form a better task-specific summary.

### Decodability does not imply usability

A feature can be linearly decodable from a residual while remaining poorly
aligned with the representation expected by the next transition. A later
computation may be unable to use a feature that a diagnostic head can recover.
The DiscoLoop result discussed below demonstrates this failure directly.

### Causal token states are asymmetric

At token position `i`, a causal model represents only the prefix through `i`.
An early entity mention cannot encode a disambiguating qualifier that appears
later. A globally attending external module can combine early and late states,
but suffix integration and binding then become responsibilities of the
external module.

Clause-ending or delimiter states are likely better memory addresses than the
first token of an entity mention. Explicit adapter-side positions and segment
boundaries should be provided rather than assumed to be robustly recoverable
from residual geometry.

### Weight sharing does not guarantee an algorithm

The recurrent state can encode a clock and implement different effective
behavior at each trained step. Explicit step embeddings make this easier and
weaken extrapolation claims. To test a reusable transition, omit step embeddings,
randomize unroll counts, evaluate well beyond the training horizon, and train
an absorbing completion state.

### Outcome-only supervision permits shortcuts

A final-answer loss allows the module to solve at step one and idle, decode an
answer prior from the last token, use recurrence as ensembling, infer labels
from depth or templates, or memorize a task-specific boundary.

Synthetic tasks should supervise the selected row, next pointer, and halt state.
Proof tasks should supervise premise selection, rule selection, substitutions,
derived facts, and proof completion. A symbolic checker should validate the
entire rollout.

### Extra steps can hurt

Recurrent models can oscillate, converge prematurely, or transform a correct
state into an incorrect one. Training with one fixed recurrence count does not
teach inference-time extrapolation. Variable training unrolls and an absorbing
terminal state are required, and stability should be measured several times
past the training horizon.

## Existing evidence in this repository

### Evidence for representation quality

The local experiments show that frozen Qwen residuals are useful supervised
feature banks:

- strong BoolQ, RuleTaker, ARC, and synthetic readouts;
- useful information before the final layer;
- task-dependent optimal backbone depths;
- independently trained heads sharing one frozen substrate;
- semantic roles and multi-hop relations recoverable from one pass.

This supports the premise that residual memory contains task-relevant
information and that task-specific modules can read it.

### Evidence against overconfidence

The same experiments establish important boundaries:

- Role inversion collapses despite perfect normal-order role decoding, so the
  representations are not automatically flexible compositional bindings.
- The iterated modular-map task is near chance for both readout and
  autoregressive scoring; the required operation is not available in either
  channel.
- Existing RuleTaker labels are already strongly recoverable with a one-shot
  MLP, leaving little room for recurrence and making better pooling a strong
  alternative explanation.
- Directly feeding a last-layer residual back into the LLM drives it off the
  observed activation manifold and rapidly decorrelates it from its seed.

The feedback result does not directly refute an external recurrent module. The
external design keeps the LLM residuals fixed and evolves a separately learned
state space; it never asks the LLM to consume off-manifold hidden states.

## Closest prior art

### RELISH: the strongest reliable architectural precedent

[RELISH: LLM REgression with a Latent Iterative State
Head](https://arxiv.org/abs/2604.01206), COLM 2026, uses:

```text
frozen causal LLM token states
    -> learned projection
    -> latent query repeatedly attends to token states
    -> direct scalar output
```

It evaluates Llama 3.1, Qwen3, and Gemma backbones and reports stronger
regression results than linear heads, parameter-matched MLPs, autoregressive
methods, and regression-aware methods. Macro-averaged over six datasets, four
backbones, and three runs, it reports:

| Method | Pearson | Spearman | normalized RMSE |
|---|---:|---:|---:|
| RAFT, strongest reported prior baseline | 66.1 | 65.3 | 15.7 |
| RELISH | **72.9** | **71.4** | **12.6** |

Important limitations for the present hypothesis:

- the output is scalar regression rather than proof construction;
- its refinement depth appears to use distinct blocks rather than a single
  weight-shared transition;
- most gains arrive by the second refinement and then plateau;
- on Qwen3-8B, one refinement performed best and more blocks slightly hurt.

RELISH establishes that repeated state-dependent reads over frozen token memory
can help. It does not establish recurrence, step extrapolation, or reasoning
compute that tracks problem depth.

### ReLIT: the closest claimed reasoning system, with weak evidence

[Think Deep, Speak Once: ReLIT](https://arxiv.org/abs/2608.08113), posted
August 8, 2026, claims:

```text
frozen TinyLlama token states
    -> shared recursive Transformer block
    -> evolving answer and scratchpad states
    -> one answer token through the frozen LM head
```

It reports `98.6%` on ProofWriter and `97.6%` on RuleTaker. This is nearly the
exact proposed system, but it should not be treated as decisive evidence:

- task-supervised ReLIT is compared with prompted general-purpose models rather
  than matched task-trained heads;
- one-step, parameter-matched, compute-matched, and unshared-stack baselines are
  absent;
- proof validity, depth extrapolation, and a recurrence-count matrix are absent;
- stable outputs and learned halting do not prove reasoning;
- the backbone is a chat checkpoint, frozen only after instruction post-training;
- the released repository currently contains an incomplete notebook rather than
  a complete reproduction path.

ReLIT is strong evidence that the exact idea is already being pursued, but weak
evidence that it works for the intended reason.

### READ: broad recurrent side-network precedent

[READ: Recurrent Adaptation of Large
Transformers](https://arxiv.org/abs/2305.15348), 2023, freezes T5 and trains a
recurrent side network over cached intermediate hidden states. It achieves
competitive GLUE results at lower training memory and energy.

It establishes the broad pattern:

```text
immutable pretrained states
    -> separately trained recurrent reader
    -> task output
```

Its recurrence traverses backbone layers rather than repeatedly querying one
fixed token-memory matrix, and its backbone is T5 rather than a causal decoder.

### Differentiable Cache Augmentation

[Deliberation in Latent Space via Differentiable Cache
Augmentation](https://arxiv.org/abs/2412.17747), 2024, freezes Gemma-2 and
trains a coprocessor over its complete key-value cache. Reported examples with
64 latent embeddings include:

- GSM8K: `21.38 -> 31.43`;
- MMLU: `52.00 -> 56.70`;
- ARC-Challenge: `50.26 -> 54.44`.

This is strong evidence that external computation can improve a frozen causal
LLM. It is not lightweight recurrence: the coprocessor is another large
Transformer, it produces cache augmentations in one pass, final answers are
still generated by the frozen LLM, and training uses a pretraining-scale data
budget.

A relevant ablation found that last-layer activations produced `23.20%` GSM8K,
versus `26.76%` from the complete cache with 32 latents. One residual layer may
therefore be materially poorer than access to multi-layer cache information.

### Frozen-weight recurrent retrofit

[Retrofitting Recurrent Depth into a Pretrained Language
Model](https://arxiv.org/abs/2608.11233) reports installing a recurrent
procedure in Qwen2.5-0.5B with frozen pretrained weights and about six million
trainable LoRA and bridge parameters. With explicit intermediate-state
supervision, the system demonstrates synthetic pointer-chain continuation
beyond the target depth.

This is important evidence that frozen pretrained weights can support an
installed recurrent procedure, but it modifies execution inside the LLM,
reuses pretrained layers, and trains adapters within those layers. Zero-shot
transfer to verbal task surfaces was minimal; task-specific verbal training was
required.

### Negative recurrence results

[CART: Context-Anchored Recurrent
Transformer](https://arxiv.org/abs/2606.01495) repeatedly cross-attends to a
fixed prelude-derived memory, structurally close to the proposal. It found that
extra inference loops beyond the trained count consistently hurt, dense
parameter-matched models beat the recurrent architecture, and recurrence added
only a small language-modeling benefit under its recipe.

[DiscoLoop](https://arxiv.org/abs/2607.00341) found a representation bottleneck
with direct relevance here: an intermediate bridge entity was nearly perfectly
decodable, yet its hidden state had only about `0.3` cosine similarity with the
corresponding token embedding and was poorly usable by the next loop. Injecting
an embedding-aligned channel nearly closed the generalization gap.

The practical lesson is that **decodability is not compositional usability**.
A learned projection, normalization, explicit symbol-like bottleneck, or
mixed continuous/discrete channel may be necessary between recurrent steps.

### Other relevant lines

- Perceiver IO established iterative latent queries over fixed input arrays and
  direct structured outputs, but without a frozen LLM.
- Universal Transformers and looped Transformers establish recurrence and
  depth sharing, but generally train the representation and recurrence jointly.
- SoftCoT freezes the target LLM and trains a projection from assistant-model
  hidden states, but is not recurrent over the target LLM's memory.
- Coconut and related continuous-thought methods train the backbone to consume
  latent thoughts and therefore do not satisfy the frozen-backbone condition.

## Novelty implications

Broad claims such as “the first recurrent module over frozen Transformer
states” are not supportable. READ predates that pattern, Perceiver IO predates
iterative latent processing over fixed memory, Differentiable Cache
Augmentation predates the frozen causal-LLM coprocessor, and RELISH is an exact
frozen token-memory iterative-readout match for regression.

A defensible contribution would need to be empirical and narrower:

- an unmodified frozen base causal LLM rather than a chat or reasoning model;
- token-level residual memory extracted once;
- a genuinely weight-shared external transition;
- direct, mechanically verified proof structures rather than labels;
- recurrence whose useful step count tracks proof depth;
- extrapolation to unseen proof depths;
- strong matched one-shot and unshared-depth controls;
- measured latency and throughput;
- modular task addition without retraining the backbone or existing modules.

The contribution would be this complete behavior and its controls, not the
architecture diagram alone.

## The scientific crux

Once the frozen pass is complete, the system is a small recurrent Transformer
operating on fixed pretrained features. The strongest question is therefore
not merely whether it works:

> Does frozen LLM memory make recurrent computation more effective than the
> same module over raw token embeddings or a frozen random encoder?

Interpretation by outcome:

- If recurrence performs equally over raw embeddings, the recurrent algorithm
  worked but the LLM was incidental.
- If LLM memory mainly improves sample efficiency, the LLM contributed language
  understanding rather than reasoning.
- If more recurrent steps causally track proof depth and outperform the same
  architecture over weaker memory sources, the full hypothesis receives
  support.

## Required comparisons

At fixed `T`, an unrolled recurrent module is a `T`-block network with tied
weights. It is not intrinsically more expressive than an untied stack.
Potential advantages are parameter efficiency, an algorithmic inductive bias,
adaptive compute, and step extrapolation.

Report both fairness axes:

1. **Parameter matched:** recurrence receives more compute by reusing weights.
2. **Compute matched:** an untied stack receives more parameters unless width is
   reduced.

Required baselines:

- linear and MLP heads on the final prompt state;
- one cross-attention pooling block;
- parameter-matched wider one-shot head;
- compute-matched unshared cross-attention stack;
- recurrence with its latent state reset every step;
- one memory read followed by latent-only updates;
- repeated memory reads with a fixed, non-updating query;
- raw token embeddings as memory;
- a frozen random Transformer as memory;
- the frozen base model's native answer where applicable.

A tied recurrent model does not need to beat an untied stack to demonstrate
iteration, but it must beat state-reset and fixed-query controls. To claim an
architectural advantage, it should lie on a better accuracy-parameter-compute
frontier.

## Cheapest decisive experiment

Use a frozen Qwen3-0.6B base model and a randomized textual pointer environment.

### Memory sources

Train the identical external module separately over:

1. raw token embeddings;
2. a frozen random Transformer;
3. Qwen embedding-layer residuals;
4. an early Qwen layer;
5. a task-effective middle Qwen layer;
6. the final Qwen layer.

### Data and training

- Generate fresh random nonce entities for every example.
- Shuffle link rows and include balanced distractors.
- Train on chains of length 1–4.
- Supervise the selected row and next pointer at every step.
- Randomize execution length during training.
- Omit explicit step embeddings.
- Include an absorbing terminal state.
- Balance labels, depth, sequence length, and row position.

### Evaluation matrix

Cross:

- required chain length `L = 1...12`;
- executed recurrence `T = 1...16`.

The intended signature is:

- failure when `T < L`;
- success when `T >= L`;
- continued success beyond the training horizon;
- no degradation after entering the terminal state.

### Causal checks

- Swap `Z_k` between examples sharing the same partial path. Subsequent
  predictions should follow the transplanted pointer.
- Delete or replace the edge required specifically at step `k`. The trajectory
  should remain unchanged before `k` and diverge afterward.
- Reset the state each step or hold the query fixed. Accuracy should collapse.
- Remove the final prompt token. If performance is unchanged, the system is less
  likely to be decoding an answer-like summary already produced by the LLM.

Proceed to ProofWriter or RuleTaker proof traces only if this experiment passes.

## Final conclusion

The frozen token-memory architecture is technically sound and likely trainable.
It avoids the most serious off-manifold problem of feeding residuals back into
the frozen LLM, and strong partial precedents show that external processing can
extract additional value from frozen representations.

The strong claim remains uncertain: a small weight-shared external module may
not acquire a reusable transition that improves according to proof depth and
extrapolates by running longer. The closest direct positive report lacks the
controls needed to settle that question, and carefully controlled recurrent
work also reports overthinking, representation mismatch, and dense-baseline
wins.

The most likely useful outcome is a strong iterative structured head over
frozen representations. Genuine test-time reasoning depth from a reusable
recurrent procedure should currently be treated as roughly a one-in-three to
one-in-two proposition on controlled proof tasks, and substantially less likely
on broad natural reasoning.
