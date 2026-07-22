# Frozen-parameter training

Freezing parameters is a first-class capability: you state WHAT is
frozen once, and the lowering derives every consequence — which wgrad
GEMMs run, which dW/O objects exist and how big they are, which
backward/recompute/optimizer tasks exist at all, how far the gradient
chain reaches, and what each layer saves in its activation context.
The engine, planner, and simulator need no configuration: the program
they receive simply has fewer/smaller objects and tasks.

## 1. Specifying what to freeze

Freezing is part of the OPTIMIZER POLICY (`opt_policy` on any Shaped
config): a field whose resolved rule is `"frozen"` gets no gradient
storage, no optimizer state, and no update. The `freeze()` composer in
`dataflow_training.blocks.optim` is the front door:

```python
from dataclasses import replace
from dataflow_training.blocks.optim import freeze

# whole layers (classic frozen-prefix / continued-pretraining shape)
cfg = replace(cfg, opt_policy=freeze(layers=range(0, 16)))

# one field across every layer
cfg = replace(cfg, opt_policy=freeze(fields=("wq",)))

# targeted (field, layer) pairs
cfg = replace(cfg, opt_policy=freeze(pairs=(("wo", 3), ("w1", 7))))

# embedding / head
cfg = replace(cfg, opt_policy=freeze(embed=True, layers=range(0, 4)))

# compose over any base policy: muon recipe on whatever still trains
cfg = replace(cfg, opt_policy=freeze(base="muon", layers=range(0, 8)))
```

Precedence: field-level freezes apply everywhere, including inside a
layer rule; anything not frozen resolves through the base policy
(default adamw; `"muon"` selects the recipe; any policy object works).
Advanced users can write `OptPolicy`/`MuonRecipePolicy` directly — the
composer is sugar over the same mechanism, and the goldens dispatch
through the identical policy, which is why every freeze configuration
is golden-comparable without extra work.

## 2. How the machinery handles it

Three cooperating layers, each consuming the same policy oracle
(`frozen(field, layer) -> bool`):

1. **Layouts** (`grad_layout(..., opt_policy=...)`,
   `opt_state_layout(...)`): frozen fields DROP OUT of the dW and O
   layouts. A partially frozen layer's dW/O are compact packed layouts
   over the TRAINABLE fields only — smaller objects, not full-size
   objects with holes. Every consumer (the backward's grad views, the
   optimizer's views, the lowering's byte accounting) derives from the
   same call, so offsets agree everywhere by construction.
2. **The FreezePlan analyzer** (`dataflow_training/lowering/freeze_plan.py`):
   `derive_freeze_plan` classifies each layer —
   `train` / `partial` / `passthrough` (fully frozen, something below
   still trains) / `truncated` (fully frozen, NOTHING below trains) —
   using the suffix rule: layer i's backward must produce `dy_{i-1}`
   iff something at depth < i (embedding included) trains. When no
   layer is FULLY frozen, derivation returns `None` and the program is
   byte-identical to an unfrozen build (partial layers need no
   structural change).
   Every family's builder derives its plan (the derivation returns
   `None` on default policies, so unfrozen programs are
   byte-identical), and the DSA families derive CE plans in sparse
   mode too, so structural freezes compose with sparse attention.

3. **The surgery** (`dataflow_training/lowering/freeze_program.py`, dispatched by
   `build_shaped_program(freeze=plan)`): drops truncated layers'
   backward AND recompute tasks, their saved contexts (A) and
   recompute rewrites, and `embed_bwd` when the embedding is frozen.
   The common program builder carries no freeze branches. The existing
   zero-byte scrub in `apply_exact_sizes` then prunes empty dW/O
   objects and any optimizer task whose dW vanished.

## 3. Consequences, per regime

| layer state | wgrads | dgrads | dW / O | backward task | A (saved ctx) |
|---|---|---|---|---|---|
| trainable | all | yes | full | yes | full |
| partial | trainable fields ONLY — frozen fields' wgrad GEMMs are **skipped, not just unwritten** (call sites guard on `acc.wanted(field)`) | yes | packed over trainable fields | yes | full |
| fully frozen, pass-through | none | yes (guards-first: the chain must reach trainable layers below) | none | yes (dgrad-only in effect) | full |
| fully frozen, truncated | none | none | none | **no task** (recompute task dropped too) | **none** |

Plus: optimizer tasks vanish wherever their dW vanished (fully frozen
layers, frozen embedding/head); a frozen embedding drops `embed_bwd`
outright; a frozen head keeps the `head_loss` task (CE still needs the
forward and the dy it emits) but its `dW_head` object is gone.

Memory: dW and O scale with the trainable set (a fully frozen mini
layer contributes ZERO gradient/optimizer bytes); truncated layers
additionally save nothing per round. Compute: frozen wgrad GEMMs never
execute; under guards-first, pass-through layers still pay their
dgrads — those are genuinely needed to reach trainable layers below,
and the remaining skippable work (small norm recomputes feeding only
wgrads) was judged not worth dedicated backward variants.

## 4. Profiling under freeze plans

Task timings come from measured profiles keyed by a cost-equivalence
signature. Sizes alone would under-discriminate frozen plans: two
trainable-field subsets with EQUAL byte totals (freezing `wq` vs `wk`)
would share one timing while skipping different GEMMs. Backward
signatures therefore carry a **freeze fingerprint** — the task's
trainable dW field names, read from the executable's policy-filtered
grad layout — so every distinct skip-combination profiles separately
(pass-through backwards, with no dW at all, are likewise distinct).
The fingerprint is EMPTY when nothing in the task's weight layout is
frozen, so default-policy signatures and their caches are untouched.

## 5. Verification

`tests/dataflow_training/training/e2e/test_freeze_plan.py` gates the analyzer (every freeze
axis), the composer precedence, and four end-to-end engine-vs-golden
model steps: truncated prefix, pass-through, fleet-partial fields, and
grad accumulation. The goldens honor the same policy through their
optimizer dispatch, so frozen params must stay bit-identical on both
sides while trainable params match. Trimmed-context correctness is
self-verifying at runtime: a backward that reads a field outside the
saved set fails loudly (dict KeyError) inside these same gates.

## 6. Special case: dense warm-up for DSA models (dsv32 / glm52)

The papers' recipe trains the lightning indexer FIRST, under full
dense attention, with the main model frozen — the indexer's objective
is a KL against the model's own attention distributions, and CE plays
no role (see the DSA report: the indexer's training signal is only
L_I, permanently detached from CE). This is expressed as a freeze
configuration, not a separate system:

- `sparse_mode=False` (presets: `dsv32_mini_warmup`,
  `glm52_mini_warmup`) injects `opt_policy = frozen-everything except
  the indexer fields (adamw)` and derives a FreezePlan with
  `objective="indexer_kl"`.
- The surgery then produces the specialized program: NO head, NO
  targets, NO dy chain anywhere; `loss_{s}_{r}` keeps its id but IS
  the objective — the group KL, created by the first contributing
  backward in chain order and accumulated by the rest (glm52: the
  IndexShare leaders, against the member-averaged full-prefix target;
  dsv32: every layer). What the bench prints as loss is therefore the
  quantity being optimized.
- dW/O collapse to the indexer fields (a few MiB per model), and the
  saved context trims to the objective's inputs
  (`DSA_WARMUP_CTX_FIELDS` in `dataflow_training/blocks/layouts.py`: the compressed
  latents, norms rstds, and lse the KL backward re-derives everything
  from). At the documentation shape this takes glm52-mini's per-round
  A from ~195 GiB to ~7.4 GiB; the recompute boundary shortens to the
  last stage emitting a saved field (`recompute_stage_count_present`).
- The sparse-stage `train_indexer=False` ablation (RL post-training
  consumes saved selections verbatim) is the same mechanism: the knob
  composes `freeze(base=cfg.opt_policy, fields=<indexer fields>)` onto
  the policy — the five idx fields vanish from dW/O — while remaining
  the compute switch (no KL backward, no dAuxTemp chain). There is no
  other freezing mechanism anywhere.
- Everything above about freezing applies unchanged: frozen embedding
  and head hold their bytes but own no gradients, optimizer tasks
  exist only for the indexer-bearing layers, and the goldens train the
  identical centroid-KL objective for the parity gates.

## 7. Benchmarking frozen configurations

Freezing rides the CONFIG (`opt_policy=freeze(...)`), so every driver
and bench tool that takes a config runs frozen shapes with no dedicated
flags — build the preset, `replace(cfg, opt_policy=freeze(...))`, and
plan/profile/run as usual. Frozen backward signatures re-profile
automatically (the freeze fingerprint, §4); unfrozen signatures hit
the existing cache.

Generated per-preset references (task graph, object tables, measured
kernel sequences): `docs/models/<family>/<preset>_16x4K.md`.
