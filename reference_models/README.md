# `reference_models/` — the truth tree

Plain, idiomatic PyTorch (`nn.Module` + autograd) implementations that
are the **correctness ground truth** for the whole project: the third
universe of the split (engine `dataflow` / workload `dataflow_training`
/ truth `reference_models` — docs/architecture.md). The engine must
reproduce these models' loss curves from a byte-identical
initialization on the identical data stream; that comparison — its
instrument ladder, calibrated bands, and gotcha catalog — is the
per-family equivalence bar, and its methodology lives in
[docs/correctness_compare.md](../docs/correctness_compare.md).

## Contract

- **No `dataflow` imports — and no cross-imports between these files.** Every
  module here depends only on `torch` and is a COMPLETE, SELF-CONTAINED
  reference: shared primitives (RMSNorm, RoPE, SwiGLU, L2Norm, MoE routing,
  MLA, DSA) are reimplemented in each file, redundantly and on purpose. A
  second, from-scratch implementation catches bugs a shared codebase would
  hide (including bugs in the engine's own hand-written reference ops).
  Nothing in `src/` may import this package either (rule R1,
  `tests/test_import_boundaries.py`) — the workload's bridges import IT,
  never the reverse.
- **Reads like a normal model.** Standard modules, standard autograd — no
  packed layouts, no manual backward, no engine concepts. `(B, T)` int tokens
  (each row an independent causal sequence); `forward -> (B, T, vocab)`;
  `loss(tokens, targets) -> mean CE (fp32)`; optional `grad_checkpoint`.
- **Varlen-native.** Packed mixed-length rounds are first-class: pass
  `seq_lens` with a `(1, sum(seq_lens))` packed row and the model applies
  per-sequence positions and a block-diagonal causal mask — the twin of
  the engine's always-varlen attention + `Segments` wire contract, so
  ragged parity gates compare exactly, never approximately.
- **Numeric conventions match the engine** (so curves track within bf16
  kernel-order noise, not a divergent fp32 model): bf16 weights/activations
  with fp32 reductions for RMSNorm / RoPE / softmax / attention logits / CE /
  router / recurrence state, RMS eps `1e-5`, L2 eps `1e-6`.

## Families

| file | architecture |
|---|---|
| `llama3.py` | dense: GQA + RoPE + SwiGLU (the scaling-study family) |
| `qwen3.py` | dense: per-head qk-norm, decoupled head_dim, GQA, RoPE 1e6 |
| `qwen35.py` | hybrid: Gated-DeltaNet linear layers + gated full-attention |
| `olmoe.py` | full-row qk-norm attention + MoE (softmax-then-topk) |
| `qwen3moe.py` | qwen3 attention + MoE (topk-then-softmax) |
| `qwen35moe.py` | qwen3.5 hybrid + MoE + sigmoid-gated shared expert |
| `dsv3.py` | MLA (latent attention) + MoE (`sigmoid_noaux_tc`) + shared |
| `dsv32.py` | MLA + DSA (sparse attention) + MoE + shared |
| `glm52.py` | MLA + DSA (IndexShare) + MoE + shared |

Every registered model family carries a twin (`ModelFamily.twin_module`
points here) — a family without one cannot take the full correctness
treatment, and adding a family includes adding its twin
(docs/extending.md §3).

## MoE load-balancing loss (optional)

The six MoE families (olmoe, qwen3moe, qwen35moe, dsv3, dsv32, glm52) expose
an OPTIONAL load-balancing auxiliary loss:

    model.loss(tokens, targets, aux_coef=0.0)   # aux_coef=0 -> pure mean CE
    model.loss(tokens, targets, aux_coef=0.01)  # + alpha * sum_layers L_layer

Softmax routers (olmoe, qwen3moe, qwen35moe): `L_layer = E * sum_e f_e * p̄_e`
(the standard Switch/GShard term, matching the engine's
`moe_aux_loss_reference` / flextrain): `f_e = count_e/(T*K)` from the discrete
top-K assignments, `p̄_e` = mean full-softmax router probability per expert.
`sigmoid_noaux_tc` routers (dsv3, dsv32, glm52): DeepSeek-V3's complementary
SEQUENCE-WISE loss (matching the engine's `moe_seq_aux_loss_reference`) — per
`(B, T)` row, `sum_e f_e^s * P_e^s` with `f_e^s = count_e^s * E/(K*T)` and
`P_e^s` the row-mean NORMALIZED-sigmoid prob, summed over rows; their
optimizer-time bias sign rule is exposed as `MoE.apply_bias_update(speed)`
for the training harness.
Uniform routing gives `L_layer = 1`; imbalance pushes it above 1. The shared
expert (where present) has no router and is excluded. `model.load_balance_loss()`
returns the summed per-layer term from the most recent forward.

## Weight orientation (for a parity bridge)

Projections are `nn.Linear` with weight `(out, in)`; the engine stores packed
`(in, out)` matrices, so a bridge loads `linear.weight = packed.T`. Embedding
and LM-head tables are `(vocab, d)` and load directly; 1-D norm gains directly;
a depthwise conv is `(D, W) -> (D, 1, W)`. Raw-parameter tensors (expert
stacks everywhere; per-file choices like qwen3moe's router or dsv32's shared
expert) are already in the engine orientation and load directly.
`dataflow_training.model_families.bridges` re-exports one bridge module per
family (`model_families/<family>/bridge.py`, all nine); every family is
gate-checked against its twin from a byte-identical init (state_dict
byte-identity + forward + training-curve agreement) and against the ENGINE
SERVICE over real fineweb steps
(`tests/dataflow_training/models/test_engine_vs_reference.py`,
`tests/dataflow_training/pretrain/test_engine_parity_families.py`).

## Import

`reference_models/` lives at the repo root (outside the installed `src/`
tree) — isolation is physical, not just conventional. The root
`conftest.py` puts the repo root on `sys.path` for tests; the workload's
`model_families` package init arms the same path before its bridge
modules import, so `import reference_models` works when run from the
repo root. Suites whose SUBJECT is a twin itself live in
`tests/reference_models/` (see its README).
