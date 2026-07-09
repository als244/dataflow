# `references/` — isolated ground-truth models

Plain, idiomatic PyTorch (`nn.Module` + autograd) implementations used as the
**correctness ground truth** for the pretraining parity study. The engine
must reproduce these models' loss curves from a byte-identical initialization
on the identical data stream.

## Contract

- **No `dataflow` imports.** Every module here depends only on `torch`. This
  is deliberate: a second, from-scratch implementation catches bugs that a
  shared codebase would hide (including bugs in the engine's own hand-written
  reference ops).
- **Reads like a normal model.** Standard modules, standard autograd — no
  packed layouts, no manual backward, no engine concepts.
- **Numeric conventions match the engine** (so curves track within bf16
  kernel-order noise, not a divergent fp32 model): bf16 weights/activations
  with fp32 RMSNorm/RoPE/softmax/CE reductions, RMS eps `1e-5`, llama
  rotate-half RoPE, GQA, SwiGLU, untied LM head.

## Weight orientation (for the parity bridge)

Projections are `nn.Linear` with weight `(out, in)`; the engine stores packed
`(in, out)` matrices, so the bridge (`dataflow.pretrain.bridge`) loads
`linear.weight = packed.T`. The embedding and LM-head tables are `(vocab, d)`
and load directly. RMSNorm gains are 1-D and load directly.

## Contents

- `llama3.py` — `Llama3(nn.Module)` + `Llama3Config`. The scaling-study family.

## Import

`references/` lives at the repo root (outside the installed `src/` tree). The
root `conftest.py` puts the repo root on `sys.path` for tests; scripts under
`dataflow.pretrain` bootstrap the same path, so `import references` works when
run from the repo root.
