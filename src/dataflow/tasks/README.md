# dataflow.tasks — executable library

**Purpose.** Task executables: the code the runtime launches for each
program task. Ops (math + reference + costs) compose into block executables
resolved by `compute_block_key`. The only layer that imports torch/triton;
never imports the simulator.

## Contracts

- **Executable**: `launch(TaskContext)` — enqueue device work on
  `ctx.stream` only; no synchronization; no globals; idempotent. Buffers
  arrive by object id; the positional order convention per block key is
  documented in `llama3_blocks.py`.
- **Op anatomy** (`ops.py`): every op has a launch form (writes into
  provided tensors where torch supports it) and a *reference* form — pure,
  autograd-able — used by gradcheck and golden models. Numerics discipline:
  bf16 storage, fp32 reductions/normalization statistics.
- **Layouts** (`layouts.py`): packed offsets for weights / saved context /
  optimizer state inside single objects. Layouts are the size source of
  truth (lowering asks `total_bytes`) and the view source for executables.
  Import-light: usable without torch.
- **Interop** (`interop.py`): ctypes-built DLPack capsules wrap raw runtime
  pointers as torch tensors (zero-copy, no ownership); pinned host memory
  presents as CPU tensors; `external_stream` routes torch kernels onto
  runtime streams. The only file touching raw pointers from torch.

## Torch-allocator discipline

Executables may let torch allocate op-internal scratch (flash attention
outputs, matmul workspaces) — that scratch is *measured* per unique task by
`dataflow.training.profiling` (allocator peak-delta with runtime-owned I/O
buffers invisible to torch) and written back into programs. Runtime-owned
buffers never come from torch. `torch.compile` is not used (no workspace
introspection; cudagraph pools break external-buffer ownership).

## Adding an op / block

See `docs/extending.md`: implement launch + reference + costs, register in
the block composition, and run the gradcheck ladder
(`dataflow.training.testing`): op backward → block backward (incl.
recompute-equivalence + accumulation semantics) → model step.
