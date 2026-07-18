# dataflow.tasks — executable library

**Purpose.** Task executables: the code the runtime launches for each
program task. Ops (math + reference + costs) compose into block executables
resolved by `compute_block_key`. The only layer that imports torch/triton;
never imports the simulator.

## Layout

| where | what |
|---|---|
| `base_blocks.py` | family-NEUTRAL executables: `_Base`, embed fwd/bwd, `HeadLoss`, the optimizer-step dispatch (`AdamWStep`) |
| `adamw/` | the optimizer-step comm variants, one per file: `dp` (plain allreduce + standalone), `shards` (field-snapped regions + owner broadcast), `rs` (byte-equal reduce_scatter/all_gather), over the shared `update` core |
| `models/` | one module per model family (`<family>_blocks.py`): stages, backwards, resolver. `models/llama3_blocks.py` also hosts the `Block*` templates every family subclasses (their default STAGES are llama's) |
| `modules/` | shared building-block modules: the pluggable `moe/` package, `dsa_forms.py`, `mla_forms.py` |
| `kernels/` | the registry op implementations (eager/triton/aten/fla) |
| `layouts.py`, `ops.py`, `optim.py`, `interop.py` | packed layouts + dims, eager op library, optimizer defs/policies, torch-buffer interop |

## Contracts

- **Executable**: `launch(TaskContext)` — enqueue device work on
  `ctx.stream` only; no synchronization; no globals; idempotent. Buffers
  arrive by object id; the positional order convention per block key is
  documented in `llama3_blocks.py`. Families: llama3
  (`llama3_blocks.py`) and qwen3 (`qwen3_blocks.py` — qk-norm via the
  rmsnorm registry family at head_dim-wide rows; embed/head/loss/optimizer
  executables shared).
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

See `docs/extending.md` (the canonical walkthrough): implement launch +
reference + costs, author the block forward as a `STAGES` tuple (the
recompute variant is DERIVED from it — run stages through the last
context-emitting one — never hand-written), and run the gradcheck ladder
(`dataflow.training.testing`): op backward → block backward (incl.
recompute-equivalence + accumulation semantics) → model step.

## Kernel registry (`kernels/`)

Elementwise/reduction ops dispatch through a registry of swappable
implementations — one file per op family (`swiglu.py`, `rope.py`,
`rmsnorm.py`, `cross_entropy.py`, `adamw.py`), each registering an eager
fallback (always available) and a fused Triton default. GEMMs (cuBLAS),
flash attention and embed scatter/gather (aten) stay direct calls.

Implementations are opaque callables under a fixed ABI — how one is built
(Triton, CuTe DSL, TileLang, a ctypes binding, a cubin launch) is invisible
to the registry:

- ``fn(kctx, *args)`` enqueues all work and returns; no syncs; torch impls
  may use the ambient stream, foreign toolchains use ``kctx.stream_handle``.
- Tensors are contiguous torch views over runtime buffers (``.data_ptr()``
  freely); no references retained past the call; callers own all outputs.
- Workspace is declared: ``none`` | ``arena(bytes_fn)`` | ``internal(hint)``
  — internal allocation is legal but discouraged; hints are validated
  against profiled peaks. ``allocates="vendor"`` flags possible implicit
  syncs.
- ``deterministic`` gates the plan-invariance test mode; ``requires(caps)``
  gates per-device availability (how an AMD/HIP impl will register later).

Selection is pinned once per resolve (override > requires > priority) and
recorded (``KernelSet.describe()``) in profiles and result summaries —
measured task costs are measurements of a *specific* kernel set, so a
mismatch at reuse time is loud, not silent. ``DATAFLOW_KERNELS=eager``
forces the eager set process-wide (numerics bisection / A-B baseline).
Cross-implementation equivalence is tolerance-based against the shared
op references (FMA contraction and transcendental ulps make bitwise
cross-impl equality a non-goal); every impl passes the same gradcheck
ladder.
