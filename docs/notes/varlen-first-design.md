# Design: varlen-first activations across all models

Requested by Shein 2026-07-03: activations represented as
`[total_tokens, hidden]` (no batch dim) in ALL models, with the batch's
sequence lengths tracked and passed to the sequence-dependent ops (rope,
attention, and the qwen35 conv/linear-attn resets). Status: DESIGN.

## What is already true (audit, 2026-07-03)

- Every block in all three families ALREADY computes on
  `[total_tokens, d]` — `t = x.shape[0]` everywhere; the packed
  layouts, contexts, and lowered object sizes are token-major with no
  batch dim. The batch dim exists only INSIDE two ops, derived from a
  uniform `seq_len` scalar:
  - `ops.flash_fwd/bwd`: `b = t // seq_len`, dense reshape to
    `(b, s, h, hd)` + `repeat_interleave` for GQA, aten
    `_scaled_dot_product_flash_attention`.
  - rope triton kernel: `pos = row % seq_len`.
- qwen35's DeltaNet path is already fully varlen: `(1, t)` +
  `cu_seqlens`/`chunk_indices` into fla chunk + conv kernels, reset
  semantics pinned by the batch=2 packed-sequence E2E test. The M5.2
  microbench (tools/bench_qwen35_kernels.py) showed varlen == dense for
  those kernels at our shapes.

So this feature = replace the uniform-`seq_len` plumbing with explicit
sequence-length metadata, and convert the two remaining ops. The
REPRESENTATION does not change; kernel-internal fast paths may still
reshape when uniform (that is an op detail, not IR).

## Sequence metadata

- Lowering-level: `dims.seq_lens: tuple[int, ...]` per round (sum =
  round tokens). Default `(seq_len,) * batch` — today's configs are the
  uniform special case.
- Runtime-level: one tiny per-round input object `seq_meta_{s}_{r}`
  (int32) holding `cu_seqlens` (B+1 entries) — the same pattern as
  `tokens_{s}_{r}`; positions for rope derive from it on device (or a
  second small buffer `positions` (t,) built at the same time — decide
  at implementation by kernel convenience; positions-buffer keeps the
  rope kernel trivial).
- STATIC plans: the ragged pattern is fixed per plan (shapes/costs are
  plan constants — same contract as today). Per-step VARYING lengths
  require `--placement dynamic` (already documented for shape-unstable
  programs) and re-profiled costs; out of scope for v1. v1 = static
  ragged pattern, uniform default.

## Op conversions

1. **rope (fwd/bwd + eager)**: kernel takes a `positions` int32 tensor
   instead of `seq_len`; `pos = tl.load(positions + row)`. Registry
   signature change; llama3/qwen3/qwen35-attn blocks pass the round's
   positions view. (qwen35 partial-rope wrapper unchanged otherwise.)
2. **flash attention**: varlen invocation — aten
   `_flash_attention_forward/backward` with `cu_seq_q/cu_seq_k` +
   `max_seqlen` (the flash-attn varlen form; the M3 interop notes
   already map the dense form's philox/cum_seq quirks). Bonuses:
   varlen flash takes GQA natively → drop the `repeat_interleave` k/v
   materialization; lse shape becomes `(h, t)`.
   - Fast path: when `len(set(seq_lens)) == 1` KEEP today's dense
     reshape path (uniform batches — all current sweeps) unless a
     microbench shows varlen-flash parity at seq 1024 (llama3 8k-token
     rounds); decide with data, keep both behind `ops.flash_*`.
3. **qwen35 conv/fla**: already varlen — swap `_cu_seqlens(dims,...)`'s
   arithmetic construction for the `seq_meta` object so ragged rounds
   are honored (it currently assumes uniform `tokens/seq_len` splits).
4. **loss/CE + embed**: token-local, no change. Cross-sequence loss
   normalization stays mean-over-tokens (matches goldens).

## Correctness surface

- Goldens: accept `seq_lens`, build block-diagonal causal masks; rope
  positions reset per sequence. (Golden attention is exact-reference —
  masks are cheap at tiny scale.)
- Tests per family: (a) rope positions parity vs eager reference on a
  ragged pattern; (b) flash varlen vs dense at uniform lengths
  (numerical parity + lse layout); (c) ragged E2E ladder-3 at tiny
  scale — mixed lengths e.g. (700, 324) — loss + grads vs golden;
  (d) qwen35's existing batch=2 E2E extends to a non-uniform split.
- Profile signatures: task cost depends on the ragged PATTERN (per-seq
  quadratic attention at fixed t), so the seq_lens tuple must enter the
  profile/plan signature hash (uniform default hashes as today —
  cache-compatible).

## Interaction with other work

- Orthogonal to the dtype policy (task #2); sequencing: dtypes first
  (Shein's order). Both touch goldens/blocks — rebase whichever lands
  second.
- M5.2 findings: no perf motive here — the varlen kernels measured at
  parity. This is a semantics/capability feature (true packed-sequence
  training), not an optimization.
