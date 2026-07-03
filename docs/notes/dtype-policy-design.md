# Design: first-class user-supplied dtypes (per param / grad / opt state)

Requested by Shein 2026-07-03: stop assuming bf16 — users specify the
dtype per parameter, per parameter-gradient, and per optimizer-state
entry.

**Status: IMPLEMENTED (2026-07-03).** All three families; suite 153
green including the mixed-policy E2E gates
(tests/training/test_dtype_policy_e2e.py: fp32 norm weights + fp32
moments + qwen35 fp32 A_log/dt_bias, real engine vs golden). Usage:

```python
from dataflow.tasks.layouts import DTypePolicy, ParamDTypes
cfg = ShapedLlamaConfig(..., dtypes=DTypePolicy(
    default=ParamDTypes(opt="fp32"),               # fp32 AdamW moments
    overrides=(("*_norm_w", ParamDTypes(param="fp32", grad="fp32", opt="fp32")),),
))
```

Field-name patterns are fnmatch, first match wins; embed/head tables are
addressed "embed.w" / "head.w" / "head.final_norm_w" (both pack a field
named "w"); qwen3.5 TIED embeddings pack the head layout into W_embed and
stay addressed as head.*.

**Depth-dependent policies (Shein 2026-07-03 follow-up, same day):**
`layer_overrides` on DTypePolicy — ordered `(layer-index tuple,
sub-policy)` entries; the first entry containing the layer wins and its
sub-policy answers ALL lookups for that layer (no fallthrough into the
outer overrides); loose objects (embed/head) always use the outer
policy. Layer indices are explicit ints (`tuple(range(4))` for "first
four layers"):

```python
DTypePolicy(layer_overrides=(
    ((0, 31), DTypePolicy(default=ParamDTypes(opt="fp32"),
                          overrides=(("*_norm_w", FP32_ALL),))),
))
```

Per-layer dtypes mean per-layer packed SIZES, so the whole path is
layer-resolved: layout builders take `layer=`; block executables derive
the layer from their `W_{i}` object (`_Base.layer_of(task)` →
`wl_for/gl_for`); AdamWStep and the qwen3.5 per-kind optimizer dispatch
build the layout at the task's layer (size assert stays the tripwire);
lowerings size every W_i/dW_i/O_i from its own layer's layouts
(`_size_of_factory` in all three families); fills and goldens unpack
per layer. Gates: `test_depth_dependent_layer_sizes_diverge` (W_0/dW/O
bytes differ from layer 1's) + llama and qwen35 depth-dependent
model-steps vs golden. v1 validated recipes: fp32 on elementwise
fields (*_norm_w, A_log, dt_bias) and fp32 moments anywhere; fp32 on
GEMM/table fields is NOT plumbed (matmul dtype mismatch raises loudly —
the cast-at-use cost model needs its own decision). Grad-accum caveat:
the runtime accumulates dW in grad STORAGE dtype every round; the golden
accumulates in autograd (leaf) precision and rounds at the update — at
ga>1 with grad="bf16" these differ within ladder tolerance (documented,
tolerance-covered). The historical flat-O mapping is gone: O is
per-field [m_f | v_f] pairs (padding never touched — this also retired
the benign NaN-in-padding poison artifact).

## Where the bf16 assumption lives today (audit, 2026-07-03)

- `PackedLayout`/`Field` (tasks/layouts.py) are ALREADY per-field dtype
  aware ("bf16"/"fp32"/"int32" + byte math). Alignment padding for mixed
  dtypes works (qwen35 packs int-free mixed-width fields; the 8-byte
  padding-gap semantics under poison are understood). Nothing to change
  at this layer.
- The bf16 assumption is in the USERS of the layer:
  - every weight/grad field is emitted "bf16" by the lowerings
    (A_log/dt_bias included — stored bf16, `.float()`ed at use);
  - `AdamWStep` (tasks/llama3_blocks.py) views the WHOLE packed W buffer
    as flat bf16 (`elems = size_bytes // 2`), dW likewise;
  - `adamw_state_layout(elems)` pins moments to 2 x bf16 (fp32 in
    registers only, bf16 STORAGE round-trip — a deliberate M3 semantics
    choice mirrored by the goldens);
  - goldens encode bf16 leaf + bf16-moment rounding;
  - the adamw kernels themselves are already dtype-generic
    (`ptr.dtype.element_ty` casts in the triton kernel; eager follows
    view dtypes).

Reference concepts (flextrain docs/dtypes.md — concepts only): four
dtype ROLES per parameter (compute / master / grad / opt_state) with
casts at buffer boundaries.

## Policy surface

New `dataflow/training/dtypes.py`:

```python
@dataclass(frozen=True)
class ParamDTypes:
    param: str = "bf16"   # W field storage = what device kernels read
    grad:  str = "bf16"   # dW field storage
    opt:   str = "bf16"   # AdamW m and v storage

@dataclass(frozen=True)
class DTypePolicy:
    default: ParamDTypes = ParamDTypes()
    # fnmatch patterns over packed FIELD names ("w_qkvz", "A_log",
    # "*_norm_w", ...), first match wins; the field name is the
    # user-visible unit of "a parameter" everywhere else already
    overrides: tuple[tuple[str, ParamDTypes], ...] = ()
    def for_field(self, name: str) -> ParamDTypes: ...
```

`ShapedLlamaConfig / ShapedQwen3Config / ShapedQwen35Config` gain
`dtypes: DTypePolicy = DTypePolicy()`. Default policy == today's
behavior bit-for-bit at the tensor level (layout hashes DO change, see
Tests).

Explicitly deferred (documented, not built):
- a distinct MASTER dtype (flextrain's role 2). In our IR the backing
  copy of W IS the master; master != param requires cast-on-transfer in
  the engine (h2d cast kernels, new transfer semantics). Phase 2 if
  wanted.
- activation dtype policy (compute stays bf16; fp32 rstd side-buffers
  as today). Orthogonal to this feature.
- fp8 (needs scaling machinery, out of scope).
- validated dtypes: bf16/fp32 (fp16 parses but numerics unvalidated).

## Changes by layer

1. **Lowerings** (llama3/qwen3/qwen35): weight-layout builders take the
   policy; each trainable field's dtype = `policy.for_field(name).param`;
   the dW layout mirrors W field-by-field at `.grad`; the O layout
   becomes PER-FIELD `[m_<f>, v_<f>]` pairs at `.opt` (replacing
   `adamw_state_layout(flat elems)`). Sizes remain packed-bytes truth →
   `apply_exact_sizes` and placement flow with zero changes.
2. **AdamWStep exec**: iterate weight-layout fields; view w/g/m/v per
   field at their own dtypes/offsets; one `adamw_step` kernel launch per
   field (~10/block, ~400 launches/step — noise next to 585 tasks).
   Fast path: when every field shares one ParamDTypes and the layout is
   gap-free, keep today's single flat launch. Fixes as a side effect:
   AdamW updating padding bytes from undefined dW padding (the benign
   NaN-in-padding poison artifact) — per-field views never touch gaps.
3. **Blocks**: weight views already come typed from the layout. GEMM
   fields stored fp32 get a `.to(bf16)` at use (documented cost:
   materialized cast per round — the SUPPORTED recipes are elementwise
   fields: `*_norm_w`, `A_log`, `dt_bias`, gates). Elementwise ops
   (`rmsnorm_*`, gated norm, adamw) are dtype-generic already or cast
   internally to fp32 registers.
4. **Goldens** (all three families): storage-rounding parameterized by
   the same policy — leaves round-trip through `param` dtype after init
   and each update, moments through `opt`, grads accumulate at compute
   precision then store at `grad` (embed-host bf16-accum precedent).
5. **Kernels**: adamw triton/eager are dtype-generic; add a mixed-dtype
   kernel test (fp32 w/m/v vs fp32 reference; bf16 g).

## Tests / gates

- test_kernels: adamw at fp32 and mixed dtypes.
- Ladder 3 (tiny llama + tiny qwen35) under a mixed policy —
  `*_norm_w` param fp32, A_log/dt_bias param+opt fp32, default opt fp32
  — vs goldens with the same comparators as the bf16 ladders.
- Lowering-stability tripwires (tests/training/test_lowering_stability.py):
  hashes WILL change (O layout restructure). Update constants in the
  same commit with a note — that test exists to catch unintended drift;
  this is intended drift.
- Plan-invariance / poison / interleave: unchanged semantics, rerun.
- Profile cache: signatures change wherever layouts change → natural
  cache split; document that non-default policies profile fresh.

## Order of work

dtypes.py → llama lowering + AdamW exec + golden → llama ladder green →
qwen3 + qwen35 lowerings/goldens → full suites → extending.md §dtypes +
this note updated to DONE.
