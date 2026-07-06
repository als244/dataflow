"""MoE layer spec + byte-level layout helpers (torch-free).

This module is the torch-free half of the pluggable MoE-MLP module
(`dataflow.tasks.moe`): the `MoESpec` configuration knob set and the
weight/context field-spec builders that family layout builders compose.

Global-vs-local accounting rule (the expert-parallelism seam):

- ``n_experts`` (global E) is used ONLY for routing semantics — router
  width, softmax space, aux-loss normalization, the sort id-space.
- Everything that SIZES or PRICES expert state reads ``n_local_experts``
  / ``expert_ids``: the stacked weight/grad/opt fields are
  ``(E_local, ...)`` (slot j holds global expert ``expert_ids[j]`` — that
  ordering IS the load/checkpoint mapping), context segment offsets are
  local, and roofline flops use ``moe_local_rows``.

v1 runs single-rank only (``expert_ids=None`` = this rank holds all E);
partial ownership is fully plumbed through sizing/init and unit-tested,
but program lowering rejects it until a multi-rank runtime exists.
"""
from __future__ import annotations

from dataclasses import dataclass

_ROUTING_MODES = ("topk_then_softmax", "softmax_then_topk")

# v1 dtype pins: the knobs exist (they are the quantization seam — fp8
# dispatch later), but only the flextrain-parity combination is plumbed.
_V1_DISPATCH_DTYPES = ("bf16",)
_V1_COMBINE_DTYPES = ("fp32",)


@dataclass(frozen=True)
class MoESpec:
    """Structural + routing + precision knobs of one MoE MLP.

    routing_mode:
        "topk_then_softmax"  — pick top-K logits, softmax over the K
                               (weights sum to 1; norm_topk_prob=True).
        "softmax_then_topk"  — full-E softmax, take top-K probs
                               UNnormalized (weights sum <= 1;
                               norm_topk_prob=False; OLMoE).
        Tie-break is ALWAYS smallest expert index (both modes, pinned by
        ladder tests; torch.topk's CUDA tie-break violates this).
    aux_coef:
        Load-balance loss coefficient (alpha). The loss is GRADIENT-
        INJECTED per layer per round (never added to the scalar loss):
        dz[t,e] += (alpha*E/T_r) * p[t,e] * (f_e - <f, p_t>), with p the
        full-E fp32 softmax of the saved logits and f detached per-round
        counts. 0 disables the injection.
    dispatch_dtype / combine_dtype:
        Dtype of the permuted token buffers (xp/h13/yp) and of the
        combine accumulator. v1 pins ("bf16", "fp32") and raises loudly
        on anything else — same deliberate-unplumbed convention as fp32
        GEMM params in the dtype policy.
    expert_ids:
        Global expert ids THIS rank holds; None = all (the trivial
        single-rank placement). Must be unique and in [0, n_experts).
    """

    n_experts: int
    top_k: int
    d_ff_expert: int
    routing_mode: str = "topk_then_softmax"
    aux_coef: float = 0.0
    n_shared_experts: int = 0
    d_ff_shared: int = 0
    dispatch_dtype: str = "bf16"
    combine_dtype: str = "fp32"
    expert_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.routing_mode not in _ROUTING_MODES:
            raise ValueError(
                f"routing_mode {self.routing_mode!r} not in {_ROUTING_MODES}"
            )
        if not (0 < self.top_k <= self.n_experts):
            raise ValueError(
                f"top_k {self.top_k} must be in (0, n_experts={self.n_experts}]"
            )
        if self.n_shared_experts not in (0, 1):
            raise ValueError("v1 supports n_shared_experts in {0, 1}")
        if self.n_shared_experts and self.d_ff_shared <= 0:
            raise ValueError("shared expert requires d_ff_shared > 0")
        if self.dispatch_dtype not in _V1_DISPATCH_DTYPES:
            raise ValueError(
                f"dispatch_dtype {self.dispatch_dtype!r} not plumbed in v1 "
                f"(allowed: {_V1_DISPATCH_DTYPES})"
            )
        if self.combine_dtype not in _V1_COMBINE_DTYPES:
            raise ValueError(
                f"combine_dtype {self.combine_dtype!r} not plumbed in v1 "
                f"(allowed: {_V1_COMBINE_DTYPES})"
            )
        if self.expert_ids is not None:
            ids = tuple(self.expert_ids)
            if len(ids) == 0:
                raise ValueError("expert_ids must be non-empty (or None)")
            if len(set(ids)) != len(ids):
                raise ValueError("expert_ids must be unique")
            if not all(0 <= e < self.n_experts for e in ids):
                raise ValueError("expert_ids must lie in [0, n_experts)")

    # --- local-ownership accounting -------------------------------------------

    @property
    def n_local_experts(self) -> int:
        return self.n_experts if self.expert_ids is None else len(self.expert_ids)

    @property
    def is_partial(self) -> bool:
        return self.n_local_experts != self.n_experts

    def local_slot_of(self) -> dict[int, int]:
        """global expert id -> local weight slot (identity when not partial)."""
        ids = self.expert_ids or tuple(range(self.n_experts))
        return {e: j for j, e in enumerate(ids)}


def moe_local_rows(moe: MoESpec, tokens: int) -> int:
    """Grouped-GEMM row count for sizing/flops.

    Single-rank (not partial): EXACT — dropless routing places every one
    of tokens*top_k assignments locally. Partial ownership: balanced
    expectation; this is the single knob where the future EP receive-
    buffer capacity policy lands (capacity-bounded vs dynamic placement)
    — today it only sizes the unit-test layouts.
    """
    exact = tokens * moe.top_k
    if not moe.is_partial:
        return exact
    return -(-exact * moe.n_local_experts // moe.n_experts)  # ceil


def moe_weight_specs(dims, moe: MoESpec) -> list[tuple[str, tuple[int, ...]]]:
    """(name, shape) pairs for the MoE tail's weight fields, in layout order.

    Orientation is the repo convention (out = x @ w). ``w13_experts`` packs
    [x1 | x3] along the last dim — x1 (the silu input) in the FIRST F
    columns, x3 (the value) in the SECOND — the repo-wide packed-matrix
    convention (dense MLP/QKV convert later). Router is GLOBAL width;
    expert stacks are LOCAL (see module docstring).
    """
    d = dims.d_model
    e_loc, f = moe.n_local_experts, moe.d_ff_expert
    specs: list[tuple[str, tuple[int, ...]]] = [
        ("w_router", (d, moe.n_experts)),
        ("w13_experts", (e_loc, d, 2 * f)),
        ("w2_experts", (e_loc, f, d)),
    ]
    if moe.n_shared_experts:
        fs = moe.d_ff_shared
        specs += [
            ("w_shared_gate", (d, moe.n_shared_experts)),
            ("w_s13", (d, 2 * fs)),
            ("w_s2", (fs, d)),
        ]
    return specs


def moe_context_specs(dims, moe: MoESpec) -> list[tuple[str, tuple[int, ...], str]]:
    """(name, shape, dtype) triples for the MoE tail's saved-context fields.

    Saved: routing decision (logits + weights + ids + sort order + local
    segment offsets) and the pre-activations h13 (+ shared s13/gate_pre).
    NOT saved (re-derived in bwd from the saved order): xp, yp, sact,
    slot_of (the inverse permutation), dprob.
    """
    t = dims.tokens
    rows = moe_local_rows(moe, t)
    specs: list[tuple[str, tuple[int, ...], str]] = [
        ("router_logits", (t, moe.n_experts), "bf16"),
        ("route_w", (t, moe.top_k), "bf16"),
        ("route_ids", (t, moe.top_k), "int32"),
        ("route_order", (rows,), "int32"),
        ("route_offsets", (moe.n_local_experts + 1,), "int32"),
        ("h13", (rows, 2 * moe.d_ff_expert), moe.dispatch_dtype),
    ]
    if moe.n_shared_experts:
        specs += [
            ("gate_pre", (t, moe.n_shared_experts), "bf16"),
            ("s13", (t, 2 * moe.d_ff_shared), "bf16"),
        ]
    return specs
