"""Family-generic *shaped* training-program builder.

ONE chain grammar serves every model family: rounds of [embed_fwd,
{kind}_fwd x L, head_loss (fused head+CE+head-bwd, token-chunked),
({kind}_recompute?, {kind}_bwd) x L reversed, embed_bwd] plus optimizer tasks (interleaved or
tail placement), grad-accum mutation patterns, recompute rewrites, and
tied-embedding wiring. Object ids and task ids are family-INVARIANT
(``W_{i}``, ``A_{s}_{r}_{i}``, ``block_fwd_{s}_{r}_{i}``, ...) — the whole
planner/runtime/tooling stack keys on this grammar.

A family supplies:
  - a Shaped*Config obeying the shared protocol (n_layers, d_model,
    vocab_size, tokens, seq_len, batch, grad_accum_rounds, num_steps,
    optimizer_placement, embed_params, head_params, tied_embeddings?);
  - one ``LayerKindSpec`` per layer kind (sizes + roofline cost seeds +
    compute_block_key prefix) and a ``kind_of(layer)`` table — uniform
    dense-transformer families can seed theirs with
    ``roofline_block_kind_spec``.

No family is special: llama3/qwen3/qwen3.5 all pass explicit kinds
through this one builder (see their shaped_* modules). Cost seeds are
analytic rooflines only — real planning replaces them with measured
costs (training/profiling.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from dataflow.core import (
    ObjectSpec,
    OutputSpec,
    Program,
    RecomputeOption,
    RecomputeRewrite,
    TaskSpec,
    TensorMeta,
    dtype_nbytes,
)

BF16 = 2  # bytes


@dataclass(frozen=True)
class ShapedHardware:
    """Roofline knobs for runtime estimates (plausible RTX 5090 defaults)."""

    peak_bf16_tflops: float = 200.0
    matmul_eff: float = 0.75
    attn_eff: float = 0.55
    mem_bw_gbs: float = 1500.0
    mem_eff: float = 0.8
    pcie_gbs: float = 55.0

    def matmul_us(self, flops: float, bytes_: float) -> float:
        math_us = flops / (self.peak_bf16_tflops * self.matmul_eff * 1e12) * 1e6
        mem_us = bytes_ / (self.mem_bw_gbs * self.mem_eff * 1e9) * 1e6
        return max(math_us, mem_us, 1.0)

    def attn_us(self, flops: float, bytes_: float) -> float:
        math_us = flops / (self.peak_bf16_tflops * self.attn_eff * 1e12) * 1e6
        mem_us = bytes_ / (self.mem_bw_gbs * self.mem_eff * 1e9) * 1e6
        return max(math_us, mem_us, 1.0)

    def mem_us(self, bytes_: float) -> float:
        return max(bytes_ / (self.mem_bw_gbs * self.mem_eff * 1e9) * 1e6, 1.0)

    @property
    def pcie_bytes_per_us(self) -> int:
        return int(self.pcie_gbs * 1e9 / 1e6)


@dataclass(frozen=True)
class LayerKindSpec:
    """Everything the chain builder needs about ONE layer kind.

    Heterogeneous families (qwen3.5: DeltaNet + gated-attention layers)
    pass a spec per kind + a ``kind_of(layer)`` table; uniform families
    pass a single kind. Task IDS are kind-independent
    (``block_fwd_{s}_{r}_{i}``); only compute_block_keys and sizes/costs
    vary, so planner/runtime/tooling conventions hold.
    """

    key_prefix: str          # compute_block_keys: {prefix}_fwd/_bwd/_recompute
    w_bytes: int
    a_bytes: int
    fwd_us: float
    bwd_us: float
    recompute_us: float
    optimizer_us: float
    fwd_subops: list
    bwd_subops: list
    recompute_subops: list
    optimizer_subops: list
    # per-layer METADATA object (M_{s}_{r}_{i}) bytes. M holds forward
    # artifacts that are expensive or fragile to re-derive — discrete
    # decisions like expert-routing packs and top-k selections — packed
    # in one layout. Emitted by fwd, consumed VERBATIM by recompute and
    # bwd; never a recompute candidate (recompute repopulates ONLY the
    # A objects). 0 = the kind has no metadata. A normal dataflow object
    # in every other respect (placement/offload/transfers).
    meta_bytes: int = 0


class LooseCosts:
    """Roofline seeds for the family-INVARIANT loose tasks: embedding
    gather/scatter, LM head, CE loss, and the embed/head optimizers.
    Every family shares these formulas (they depend only on tokens,
    d_model, vocab_size and the embed/head param counts)."""

    def __init__(self, cfg, hw: ShapedHardware) -> None:
        t, d = cfg.tokens, cfg.d_model

        embed_bytes = BF16 * 2.0 * t * d + 4.0 * t
        self.embed_fwd_us = hw.mem_us(embed_bytes)
        self.embed_bwd_us = hw.mem_us(2.0 * embed_bytes)

        head_flops = 2.0 * t * d * cfg.vocab_size
        head_bytes = BF16 * (d * cfg.vocab_size + t * (d + cfg.vocab_size))
        # head_loss = fused final-norm + head GEMM + CE + head backward
        # (3x the head GEMM flops: fwd + dW + dX) — one task, chunked over
        # tokens so no (t, vocab) tensor is ever materialized
        self.head_loss_us = (
            hw.matmul_us(head_flops, head_bytes)
            + hw.mem_us(BF16 * 2.0 * t * cfg.vocab_size)
            + hw.matmul_us(2.0 * head_flops, 2.0 * head_bytes)
        )

        def opt_us(params: int) -> float:
            # read W, dW, m, v; write W, m, v (all bf16 here)
            return hw.mem_us(BF16 * 7.0 * params)

        self.optimizer_us = {
            "embed": opt_us(cfg.embed_params),
            "head": opt_us(cfg.head_params),
        }

        self.subops = {
            "embed_fwd": [{"kind": "roofline", "name": "gather", "flops": 0, "memory_bytes": int(embed_bytes), "efficiency": "memory"}],
            "embed_bwd": [{"kind": "roofline", "name": "scatter_accum", "flops": 0, "memory_bytes": int(2 * embed_bytes), "efficiency": "memory"}],
            "head_loss": [
                {"kind": "roofline", "name": "lm_head", "flops": int(head_flops), "memory_bytes": int(head_bytes), "efficiency": "matmul"},
                {"kind": "roofline", "name": "ce_loss", "flops": 0, "memory_bytes": int(BF16 * 2 * t * cfg.vocab_size), "efficiency": "memory"},
                {"kind": "roofline", "name": "lm_head_bwd", "flops": int(2 * head_flops), "memory_bytes": int(2 * head_bytes), "efficiency": "matmul"},
            ],
            "optimizer_embed": [{"kind": "roofline", "name": "adamw", "flops": 0, "memory_bytes": int(BF16 * 7 * cfg.embed_params), "efficiency": "memory"}],
            "optimizer_head": [{"kind": "roofline", "name": "adamw", "flops": 0, "memory_bytes": int(BF16 * 7 * cfg.head_params), "efficiency": "memory"}],
        }


def roofline_block_kind_spec(cfg, hw: ShapedHardware, *,
                             key_prefix: str = "block") -> LayerKindSpec:
    """Dense-transformer block roofline (attention + SwiGLU MLP): the
    shared cost seed for uniform families (llama3, qwen3). The config
    supplies block_params / kv_dim / d_ff / saved_ctx_width."""
    t, d, seq = cfg.tokens, cfg.d_model, cfg.seq_len

    matmul_params = cfg.block_params - 2 * d  # exclude norm vectors
    mm_flops = 2.0 * t * matmul_params
    mm_bytes = BF16 * (matmul_params + t * (2 * d + 2 * cfg.kv_dim + 2 * cfg.d_ff))
    # causal attention: ~0.5 * 4 * tokens * seq * d flops (fwd)
    attn_flops = 2.0 * t * seq * d
    attn_bytes = BF16 * t * (2 * d + 2 * cfg.kv_dim)

    fwd_us = hw.matmul_us(mm_flops, mm_bytes) + hw.attn_us(attn_flops, attn_bytes)
    # backward: dgrad + wgrad matmuls (2x fwd matmul flops), attention bwd ~2.5x fwd
    bwd_us = hw.matmul_us(2.0 * mm_flops, 2.0 * mm_bytes) \
        + hw.attn_us(2.5 * attn_flops, 2.0 * attn_bytes)

    fwd_subops = [
        {"kind": "roofline", "name": "block_matmuls", "flops": int(mm_flops), "memory_bytes": int(mm_bytes), "efficiency": "matmul"},
        {"kind": "roofline", "name": "attention", "flops": int(attn_flops), "memory_bytes": int(attn_bytes), "efficiency": "attention"},
    ]
    bwd_subops = [
        {"kind": "roofline", "name": "block_matmuls_bwd", "flops": int(2 * mm_flops), "memory_bytes": int(2 * mm_bytes), "efficiency": "matmul"},
        {"kind": "roofline", "name": "attention_bwd", "flops": int(2.5 * attn_flops), "memory_bytes": int(2 * attn_bytes), "efficiency": "attention"},
    ]
    return LayerKindSpec(
        key_prefix=key_prefix,
        w_bytes=BF16 * cfg.block_params,
        a_bytes=BF16 * (t * cfg.saved_ctx_width),
        fwd_us=fwd_us,
        bwd_us=bwd_us,
        recompute_us=fwd_us,
        optimizer_us=hw.mem_us(BF16 * 7.0 * cfg.block_params),
        fwd_subops=fwd_subops,
        bwd_subops=bwd_subops,
        recompute_subops=list(fwd_subops),
        optimizer_subops=[{"kind": "roofline", "name": "adamw", "flops": 0, "memory_bytes": int(BF16 * 7 * cfg.block_params), "efficiency": "memory"}],
    )


@dataclass(frozen=True)
class MetaShare:
    """Cross-layer metadata sharing: every layer in ``consumers`` also
    consumes the PRODUCER layer's M object (fwd, recompute and bwd) —
    e.g. GLM-5.2 IndexShare, where shared layers reuse the nearest full
    layer's selection. ``grad_bytes`` > 0 threads the backward companion
    ``dM_{s}_{r}_{producer}`` through the group's bwds in reverse layer
    order (created by the last consumer, mutated by the middles, consumed
    by the producer) — the generic shape of a cross-layer reduction
    target, accumulated exactly like dW under grad accumulation."""

    producer: int
    consumers: tuple[int, ...]
    grad_bytes: int = 0


def build_shaped_program(
    cfg,
    *,
    kinds: Mapping[str, LayerKindSpec],
    kind_of=None,
    family: str,
    hw: ShapedHardware | None = None,
    fast_memory_capacity: int | None = None,
    recompute_levels: Mapping[str, int] | None = None,
    name: str | None = None,
    meta_shared=None,
    indexer_only_objective: bool = False,
) -> Program:
    """Build the (bare) shaped program for the given recompute levels.

    ``kinds`` is REQUIRED — every family declares its layer kinds
    explicitly (uniform families pass one). ``meta_shared`` is a list of
    ``MetaShare`` for cross-layer metadata consumption (each layer's own
    M object comes from its kind's ``meta_bytes``). ``recompute_levels`` maps
    saved-context object id (``A_{s}_{r}_{i}``) to level 0 (save) or 1
    (recompute); missing ids default to 0. This function IS the
    ``build_variant`` for the recompute planner (wrap with
    ``functools.partial``). ``family`` stamps the program metadata and
    seeds the default name.
    """
    hw = hw or ShapedHardware()
    levels = dict(recompute_levels or {})
    meta_producer_of: dict[int, int] = {}
    meta_group_of: dict[int, tuple[int, ...]] = {}
    meta_grad_of: dict[int, int] = {}
    for share in (meta_shared or ()):
        mem = (share.producer,) + tuple(share.consumers)
        meta_group_of[share.producer] = mem
        meta_grad_of[share.producer] = share.grad_bytes
        for m in share.consumers:
            meta_producer_of[m] = share.producer
    loose = LooseCosts(cfg, hw)
    t, d = cfg.tokens, cfg.d_model

    def bf16(n_elems: int) -> int:
        return BF16 * n_elems

    w_embed = bf16(cfg.embed_params)
    w_head = bf16(cfg.head_params)
    y_bytes = bf16(t * d)
    ids_bytes = dtype_nbytes((t,), "int32")

    if kind_of is None:
        _default_kind = next(iter(kinds))
        kind_of = lambda i: _default_kind  # noqa: E731
    spec_of = lambda i: kinds[kind_of(i)]  # noqa: E731
    tied = bool(getattr(cfg, "tied_embeddings", False))

    initial: list[ObjectSpec] = []
    final_locations: dict[str, str] = {}

    def add_initial(oid: str, size: int, role: str, *, persist: bool = False, tensor: TensorMeta | None = None) -> None:
        initial.append(ObjectSpec(id=oid, size_bytes=size, location="backing", role=role, tensor=tensor))
        if persist:
            final_locations[oid] = "backing"

    # tied embeddings (config choice): ONE W_embed/O_embed pair serves both
    # the embedding and the LM head (the lowering sizes it as the packed
    # head layout [table | final_norm_w]); no W_head/O_head objects exist.
    add_initial("W_embed", w_embed, "parameter", persist=True,
                tensor=None if tied else TensorMeta(dtype="bf16", shape=(cfg.vocab_size, d)))
    add_initial("O_embed", 2 * w_embed, "optimizer_state", persist=True)
    for i in range(cfg.n_layers):
        add_initial(f"W_{i}", spec_of(i).w_bytes, "parameter", persist=True)
        add_initial(f"O_{i}", 2 * spec_of(i).w_bytes, "optimizer_state", persist=True)
    if not tied:
        add_initial("W_head", w_head, "parameter", persist=True,
                    tensor=TensorMeta(dtype="bf16", shape=(cfg.vocab_size, d)))
        add_initial("O_head", 2 * w_head, "optimizer_state", persist=True)
    for s in range(cfg.num_steps):
        for r in range(cfg.grad_accum_rounds):
            # The chain's very first task consumes tokens_0_0; following the
            # simulator's own builder convention it starts on fast. Everything
            # else starts on backing and is placed/prefetched by the planner.
            first = s == 0 and r == 0
            initial.append(ObjectSpec(
                id=f"tokens_{s}_{r}", size_bytes=ids_bytes,
                location="fast" if first else "backing",
                role="input", tensor=TensorMeta(dtype="int32", shape=(t,)),
            ))
            add_initial(f"targets_{s}_{r}", ids_bytes, "input", tensor=TensorMeta(dtype="int32", shape=(t,)))

    tasks: list[TaskSpec] = []
    rewrites: list[RecomputeRewrite] = []

    def task(tid: str, block: str, inputs: list[str], outputs: list[OutputSpec], runtime_us: float,
             *, mutates: tuple[str, ...] = (), group: str = "compute", params: dict | None = None,
             subops: list | None = None) -> None:
        tasks.append(TaskSpec(
            id=tid,
            inputs=tuple(inputs),
            outputs=tuple(outputs),
            mutates=mutates,
            runtime_us=runtime_us,
            group=group,
            compute_block_key=block,
            block_params=params or {},
            metadata={"cost_subops": loose.subops[block] if subops is None else subops},
        ))

    if cfg.optimizer_placement not in ("interleaved", "tail"):
        raise ValueError(
            f"optimizer_placement must be 'interleaved' or 'tail', "
            f"got {cfg.optimizer_placement!r}"
        )
    interleaved = cfg.optimizer_placement == "interleaved"

    for s in range(cfg.num_steps):
        # Optimizer emitters: one per parameter family; ids/inputs identical
        # in both placements. In interleaved mode each fires right after the
        # final mutation of its gradient (last round's backward task), so O
        # prefetches and W/O writebacks overlap the rest of the backward
        # instead of draining serially after all compute is done.
        def opt_embed(s: int = s) -> None:
            task(f"optimizer_embed_{s}", "optimizer_embed",
                 ["W_embed", f"dW_embed_{s}", "O_embed"], [],
                 loose.optimizer_us["embed"], mutates=("W_embed", "O_embed"),
                 group="optimizer")

        def opt_block(i: int, s: int = s) -> None:
            sp = spec_of(i)
            task(f"optimizer_{s}_{i}", "optimizer_block",
                 [f"W_{i}", f"dW_{s}_{i}", f"O_{i}"], [],
                 sp.optimizer_us, mutates=(f"W_{i}", f"O_{i}"),
                 group="optimizer", params={"layer": i},
                 subops=sp.optimizer_subops)

        def opt_head(s: int = s) -> None:
            if tied:
                return  # optimizer_embed covers the shared W_embed/O_embed
            task(f"optimizer_head_{s}", "optimizer_head",
                 ["W_head", f"dW_head_{s}", "O_head"], [],
                 loose.optimizer_us["head"], mutates=("W_head", "O_head"),
                 group="optimizer")

        for r in range(cfg.grad_accum_rounds):
            first_round = r == 0
            last_round = r == cfg.grad_accum_rounds - 1
            # ---- forward ----
            task(
                f"embed_fwd_{s}_{r}", "embed_fwd",
                [f"tokens_{s}_{r}", "W_embed"],
                [OutputSpec(id=f"y_embed_{s}_{r}", size_bytes=y_bytes, role="activation",
                            tensor=TensorMeta(dtype="bf16", shape=(t, d)))],
                loose.embed_fwd_us, group="forward",
            )
            for i in range(cfg.n_layers):
                sp = spec_of(i)
                a_id = f"A_{s}_{r}_{i}"
                level = levels.get(a_id, 0)
                x_id = f"y_embed_{s}_{r}" if i == 0 else f"y_{s}_{r}_{i - 1}"
                fwd_ins = [x_id, f"W_{i}"]
                outs = [OutputSpec(id=f"y_{s}_{r}_{i}", size_bytes=y_bytes, role="activation",
                                   tensor=TensorMeta(dtype="bf16", shape=(t, d)))]
                if level == 0:
                    outs.append(OutputSpec(id=a_id, size_bytes=sp.a_bytes, role="activation"))
                if sp.meta_bytes:
                    outs.append(OutputSpec(id=f"M_{s}_{r}_{i}",
                                           size_bytes=sp.meta_bytes,
                                           role="activation"))
                if i in meta_producer_of:
                    fwd_ins.append(f"M_{s}_{r}_{meta_producer_of[i]}")
                task(f"block_fwd_{s}_{r}_{i}", f"{sp.key_prefix}_fwd", fwd_ins, outs,
                     sp.fwd_us, group="forward", params={"layer": i},
                     subops=sp.fwd_subops)
                rewrites.append(RecomputeRewrite(
                    object_id=a_id,
                    f_task_id=f"block_fwd_{s}_{r}_{i}",
                    r_task_id=f"block_recompute_{s}_{r}_{i}",
                    options=(
                        RecomputeOption(level=0, saved_bytes=sp.a_bytes, recompute_us=0.0, label="save"),
                        RecomputeOption(level=1, saved_bytes=0, recompute_us=sp.recompute_us, label="recompute"),
                    ),
                    f_compute_block_key=f"{sp.key_prefix}_fwd",
                    r_compute_block_key=f"{sp.key_prefix}_recompute",
                    group_key=f"layer_{i}",
                ))
            last_y = f"y_{s}_{r}_{cfg.n_layers - 1}"

            # ---- fused head + loss + head backward (ONE task) ----
            # Chunked over tokens inside the executable: the (t, vocab)
            # logits/dlogits never exist as IR objects (or at all) — the
            # single biggest activation pair in a naive lowering.
            head_w = "W_embed" if tied else "W_head"
            head_dw = f"dW_embed_{s}" if tied else f"dW_head_{s}"
            head_inputs = [last_y, f"targets_{s}_{r}", head_w]
            head_outs = [
                OutputSpec(id=f"dy_{s}_{r}_{cfg.n_layers - 1}", size_bytes=y_bytes, role="gradient",
                           tensor=TensorMeta(dtype="bf16", shape=(t, d))),
                OutputSpec(id=f"loss_{s}_{r}", size_bytes=4, role="output",
                           tensor=TensorMeta(dtype="fp32", shape=(1,))),
            ]
            head_mutates: tuple[str, ...] = ()
            if first_round:
                # tied: head_loss runs before embed_bwd in the round, so IT
                # creates the shared dW_embed; embed_bwd then accumulates.
                head_outs.append(OutputSpec(id=head_dw, size_bytes=w_head, role="gradient"))
            else:
                head_inputs.append(head_dw)
                head_mutates = (head_dw,)
            task(f"head_loss_{s}_{r}", "head_loss", head_inputs, head_outs,
                 loose.head_loss_us, mutates=head_mutates, group="backward")
            final_locations[f"loss_{s}_{r}"] = "backing"
            if interleaved and last_round:
                opt_head()  # dW_head_{s} saw its final mutation just now

            for i in reversed(range(cfg.n_layers)):
                a_id = f"A_{s}_{r}_{i}"
                x_id = f"y_embed_{s}_{r}" if i == 0 else f"y_{s}_{r}_{i - 1}"
                sp = spec_of(i)
                meta_ins = []
                if sp.meta_bytes:
                    meta_ins.append(f"M_{s}_{r}_{i}")
                if i in meta_producer_of:
                    meta_ins.append(f"M_{s}_{r}_{meta_producer_of[i]}")
                if levels.get(a_id, 0) == 1:
                    task(f"block_recompute_{s}_{r}_{i}", f"{sp.key_prefix}_recompute",
                         [x_id, f"W_{i}"] + meta_ins,
                         [OutputSpec(id=a_id, size_bytes=sp.a_bytes, role="activation")],
                         sp.recompute_us, group="recompute", params={"layer": i},
                         subops=sp.recompute_subops)
                bwd_inputs = [f"dy_{s}_{r}_{i}", a_id, x_id, f"W_{i}"] + meta_ins
                outs = [OutputSpec(id=(f"dy_embed_{s}_{r}" if i == 0 else f"dy_{s}_{r}_{i - 1}"),
                                   size_bytes=y_bytes, role="gradient",
                                   tensor=TensorMeta(dtype="bf16", shape=(t, d)))]
                mutates: tuple[str, ...] = ()
                if first_round:
                    outs.append(OutputSpec(id=f"dW_{s}_{i}", size_bytes=sp.w_bytes, role="gradient"))
                else:
                    bwd_inputs.append(f"dW_{s}_{i}")
                    mutates = (f"dW_{s}_{i}",)
                ld_grp = meta_producer_of.get(i, i)
                if ld_grp in meta_group_of and meta_grad_of.get(ld_grp, 0):
                    mem = meta_group_of[ld_grp]
                    gid = f"dM_{s}_{r}_{ld_grp}"
                    if i == mem[-1]:
                        outs.append(OutputSpec(id=gid,
                                               size_bytes=meta_grad_of[ld_grp],
                                               role="gradient"))
                    elif i == ld_grp:
                        bwd_inputs.append(gid)
                    elif i in mem:
                        bwd_inputs.append(gid)
                        mutates = mutates + (gid,)
                task(f"block_bwd_{s}_{r}_{i}", f"{sp.key_prefix}_bwd", bwd_inputs, outs,
                     sp.bwd_us, mutates=mutates, group="backward", params={"layer": i},
                     subops=sp.bwd_subops)
                if interleaved and last_round:
                    opt_block(i)  # dW_{s}_{i} is final; W_i still resident from bwd

            embed_bwd_inputs = [f"dy_embed_{s}_{r}", f"tokens_{s}_{r}"]
            embed_outs: list[OutputSpec] = []
            embed_mutates: tuple[str, ...] = ()
            if first_round and not tied:
                embed_outs.append(OutputSpec(id=f"dW_embed_{s}", size_bytes=w_embed, role="gradient"))
            else:
                embed_bwd_inputs.append(f"dW_embed_{s}")
                embed_mutates = (f"dW_embed_{s}",)
            task(f"embed_bwd_{s}_{r}", "embed_bwd", embed_bwd_inputs, embed_outs,
                 loose.embed_bwd_us, mutates=embed_mutates, group="backward")
            if interleaved and last_round:
                opt_embed()  # embed_bwd is the round's last task; embed opt closes the step

        if not interleaved:
            # ---- legacy tail placement: all optimizers after all rounds ----
            opt_embed()
            for i in range(cfg.n_layers):
                opt_block(i)
            opt_head()

    if indexer_only_objective:
        # dense warm-up: the objective is the indexer KL alone. The
        # specialized surgery (no head/targets/dy; loss = KL accumulator
        # threaded through contributor backwards) lives in
        # warmup_program.py — this builder stays the common case.
        from .warmup_program import to_indexer_warmup

        followers = frozenset(meta_producer_of)
        _warmup_transform = lambda prog: to_indexer_warmup(prog, followers=followers)  # noqa: E731
    else:
        _warmup_transform = None

    label = name or (
        f"{family}-{cfg.n_layers}L-d{cfg.d_model}-s{cfg.seq_len}-b{cfg.batch}"
        f"-r{cfg.grad_accum_rounds}-steps{cfg.num_steps}"
    )
    program = Program(
        name=label,
        initial_objects=tuple(initial),
        tasks=tuple(tasks),
        final_locations=final_locations,
        fast_memory_capacity=fast_memory_capacity,
        backing_memory_capacity=None,
        bandwidth_from_slow=hw.pcie_bytes_per_us,
        bandwidth_to_slow=hw.pcie_bytes_per_us,
        recompute_rewrites=tuple(rewrites),
        metadata={
            "family": family,
            "primary_unit": "tokens",
            "primary_count": float(cfg.tokens * cfg.grad_accum_rounds * cfg.num_steps),
            "config": {
                "n_layers": cfg.n_layers, "d_model": cfg.d_model, "n_heads": cfg.n_heads,
                "n_kv_heads": cfg.n_kv_heads, "d_ff": cfg.d_ff, "vocab_size": cfg.vocab_size,
                "seq_len": cfg.seq_len, "batch": cfg.batch,
                "grad_accum_rounds": cfg.grad_accum_rounds, "num_steps": cfg.num_steps,
            },
        },
    )
    return _warmup_transform(program) if _warmup_transform else program
