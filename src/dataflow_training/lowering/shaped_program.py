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
    compute_block_key prefix) and per-layer ``layer_kinds`` — uniform
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
    pass a spec per kind + per-layer ``layer_kinds``; uniform families
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
    # per-layer AuxTemp object (AuxTemp_{s}_{r}_{i}) bytes: forward
    # artifacts that are expensive or fragile to re-derive — discrete
    # decisions like expert-routing packs and top-k selections — packed
    # in one layout. Emitted by fwd, consumed VERBATIM by recompute and
    # bwd; never a recompute candidate (recompute repopulates ONLY the
    # A objects). 0 = the kind has no AuxTemp. A normal dataflow object
    # in every other respect (placement/offload/transfers).
    aux_temp_bytes: int = 0
    # per-layer PERSISTENT Aux object (Aux_{i}) bytes: the per-step +
    # all-of-training expert-assignment counts (host-backed resident like
    # W/O; zeroed at round 0 by the round prologue, accumulated by every
    # round's fwd, read by the LAST round's bwd). 0 = no Aux object.
    aux_bytes: int = 0


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

        # policy-consulted embed/head seeds over the canonical tables
        # ((V, d) + the head's final norm) — legacy_params pins adamw
        # byte-identity to the historical cfg.*_params expressions.
        # Exactness boundary: exotic per-field routing on family-extra
        # embed fields (e.g. a muon'd position table) is not modeled.
        embed_opt_us, embed_opt_sub = optimizer_cost_seed(
            cfg, hw, [("w", (cfg.vocab_size, d))], ns="embed",
            legacy_params=cfg.embed_params)
        head_opt_us, head_opt_sub = optimizer_cost_seed(
            cfg, hw, [("w", (cfg.vocab_size, d)), ("final_norm_w", (d,))],
            ns="head", legacy_params=cfg.head_params)
        self.optimizer_us = {
            "embed": embed_opt_us,
            "head": head_opt_us,
        }

        self.subops = {
            "embed_fwd": [{"kind": "roofline", "name": "gather", "flops": 0, "memory_bytes": int(embed_bytes), "efficiency": "memory"}],
            "embed_bwd": [{"kind": "roofline", "name": "scatter_accum", "flops": 0, "memory_bytes": int(2 * embed_bytes), "efficiency": "memory"}],
            "head_loss": [
                {"kind": "roofline", "name": "lm_head", "flops": int(head_flops), "memory_bytes": int(head_bytes), "efficiency": "matmul"},
                {"kind": "roofline", "name": "ce_loss", "flops": 0, "memory_bytes": int(BF16 * 2 * t * cfg.vocab_size), "efficiency": "memory"},
                {"kind": "roofline", "name": "lm_head_bwd", "flops": int(2 * head_flops), "memory_bytes": int(2 * head_bytes), "efficiency": "matmul"},
            ],
            "optimizer_embed": embed_opt_sub,
            "optimizer_head": head_opt_sub,
        }


OPT_TRAFFIC = {  # elements moved per parameter element, by optimizer rule
    "adamw": 7.0,   # read W, dW, m, v; write W, m, v
    "sgdm": 5.0,    # read W, dW, m; write W, m
    "muon": 5.0,    # m-only state; NS compute charged separately
    "sgd": 3.0,
    "frozen": 0.0,
}


def optimizer_cost_seed(cfg, hw: ShapedHardware, weight_fields,
                        ns: str | None = None,
                        layer: int | None = None,
                        legacy_params: int | None = None):
    """(optimizer_us, optimizer_subops) for ONE weight object under the
    config's opt_policy. All-adamw charges the plain 7x-traffic seed;
    ``legacy_params`` pins the exact param count the lowering-stability
    hashes expect where it differs from the field sum (the loose head
    seed counts the table only, not final_norm_w); block seeds' sums
    match and omit it. Muon fields charge m-only traffic PLUS the
    Newton-Schulz matmul flops (2-D directly; 3-D stacked experts per
    slice) — without this term, roofline muon plans under-charge the
    optimizer."""
    from dataflow_training.blocks.optim import resolve_opt_policy

    from .flops import muon_ns_flops

    policy = resolve_opt_policy(getattr(cfg, "opt_policy", None) or "adamw")
    key = (lambda n: f"{ns}.{n}") if ns else (lambda n: n)
    mem_elems = 0.0
    ns_flops = 0.0
    ns_elems = 0
    pure_adamw = True
    total = 0
    for name, shape in weight_fields:
        n = 1
        for s in shape:
            n *= int(s)
        total += n
        rule = policy.for_field(key(name), layer, shape)
        if rule != "adamw":
            pure_adamw = False
        mem_elems += OPT_TRAFFIC.get(rule, 7.0) * n
        if rule == "muon":
            if len(shape) == 2:
                ns_flops += muon_ns_flops(int(shape[0]), int(shape[1]))
                ns_elems += n
            elif len(shape) == 3:
                # stacked expert matrices: NS runs per expert slice
                ns_flops += int(shape[0]) * muon_ns_flops(int(shape[1]),
                                                          int(shape[2]))
                ns_elems += n
    if pure_adamw:
        legacy = legacy_params if legacy_params is not None else total
        return hw.mem_us(BF16 * 7.0 * legacy), [
            {"kind": "roofline", "name": "adamw", "flops": 0,
             "memory_bytes": int(BF16 * 7 * legacy), "efficiency": "memory"}]
    us = hw.mem_us(BF16 * mem_elems)
    subops = [{"kind": "roofline", "name": "optimizer", "flops": 0,
               "memory_bytes": int(BF16 * mem_elems),
               "efficiency": "memory"}]
    if ns_flops:
        ns_bytes = BF16 * 4.0 * ns_elems
        us += hw.matmul_us(ns_flops, ns_bytes)
        subops.append({"kind": "roofline", "name": "muon_ns",
                       "flops": int(ns_flops),
                       "memory_bytes": int(ns_bytes),
                       "efficiency": "matmul"})
    return us, subops


def roofline_block_kind_spec(cfg, hw: ShapedHardware, *,
                             key_prefix: str = "block",
                             weight_fields=None) -> LayerKindSpec:
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
    if weight_fields is not None:
        opt_us, opt_subops = optimizer_cost_seed(cfg, hw, weight_fields)
    else:
        opt_us = hw.mem_us(BF16 * 7.0 * cfg.block_params)
        opt_subops = [{"kind": "roofline", "name": "adamw", "flops": 0, "memory_bytes": int(BF16 * 7 * cfg.block_params), "efficiency": "memory"}]
    return LayerKindSpec(
        key_prefix=key_prefix,
        w_bytes=BF16 * cfg.block_params,
        a_bytes=BF16 * (t * cfg.saved_ctx_width),
        fwd_us=fwd_us,
        bwd_us=bwd_us,
        recompute_us=fwd_us,
        optimizer_us=opt_us,
        fwd_subops=fwd_subops,
        bwd_subops=bwd_subops,
        recompute_subops=list(fwd_subops),
        optimizer_subops=opt_subops,
    )


@dataclass(frozen=True)
class AuxShare:
    """Cross-layer AuxTemp sharing: every layer in ``consumers`` also
    consumes the PRODUCER layer's AuxTemp object (fwd, recompute and bwd)
    — e.g. GLM-5.2 IndexShare, where shared layers reuse the nearest full
    layer's selection. ``grad_bytes`` > 0 threads the backward companion
    ``dAuxTemp_{s}_{r}_{producer}`` through the group's bwds in reverse layer
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
    layer_kinds: tuple[str, ...] | None = None,
    family: str,
    hw: ShapedHardware | None = None,
    fast_memory_capacity: int | None = None,
    recompute_levels: Mapping[str, int] | None = None,
    name: str | None = None,
    aux_shared=None,
    round_prologue: bool = True,
    dp_group: str | None = None,   # peer-group NAME optimizer tasks
                                   # name; present handle => allreduce(dW)
                                   # before the update (P4a data parallel)
    shard_params=None,             # {object_root -> "shard" dict} from
                                   # sharding.shard_block_params: optimizer
                                   # tasks for listed roots execute region
                                   # reduce -> owned-only update -> W
                                   # broadcast instead of the replicated
                                   # allreduce+update (ZeRO-1 style); O
                                   # sizes shrink at exact-size time via
                                   # object_size_factory's opt_update_regions
    tp_params=None,                # {object_root -> {field -> slice}}
                                   # (sharding.tp_view derived): fwd/
                                   # recompute/bwd tasks of listed layers
                                   # gain a "tp_slices" block param and a
                                   # {"tp": dp_group} comm role — blocks view
                                   # sharded weights/activations at shard
                                   # shape and run the partial+allreduce
                                   # MLP variants; per-rank object sizes
                                   # come from family_layouts(tp_view=...)
    bias_update_in_bwd: bool = False,
    retained_lbl: bool = False,
    freeze=None,  # FreezePlan | None (training/freeze_plan.py)
) -> Program:
    """Build the (bare) shaped program for the given recompute levels.

    ``kinds`` is REQUIRED — every family declares its layer kinds
    explicitly (uniform families pass one). ``aux_shared`` is a list of
    ``AuxShare`` for cross-layer AuxTemp consumption (each layer's own
    AuxTemp comes from its kind's ``aux_temp_bytes``). ``recompute_levels`` maps
    saved-context object id (``A_{s}_{r}_{i}``) to level 0 (save) or 1
    (recompute); missing ids default to 0. This function IS the
    ``build_variant`` for the recompute planner (wrap with
    ``functools.partial``). ``family`` stamps the program metadata and
    seeds the default name. ``round_prologue`` opens every round with a
    ``prologue_round_{s}_{r}`` task producing the object-backed
    ``current_round_{s}_{r}`` value (emitted only for families that
    consume the round — dense chains stay byte-stable).
    """
    hw = hw or ShapedHardware()
    levels = dict(recompute_levels or {})
    aux_producer_of: dict[int, int] = {}
    aux_group_of: dict[int, tuple[int, ...]] = {}
    aux_grad_of: dict[int, int] = {}
    for share in (aux_shared or ()):
        mem = (share.producer,) + tuple(share.consumers)
        aux_group_of[share.producer] = mem
        aux_grad_of[share.producer] = share.grad_bytes
        for m in share.consumers:
            aux_producer_of[m] = share.producer
    loose = LooseCosts(cfg, hw)
    t, d = cfg.tokens, cfg.d_model

    def bf16(n_elems: int) -> int:
        return BF16 * n_elems

    w_embed = bf16(cfg.embed_params)
    w_head = bf16(cfg.head_params)
    y_bytes = bf16(t * d)
    ids_bytes = dtype_nbytes((t,), "int32")

    if layer_kinds is None:
        default_kind = next(iter(kinds))
        layer_kinds = tuple(default_kind for _ in range(cfg.n_layers))
    layer_specs = [kinds[k] for k in layer_kinds]
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
        add_initial(f"W_{i}", layer_specs[i].w_bytes, "parameter", persist=True)
        add_initial(f"O_{i}", 2 * layer_specs[i].w_bytes, "optimizer_state", persist=True)
        if layer_specs[i].aux_bytes:
            add_initial(f"Aux_{i}", layer_specs[i].aux_bytes, "optimizer_state",
                        persist=True)
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
             comm: dict | None = None, subops: list | None = None) -> None:
        tasks.append(TaskSpec(
            id=tid,
            inputs=tuple(inputs),
            outputs=tuple(outputs),
            mutates=mutates,
            runtime_us=runtime_us,
            group=group,
            compute_block_key=block,
            block_params=params or {},
            comm_groups=comm or {},
            metadata={"cost_subops": loose.subops[block] if subops is None else subops},
        ))

    if cfg.optimizer_placement not in ("interleaved", "tail"):
        raise ValueError(
            f"optimizer_placement must be 'interleaved' or 'tail', "
            f"got {cfg.optimizer_placement!r}"
        )
    interleaved = cfg.optimizer_placement == "interleaved"
    if shard_params and dp_group is None:
        raise ValueError("shard_params requires dp_group — the shard "
                         "collectives ride the same group handle")
    if tp_params and dp_group is None:
        raise ValueError("tp_params requires dp_group — the tp "
                         "collectives ride the same group handle")

    for s in range(cfg.num_steps):
        # Optimizer emitters: one per parameter family; ids/inputs identical
        # in both placements. In interleaved mode each fires right after the
        # final mutation of its gradient (last round's backward task), so O
        # prefetches and W/O writebacks overlap the rest of the backward
        # instead of draining serially after all compute is done.
        opt_comm = {"dp": dp_group} if dp_group else None

        shards = shard_params or {}
        tps = tp_params or {}

        def opt_embed(s: int = s) -> None:
            g_embed = f"dW_embed_{s}"
            extra = ({"shard": shards["W_embed"]}
                     if "W_embed" in shards else {})
            if "W_embed" in tps:
                extra["tp_slices"] = tps["W_embed"]
            task(f"optimizer_embed_{s}", "optimizer_embed",
                 ["W_embed", g_embed, "O_embed"], [],
                 loose.optimizer_us["embed"], mutates=("W_embed", "O_embed"),
                 group="optimizer",
                 params=extra or None, comm=opt_comm)

        def opt_block(i: int, s: int = s) -> None:
            sp = layer_specs[i]
            g_blk = f"dW_{s}_{i}"
            extra = ({"shard": shards[f"W_{i}"]}
                     if f"W_{i}" in shards else {})
            if f"W_{i}" in tps:
                extra["tp_slices"] = tps[f"W_{i}"]
            task(f"optimizer_{s}_{i}", "optimizer_block",
                 [f"W_{i}", g_blk, f"O_{i}"], [],
                 sp.optimizer_us, mutates=(f"W_{i}", f"O_{i}"),
                 group="optimizer",
                 params={"layer": i, **extra}, comm=opt_comm,
                 subops=sp.optimizer_subops)

        def opt_head(s: int = s) -> None:
            if tied:
                return  # optimizer_embed covers the shared W_embed/O_embed
            g_head = f"dW_head_{s}"
            extra = ({"shard": shards["W_head"]}
                     if "W_head" in shards else {})
            if "W_head" in tps:
                extra["tp_slices"] = tps["W_head"]
            task(f"optimizer_head_{s}", "optimizer_head",
                 ["W_head", g_head, "O_head"], [],
                 loose.optimizer_us["head"], mutates=("W_head", "O_head"),
                 group="optimizer",
                 params=extra or None, comm=opt_comm)

        for r in range(cfg.grad_accum_rounds):
            first_round = r == 0
            last_round = r == cfg.grad_accum_rounds - 1
            # ---- round boundary (families that consume the round) ----
            if round_prologue:
                aux_ids = tuple(f"Aux_{i}" for i in range(cfg.n_layers)
                                if layer_specs[i].aux_bytes)
                task(f"prologue_round_{s}_{r}", "prologue_round",
                     list(aux_ids) if first_round else [],
                     [OutputSpec(id=f"current_round_{s}_{r}", size_bytes=4,
                                 role="activation",
                                 tensor=TensorMeta(dtype="int32", shape=(1,)))],
                     1.0, group="forward", params={"round": r}, subops=[],
                     mutates=(aux_ids if first_round else ()))
            # ---- forward ----
            # the current_round edge chains EVERY round task behind the
            # prologue (embed -> blocks -> head -> bwds): the planner
            # must schedule the prologue first, so its published
            # metadata (num_tokens_by_round, materialized Segments) is
            # visible to all of them. Without it the prologue floats
            # freely and dense-family tasks can run before it.
            embed_ins = [f"tokens_{s}_{r}", "W_embed"]
            if round_prologue:
                embed_ins.append(f"current_round_{s}_{r}")
            task(
                f"embed_fwd_{s}_{r}", "embed_fwd",
                embed_ins,
                [OutputSpec(id=f"y_embed_{s}_{r}", size_bytes=y_bytes, role="activation",
                            tensor=TensorMeta(dtype="bf16", shape=(t, d)))],
                loose.embed_fwd_us, group="forward",
            )
            for i in range(cfg.n_layers):
                sp = layer_specs[i]
                a_id = f"A_{s}_{r}_{i}"
                level = levels.get(a_id, 0)
                x_id = f"y_embed_{s}_{r}" if i == 0 else f"y_{s}_{r}_{i - 1}"
                fwd_ins = [x_id, f"W_{i}"]
                outs = [OutputSpec(id=f"y_{s}_{r}_{i}", size_bytes=y_bytes, role="activation",
                                   tensor=TensorMeta(dtype="bf16", shape=(t, d)))]
                if level == 0:
                    outs.append(OutputSpec(id=a_id, size_bytes=sp.a_bytes, role="activation"))
                if sp.aux_temp_bytes:
                    outs.append(OutputSpec(id=f"AuxTemp_{s}_{r}_{i}",
                                           size_bytes=sp.aux_temp_bytes,
                                           role="activation"))
                if i in aux_producer_of:
                    fwd_ins.append(f"AuxTemp_{s}_{r}_{aux_producer_of[i]}")
                fwd_mut: tuple[str, ...] = ()
                if sp.aux_bytes and round_prologue:
                    fwd_ins.append(f"current_round_{s}_{r}")
                    fwd_ins.append(f"Aux_{i}")
                    fwd_mut = (f"Aux_{i}",)   # counts accumulate; recompute never has this edge
                tp_extra = ({"tp_slices": tp_params[f"W_{i}"]}
                            if tp_params and f"W_{i}" in tp_params else {})
                tp_comm = {"tp": dp_group} if tp_extra else None
                key_prefix = (f"tp_{sp.key_prefix}" if tp_extra
                              else sp.key_prefix)
                task(f"block_fwd_{s}_{r}_{i}", f"{key_prefix}_fwd", fwd_ins, outs,
                     sp.fwd_us, group="forward",
                     params={"layer": i, **tp_extra}, comm=tp_comm,
                     mutates=fwd_mut, subops=sp.fwd_subops)
                rewrites.append(RecomputeRewrite(
                    object_id=a_id,
                    f_task_id=f"block_fwd_{s}_{r}_{i}",
                    r_task_id=f"block_recompute_{s}_{r}_{i}",
                    options=(
                        RecomputeOption(level=0, saved_bytes=sp.a_bytes, recompute_us=0.0, label="save"),
                        RecomputeOption(level=1, saved_bytes=0, recompute_us=sp.recompute_us, label="recompute"),
                    ),
                    f_compute_block_key=f"{key_prefix}_fwd",
                    r_compute_block_key=f"{key_prefix}_recompute",
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
                sp = layer_specs[i]
                aux_ins = []
                if sp.aux_temp_bytes:
                    aux_ins.append(f"AuxTemp_{s}_{r}_{i}")
                if i in aux_producer_of:
                    aux_ins.append(f"AuxTemp_{s}_{r}_{aux_producer_of[i]}")
                tp_extra = ({"tp_slices": tp_params[f"W_{i}"]}
                            if tp_params and f"W_{i}" in tp_params else {})
                tp_comm = {"tp": dp_group} if tp_extra else None
                key_prefix = (f"tp_{sp.key_prefix}" if tp_extra
                              else sp.key_prefix)
                if levels.get(a_id, 0) == 1:
                    task(f"block_recompute_{s}_{r}_{i}", f"{key_prefix}_recompute",
                         [x_id, f"W_{i}"] + aux_ins,
                         [OutputSpec(id=a_id, size_bytes=sp.a_bytes, role="activation")],
                         sp.recompute_us, group="recompute",
                         params={"layer": i, **tp_extra}, comm=tp_comm,
                         subops=sp.recompute_subops)
                bwd_inputs = [f"dy_{s}_{r}_{i}", a_id, x_id, f"W_{i}"] + aux_ins
                outs = [OutputSpec(id=(f"dy_embed_{s}_{r}" if i == 0 else f"dy_{s}_{r}_{i - 1}"),
                                   size_bytes=y_bytes, role="gradient",
                                   tensor=TensorMeta(dtype="bf16", shape=(t, d)))]
                mutates: tuple[str, ...] = ()
                if first_round:
                    outs.append(OutputSpec(id=f"dW_{s}_{i}", size_bytes=sp.w_bytes, role="gradient"))
                else:
                    bwd_inputs.append(f"dW_{s}_{i}")
                    mutates = (f"dW_{s}_{i}",)
                ld_grp = aux_producer_of.get(i, i)
                if ld_grp in aux_group_of and aux_grad_of.get(ld_grp, 0):
                    mem = aux_group_of[ld_grp]
                    gid = f"dAuxTemp_{s}_{r}_{ld_grp}"
                    if i == mem[-1]:
                        outs.append(OutputSpec(id=gid,
                                               size_bytes=aux_grad_of[ld_grp],
                                               role="gradient"))
                    elif i == ld_grp:
                        bwd_inputs.append(gid)
                    elif i in mem:
                        bwd_inputs.append(gid)
                        mutates = mutates + (gid,)
                if sp.aux_bytes and round_prologue and last_round:
                    # the STEP-aggregate counts (all rounds accumulated) —
                    # noaux families also nudge the router bias inside W
                    # here (appended: mutates[0] stays the dW convention)
                    bwd_inputs.append(f"Aux_{i}")
                    if bias_update_in_bwd:
                        mutates = mutates + (f"W_{i}",)
                    if retained_lbl and sp.aux_temp_bytes:
                        # retained-inputs LBL: the deferred exact-aggregate
                        # contraction reads EVERY round's retained pack (the
                        # own round's AuxTemp is already in aux_ins)
                        bwd_inputs.extend(f"AuxTemp_{s}_{rr}_{i}"
                                          for rr in range(r))
                task(f"block_bwd_{s}_{r}_{i}", f"{key_prefix}_bwd", bwd_inputs, outs,
                     sp.bwd_us, mutates=mutates, group="backward",
                     params={"layer": i, **tp_extra}, comm=tp_comm,
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

    if freeze is not None:
        # frozen-parameter / local-objective surgery lives in
        # freeze_program.py — this builder stays the common case
        from .freeze_program import to_frozen_form

        _freeze_transform = lambda prog: to_frozen_form(prog, freeze)  # noqa: E731
    else:
        _freeze_transform = None

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
    return _freeze_transform(program) if _freeze_transform else program
