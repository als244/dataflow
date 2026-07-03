"""Llama3-*shaped* training program generator (M0 stand-in for real lowering).

Structure and object sizes are exact for a Llama3-style dense transformer
(bf16 params/activations/grads, AdamW state); task runtimes are analytic
roofline estimates. This is NOT the real lowering (that arrives with the
tasks/training layers in M3) — it exists so the planning/simulation/runtime
pipeline can be built and gated against realistic full-scale programs first,
and it doubles as the parity-harness input for M1/M2.

Naming follows the simulator's conventions so programs cross-check against
dataflow_sim's built-in builders:

    tokens_{s}_{r}, targets_{s}_{r}      int32 token/target ids (initial, backing)
    W_embed, W_{i}, W_head               parameters (initial, backing)
    O_embed, O_{i}, O_head               AdamW state, 2x params (initial, backing)
    y_embed_{s}_{r}, y_{s}_{r}_{i}       block outputs (activations)
    A_{s}_{r}_{i}                        saved backward context per block
    logits_{s}_{r}, dlogits_{s}_{r}      head output / loss gradient
    dy_*                                 activation gradients
    dW_embed_{s}, dW_{s}_{i}, dW_head_{s}  parameter gradients (accumulated over rounds)
    loss_{s}_{r}                         scalar loss (final: backing)

Task chain per step s: rounds of [embed_fwd, block_fwd x L, head_fwd,
loss_bwd, head_bwd, (block_recompute?, block_bwd) x L reversed, embed_bwd],
then optimizer tasks (mutating W and O in place).
"""
from __future__ import annotations

from dataclasses import dataclass, field
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
class ShapedLlamaConfig:
    n_layers: int = 32
    d_model: int = 4096
    n_heads: int = 32
    n_kv_heads: int = 8
    d_ff: int = 14336
    vocab_size: int = 128_256
    seq_len: int = 4096
    batch: int = 1
    grad_accum_rounds: int = 1
    num_steps: int = 1

    @property
    def tokens(self) -> int:
        return self.seq_len * self.batch

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    # -- parameter counts ----------------------------------------------------
    @property
    def block_params(self) -> int:
        d, kv, ff = self.d_model, self.kv_dim, self.d_ff
        attn = d * d + d * kv + d * kv + d * d  # Wq, Wk, Wv, Wo
        mlp = 3 * d * ff                        # W1 (gate), W3 (up), W2 (down)
        norms = 2 * d
        return attn + mlp + norms

    @property
    def embed_params(self) -> int:
        return self.vocab_size * self.d_model

    @property
    def head_params(self) -> int:
        return self.vocab_size * self.d_model

    # -- activation widths (elements per token) --------------------------------
    @property
    def saved_ctx_width(self) -> int:
        """Elements per token saved by a block forward for its backward.

        Roughly: block input (d), attention normed input (d), q (d),
        k/v (2*kv), attention output (d), softmax lse (heads), mlp normed
        input (d), gate/up projections (2*ff). Matches the scale of real
        save-all footprints for this architecture.
        """
        d, kv, ff, h = self.d_model, self.kv_dim, self.d_ff, self.n_heads
        return 5 * d + 2 * kv + h + 2 * ff

    @classmethod
    def tiny(cls) -> "ShapedLlamaConfig":
        return cls(
            n_layers=2,
            d_model=64,
            n_heads=4,
            n_kv_heads=2,
            d_ff=160,
            vocab_size=512,
            seq_len=64,
            batch=1,
        )

    @classmethod
    def llama3_8b(cls, *, seq_len: int = 4096, batch: int = 1, grad_accum_rounds: int = 1, num_steps: int = 1) -> "ShapedLlamaConfig":
        return cls(seq_len=seq_len, batch=batch, grad_accum_rounds=grad_accum_rounds, num_steps=num_steps)


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


@dataclass
class _Costs:
    """Per-task analytic flops/bytes and roofline runtimes for one config."""

    cfg: ShapedLlamaConfig
    hw: ShapedHardware
    block_fwd_us: float = field(init=False)
    block_bwd_us: float = field(init=False)
    block_recompute_us: float = field(init=False)
    embed_fwd_us: float = field(init=False)
    embed_bwd_us: float = field(init=False)
    head_fwd_us: float = field(init=False)
    head_bwd_us: float = field(init=False)
    loss_bwd_us: float = field(init=False)
    optimizer_us: dict = field(init=False)
    subops: dict = field(init=False)

    def __post_init__(self) -> None:
        cfg, hw = self.cfg, self.hw
        t, d, seq = cfg.tokens, cfg.d_model, cfg.seq_len

        matmul_params = cfg.block_params - 2 * d  # exclude norm vectors
        mm_flops = 2.0 * t * matmul_params
        mm_bytes = BF16 * (matmul_params + t * (2 * d + 2 * cfg.kv_dim + 2 * cfg.d_ff))
        # causal attention: ~0.5 * 4 * tokens * seq * d flops (fwd)
        attn_flops = 2.0 * t * seq * d
        attn_bytes = BF16 * t * (2 * d + 2 * cfg.kv_dim)

        mm_fwd = hw.matmul_us(mm_flops, mm_bytes)
        attn_fwd = hw.attn_us(attn_flops, attn_bytes)
        self.block_fwd_us = mm_fwd + attn_fwd
        self.block_recompute_us = self.block_fwd_us
        # backward: dgrad + wgrad matmuls (2x fwd matmul flops), attention bwd ~2.5x fwd
        mm_bwd = hw.matmul_us(2.0 * mm_flops, 2.0 * mm_bytes)
        attn_bwd = hw.attn_us(2.5 * attn_flops, 2.0 * attn_bytes)
        self.block_bwd_us = mm_bwd + attn_bwd

        embed_bytes = BF16 * 2.0 * t * d + 4.0 * t
        self.embed_fwd_us = hw.mem_us(embed_bytes)
        self.embed_bwd_us = hw.mem_us(2.0 * embed_bytes)

        head_flops = 2.0 * t * d * cfg.vocab_size
        head_bytes = BF16 * (d * cfg.vocab_size + t * (d + cfg.vocab_size))
        self.head_fwd_us = hw.matmul_us(head_flops, head_bytes)
        self.head_bwd_us = hw.matmul_us(2.0 * head_flops, 2.0 * head_bytes)

        self.loss_bwd_us = hw.mem_us(BF16 * 2.0 * t * cfg.vocab_size)

        def opt_us(params: int) -> float:
            # read W, dW, m, v; write W, m, v (all bf16 here)
            return hw.mem_us(BF16 * 7.0 * params)

        self.optimizer_us = {
            "embed": opt_us(cfg.embed_params),
            "block": opt_us(cfg.block_params),
            "head": opt_us(cfg.head_params),
        }

        self.subops = {
            "block_fwd": [
                {"kind": "roofline", "name": "block_matmuls", "flops": int(mm_flops), "memory_bytes": int(mm_bytes), "efficiency": "matmul"},
                {"kind": "roofline", "name": "attention", "flops": int(attn_flops), "memory_bytes": int(attn_bytes), "efficiency": "attention"},
            ],
            "block_bwd": [
                {"kind": "roofline", "name": "block_matmuls_bwd", "flops": int(2 * mm_flops), "memory_bytes": int(2 * mm_bytes), "efficiency": "matmul"},
                {"kind": "roofline", "name": "attention_bwd", "flops": int(2.5 * attn_flops), "memory_bytes": int(2 * attn_bytes), "efficiency": "attention"},
            ],
            "block_recompute": [
                {"kind": "roofline", "name": "block_matmuls", "flops": int(mm_flops), "memory_bytes": int(mm_bytes), "efficiency": "matmul"},
                {"kind": "roofline", "name": "attention", "flops": int(attn_flops), "memory_bytes": int(attn_bytes), "efficiency": "attention"},
            ],
            "embed_fwd": [{"kind": "roofline", "name": "gather", "flops": 0, "memory_bytes": int(embed_bytes), "efficiency": "memory"}],
            "embed_bwd": [{"kind": "roofline", "name": "scatter_accum", "flops": 0, "memory_bytes": int(2 * embed_bytes), "efficiency": "memory"}],
            "head_fwd": [{"kind": "roofline", "name": "lm_head", "flops": int(head_flops), "memory_bytes": int(head_bytes), "efficiency": "matmul"}],
            "head_bwd": [{"kind": "roofline", "name": "lm_head_bwd", "flops": int(2 * head_flops), "memory_bytes": int(2 * head_bytes), "efficiency": "matmul"}],
            "loss_bwd": [{"kind": "roofline", "name": "ce_loss_bwd", "flops": 0, "memory_bytes": int(BF16 * 2 * t * cfg.vocab_size), "efficiency": "memory"}],
            "optimizer_embed": [{"kind": "roofline", "name": "adamw", "flops": 0, "memory_bytes": int(BF16 * 7 * cfg.embed_params), "efficiency": "memory"}],
            "optimizer_block": [{"kind": "roofline", "name": "adamw", "flops": 0, "memory_bytes": int(BF16 * 7 * cfg.block_params), "efficiency": "memory"}],
            "optimizer_head": [{"kind": "roofline", "name": "adamw", "flops": 0, "memory_bytes": int(BF16 * 7 * cfg.head_params), "efficiency": "memory"}],
        }


def build_shaped_llama3(
    cfg: ShapedLlamaConfig,
    *,
    hw: ShapedHardware | None = None,
    fast_memory_capacity: int | None = None,
    recompute_levels: Mapping[str, int] | None = None,
    name: str | None = None,
) -> Program:
    """Build the (bare) shaped program for the given recompute levels.

    ``recompute_levels`` maps saved-context object id (``A_{s}_{r}_{i}``) to
    level 0 (save) or 1 (recompute). Missing ids default to 0. This function
    IS the ``build_variant`` for the recompute planner (wrap with
    ``functools.partial``).
    """
    hw = hw or ShapedHardware()
    levels = dict(recompute_levels or {})
    costs = _Costs(cfg, hw)
    t, d = cfg.tokens, cfg.d_model

    def bf16(n_elems: int) -> int:
        return BF16 * n_elems

    w_block = bf16(cfg.block_params)
    w_embed = bf16(cfg.embed_params)
    w_head = bf16(cfg.head_params)
    y_bytes = bf16(t * d)
    a_bytes = bf16(t * cfg.saved_ctx_width)
    logits_bytes = bf16(t * cfg.vocab_size)
    ids_bytes = dtype_nbytes((t,), "int32")

    initial: list[ObjectSpec] = []
    final_locations: dict[str, str] = {}

    def add_initial(oid: str, size: int, role: str, *, persist: bool = False, tensor: TensorMeta | None = None) -> None:
        initial.append(ObjectSpec(id=oid, size_bytes=size, location="backing", role=role, tensor=tensor))
        if persist:
            final_locations[oid] = "backing"

    add_initial("W_embed", w_embed, "parameter", persist=True,
                tensor=TensorMeta(dtype="bf16", shape=(cfg.vocab_size, d)))
    add_initial("O_embed", 2 * w_embed, "optimizer_state", persist=True)
    for i in range(cfg.n_layers):
        add_initial(f"W_{i}", w_block, "parameter", persist=True)
        add_initial(f"O_{i}", 2 * w_block, "optimizer_state", persist=True)
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
             *, mutates: tuple[str, ...] = (), group: str = "compute", params: dict | None = None) -> None:
        tasks.append(TaskSpec(
            id=tid,
            inputs=tuple(inputs),
            outputs=tuple(outputs),
            mutates=mutates,
            runtime_us=runtime_us,
            group=group,
            compute_block_key=block,
            block_params=params or {},
            metadata={"cost_subops": costs.subops[block]},
        ))

    for s in range(cfg.num_steps):
        for r in range(cfg.grad_accum_rounds):
            first_round = r == 0
            # ---- forward ----
            task(
                f"embed_fwd_{s}_{r}", "embed_fwd",
                [f"tokens_{s}_{r}", "W_embed"],
                [OutputSpec(id=f"y_embed_{s}_{r}", size_bytes=y_bytes, role="activation",
                            tensor=TensorMeta(dtype="bf16", shape=(t, d)))],
                costs.embed_fwd_us, group="forward",
            )
            for i in range(cfg.n_layers):
                a_id = f"A_{s}_{r}_{i}"
                level = levels.get(a_id, 0)
                x_id = f"y_embed_{s}_{r}" if i == 0 else f"y_{s}_{r}_{i - 1}"
                outs = [OutputSpec(id=f"y_{s}_{r}_{i}", size_bytes=y_bytes, role="activation",
                                   tensor=TensorMeta(dtype="bf16", shape=(t, d)))]
                if level == 0:
                    outs.append(OutputSpec(id=a_id, size_bytes=a_bytes, role="activation"))
                task(f"block_fwd_{s}_{r}_{i}", "block_fwd", [x_id, f"W_{i}"], outs,
                     costs.block_fwd_us, group="forward", params={"layer": i})
                rewrites.append(RecomputeRewrite(
                    object_id=a_id,
                    f_task_id=f"block_fwd_{s}_{r}_{i}",
                    r_task_id=f"block_recompute_{s}_{r}_{i}",
                    options=(
                        RecomputeOption(level=0, saved_bytes=a_bytes, recompute_us=0.0, label="save"),
                        RecomputeOption(level=1, saved_bytes=0, recompute_us=costs.block_recompute_us, label="recompute"),
                    ),
                    f_compute_block_key="block_fwd",
                    r_compute_block_key="block_recompute",
                    group_key=f"layer_{i}",
                ))
            last_y = f"y_{s}_{r}_{cfg.n_layers - 1}"
            task(f"head_fwd_{s}_{r}", "head_fwd", [last_y, "W_head"],
                 [OutputSpec(id=f"logits_{s}_{r}", size_bytes=logits_bytes, role="activation",
                             tensor=TensorMeta(dtype="bf16", shape=(t, cfg.vocab_size)))],
                 costs.head_fwd_us, group="forward")

            # ---- loss + backward ----
            task(f"loss_bwd_{s}_{r}", "loss_bwd", [f"logits_{s}_{r}", f"targets_{s}_{r}"],
                 [OutputSpec(id=f"dlogits_{s}_{r}", size_bytes=logits_bytes, role="gradient",
                             tensor=TensorMeta(dtype="bf16", shape=(t, cfg.vocab_size))),
                  OutputSpec(id=f"loss_{s}_{r}", size_bytes=4, role="output",
                             tensor=TensorMeta(dtype="fp32", shape=(1,)))],
                 costs.loss_bwd_us, group="backward")
            final_locations[f"loss_{s}_{r}"] = "backing"

            head_grad_inputs = [f"dlogits_{s}_{r}", last_y, "W_head"]
            head_outs = [OutputSpec(id=f"dy_{s}_{r}_{cfg.n_layers - 1}", size_bytes=y_bytes, role="gradient",
                                    tensor=TensorMeta(dtype="bf16", shape=(t, d)))]
            head_mutates: tuple[str, ...] = ()
            if first_round:
                head_outs.append(OutputSpec(id=f"dW_head_{s}", size_bytes=w_head, role="gradient"))
            else:
                head_grad_inputs.append(f"dW_head_{s}")
                head_mutates = (f"dW_head_{s}",)
            task(f"head_bwd_{s}_{r}", "head_bwd", head_grad_inputs, head_outs,
                 costs.head_bwd_us, mutates=head_mutates, group="backward")

            for i in reversed(range(cfg.n_layers)):
                a_id = f"A_{s}_{r}_{i}"
                x_id = f"y_embed_{s}_{r}" if i == 0 else f"y_{s}_{r}_{i - 1}"
                if levels.get(a_id, 0) == 1:
                    task(f"block_recompute_{s}_{r}_{i}", "block_recompute", [x_id, f"W_{i}"],
                         [OutputSpec(id=a_id, size_bytes=a_bytes, role="activation")],
                         costs.block_recompute_us, group="recompute", params={"layer": i})
                bwd_inputs = [f"dy_{s}_{r}_{i}", a_id, x_id, f"W_{i}"]
                outs = [OutputSpec(id=(f"dy_embed_{s}_{r}" if i == 0 else f"dy_{s}_{r}_{i - 1}"),
                                   size_bytes=y_bytes, role="gradient",
                                   tensor=TensorMeta(dtype="bf16", shape=(t, d)))]
                mutates: tuple[str, ...] = ()
                if first_round:
                    outs.append(OutputSpec(id=f"dW_{s}_{i}", size_bytes=w_block, role="gradient"))
                else:
                    bwd_inputs.append(f"dW_{s}_{i}")
                    mutates = (f"dW_{s}_{i}",)
                task(f"block_bwd_{s}_{r}_{i}", "block_bwd", bwd_inputs, outs,
                     costs.block_bwd_us, mutates=mutates, group="backward", params={"layer": i})

            embed_bwd_inputs = [f"dy_embed_{s}_{r}", f"tokens_{s}_{r}"]
            embed_outs: list[OutputSpec] = []
            embed_mutates: tuple[str, ...] = ()
            if first_round:
                embed_outs.append(OutputSpec(id=f"dW_embed_{s}", size_bytes=w_embed, role="gradient"))
            else:
                embed_bwd_inputs.append(f"dW_embed_{s}")
                embed_mutates = (f"dW_embed_{s}",)
            task(f"embed_bwd_{s}_{r}", "embed_bwd", embed_bwd_inputs, embed_outs,
                 costs.embed_bwd_us, mutates=embed_mutates, group="backward")

        # ---- optimizer (after all rounds of step s) ----
        task(f"optimizer_embed_{s}", "optimizer_embed",
             ["W_embed", f"dW_embed_{s}", "O_embed"], [],
             costs.optimizer_us["embed"], mutates=("W_embed", "O_embed"), group="optimizer")
        for i in range(cfg.n_layers):
            task(f"optimizer_{s}_{i}", "optimizer_block",
                 [f"W_{i}", f"dW_{s}_{i}", f"O_{i}"], [],
                 costs.optimizer_us["block"], mutates=(f"W_{i}", f"O_{i}"), group="optimizer",
                 params={"layer": i})
        task(f"optimizer_head_{s}", "optimizer_head",
             ["W_head", f"dW_head_{s}", "O_head"], [],
             costs.optimizer_us["head"], mutates=("W_head", "O_head"), group="optimizer")

    label = name or (
        f"llama3-shaped-{cfg.n_layers}L-d{cfg.d_model}-s{cfg.seq_len}-b{cfg.batch}"
        f"-r{cfg.grad_accum_rounds}-steps{cfg.num_steps}"
    )
    return Program(
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
            "family": "llama3-shaped",
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
