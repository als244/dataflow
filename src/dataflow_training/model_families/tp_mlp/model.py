"""Tensor-parallel MLP demonstration family (Megatron-style TP over
one llama3 MLP: ``y = w2 @ (silu(w1 @ x) * (w3 @ x))``).

NOT a performance path — a correctness gate for MID-TASK collectives
on the group plane. Column-sharding w1/w3 and row-sharding w2 over
``d_ff`` makes the forward need ONE ``allreduce(y_partial)`` and the
backward ONE ``allreduce(dx_partial)``, while every dW and any
optimizer state stay fully local: the exact opposite comm shape of
data parallelism, exercised from ordinary compute blocks instead of
the optimizer epilogue.

Sharding decomposition (rank r holds columns ``S_r`` of ff):
    a1 = x @ w1[:, S_r]        a3 = x @ w3[:, S_r]      (local)
    h  = silu(a1) * a3                                   (local)
    y  = allreduce_sum(h @ w2[S_r, :])                   (1 collective)
    dh = dy @ w2[S_r, :].T                               (local)
    dw2 = h.T @ dy   dw1 = x.T @ da1   dw3 = x.T @ da3   (local)
    dx = allreduce_sum(da1 @ w1[:, S_r].T + da3 @ w3[:, S_r].T)

Every non-allreduced quantity is BITWISE equal to the corresponding
slice of the full-width computation (column slices of independent
dot products); the allreduced y/dx equal a split-order full-width
reference bitwise too (world-2 bf16 add commutes). The gates in
tests/fleet/test_tp_mlp.py assert exactly that — bitwise on the
reference's own GPU architecture; across architectures gemm
accumulation order may break rounding ties differently (measured on
the 3090/5090 pair: 1 element in 8192 at |d| 1.8e-12), so
cross-arch ranks get an ulp-dust budget instead.

The family registers itself on import (``register_family``); remote
daemons load it with ``dataflowd start --plugin
dataflow_training.model_families.tp_mlp``. Standalone runs (no group handle
yet, e.g. the kernel warm-up before ``create_peer_group``) keep the
PARTIAL sums in y/dx — numerically wrong by design, like a
standalone run of a dp_group artifact is rank-local; re-seed after
warm-up exactly as the fleet drivers do.
"""
from __future__ import annotations

from dataclasses import dataclass

from dataflow.core.program import ObjectSpec, OutputSpec, Program, TaskSpec

GROUP_ROLE = "tp"


@dataclass(frozen=True)
class TpMlpConfig:
    t: int = 64                 # tokens
    d_model: int = 64
    d_ff: int = 256
    world: int = 2
    rank: int = 0
    seed: int = 17

    @classmethod
    def tiny(cls) -> "TpMlpConfig":
        return cls()


@dataclass(frozen=True)
class TpMlpDims:
    t: int
    d: int
    ff: int
    ffs: int                    # this rank's shard width
    world: int
    rank: int
    # the generic service run path builds uniform Segments from these
    # (no attention consumes them here)
    seq_len: int
    tokens: int


def dims_of_tp_mlp(cfg: TpMlpConfig) -> TpMlpDims:
    if cfg.d_ff % cfg.world:
        raise ValueError(f"d_ff {cfg.d_ff} not divisible by world "
                         f"{cfg.world}")
    return TpMlpDims(t=cfg.t, d=cfg.d_model, ff=cfg.d_ff,
                     ffs=cfg.d_ff // cfg.world, world=cfg.world,
                     rank=cfg.rank, seq_len=cfg.t, tokens=cfg.t)


def tp_weight_layout(dims: TpMlpDims):
    from dataflow_training.blocks.layouts import PackedLayout

    return PackedLayout.build([
        ("w1", (dims.d, dims.ffs), "bf16"),
        ("w3", (dims.d, dims.ffs), "bf16"),
        ("w2", (dims.ffs, dims.d), "bf16"),
    ])


def tp_saved_layout(dims: TpMlpDims):
    from dataflow_training.blocks.layouts import PackedLayout

    return PackedLayout.build([
        ("a1", (dims.t, dims.ffs), "bf16"),
        ("a3", (dims.t, dims.ffs), "bf16"),
    ])


def lower_tp_mlp(cfg: TpMlpConfig) -> Program:
    dims = dims_of_tp_mlp(cfg)
    wl = tp_weight_layout(dims)
    al = tp_saved_layout(dims)
    td_bytes = dims.t * dims.d * 2
    return Program(
        name=f"tp-mlp-w{cfg.world}r{cfg.rank}",
        initial_objects=(
            ObjectSpec("W_tp", wl.total_bytes, role="parameter"),
            ObjectSpec("x_0", td_bytes, role="activation"),
            ObjectSpec("dy_0", td_bytes, role="activation"),
        ),
        tasks=(
            TaskSpec(id="tp_fwd_0_0_0",
                     inputs=("x_0", "W_tp"),
                     outputs=(
                         OutputSpec("y_0", td_bytes, role="activation"),
                         OutputSpec("A_0", al.total_bytes,
                                    role="activation"),
                     ),
                     runtime_us=50.0, group="forward",
                     compute_block_key="tp_fwd",
                     comm_groups={"tp": GROUP_ROLE}),
            TaskSpec(id="tp_bwd_0_0_0",
                     inputs=("dy_0", "x_0", "W_tp", "A_0"),
                     outputs=(
                         OutputSpec("dW_tp_0", wl.total_bytes,
                                    role="gradient"),
                         OutputSpec("dx_0", td_bytes, role="gradient"),
                     ),
                     runtime_us=100.0, group="backward",
                     compute_block_key="tp_bwd",
                     comm_groups={"tp": GROUP_ROLE}),
        ),
        final_locations={"y_0": "backing", "dW_tp_0": "backing",
                         "dx_0": "backing"},
        # planner transfer model (bytes/us); the shaped families' PCIe
        # figure — the toy is placement-trivial either way
        bandwidth_from_slow=55000,
        bandwidth_to_slow=55000,
    )


def full_width_draws(cfg: TpMlpConfig, seed: int) -> dict:
    """The SINGLE source of truth for the toy's values: full-width
    bf16 draws from one sequential stream. Ranks slice their shard;
    the test's reference consumes the same tensors whole."""
    import torch

    g = torch.Generator().manual_seed(int(seed) + cfg.seed)
    def draw(*shape):
        return (torch.randn(*shape, generator=g) * 0.05).bfloat16()
    return {
        "w1": draw(cfg.d_model, cfg.d_ff),
        "w3": draw(cfg.d_model, cfg.d_ff),
        "w2": draw(cfg.d_ff, cfg.d_model),
        "x": draw(cfg.t, cfg.d_model),
        "dy": draw(cfg.t, cfg.d_model),
    }


def initial_values_tp_mlp(program: Program, cfg: TpMlpConfig, backend,
                          *, seed: int = 0, into=None):
    import torch

    from dataflow.runtime.interop import torch_view

    dims = dims_of_tp_mlp(cfg)
    wl = tp_weight_layout(dims)
    full = full_width_draws(cfg, seed)
    lo = cfg.rank * dims.ffs
    hi = lo + dims.ffs
    buffers = into if into is not None else {}
    for spec in program.initial_objects:
        if spec.id not in buffers:
            if backend is None:
                raise ValueError(f"no buffer for {spec.id} and no "
                                 f"backend to allocate one")
            buffers[spec.id] = backend.alloc("backing", spec.size_bytes)
        buf = buffers[spec.id]
        if spec.id == "W_tp":
            wl.view(buf, "w1").copy_(full["w1"][:, lo:hi])
            wl.view(buf, "w3").copy_(full["w3"][:, lo:hi])
            wl.view(buf, "w2").copy_(full["w2"][lo:hi, :])
        elif spec.id == "x_0":
            torch_view(buf, (dims.t, dims.d), torch.bfloat16).copy_(
                full["x"])
        elif spec.id == "dy_0":
            torch_view(buf, (dims.t, dims.d), torch.bfloat16).copy_(
                full["dy"])
    return buffers


def silu_grads(a1, a3, dh):
    """(da1, da3) for h = silu(a1) * a3, in fp32 like the real
    kernels' register math, cast back to storage dtype."""
    import torch

    s = torch.sigmoid(a1.float())
    silu = a1.float() * s
    dsilu = s * (1.0 + a1.float() * (1.0 - s))
    da1 = (dh.float() * a3.float() * dsilu).to(a1.dtype)
    da3 = (dh.float() * silu).to(a3.dtype)
    return da1, da3


@dataclass(frozen=True)
class TpMlpFwd:
    dims: TpMlpDims

    def launch(self, ctx) -> None:
        import torch

        from dataflow.runtime.interop import external_stream, torch_view

        d = self.dims
        es = external_stream(ctx.stream)
        with torch.cuda.stream(es):
            x = torch_view(ctx.inputs[ctx.task.inputs[0]],
                           (d.t, d.d), torch.bfloat16)
            w_buf = ctx.inputs[ctx.task.inputs[1]]
            wl = tp_weight_layout(d)
            a1 = x @ wl.view(w_buf, "w1")
            a3 = x @ wl.view(w_buf, "w3")
            al = tp_saved_layout(d)
            a_buf = ctx.outputs[ctx.task.outputs[1].id]
            al.view(a_buf, "a1").copy_(a1)
            al.view(a_buf, "a3").copy_(a3)
            h = torch.nn.functional.silu(a1) * a3
            y = torch_view(ctx.outputs[ctx.task.outputs[0].id],
                           (d.t, d.d), torch.bfloat16)
            y.copy_(h @ wl.view(w_buf, "w2"))   # partial sum, in place
            gh = (getattr(ctx, "groups", None) or {}).get(
                ctx.task.comm_groups.get("tp"))
            if gh is None:
                return          # standalone/warm-up: partials stand
            produced = torch.cuda.Event()
            produced.record(es)
            gh.stream.wait_event(produced)
            gh.allreduce(y)                     # y := sum over shards
            summed = torch.cuda.Event()
            summed.record(gh.stream)
            es.wait_event(summed)


@dataclass(frozen=True)
class TpMlpBwd:
    dims: TpMlpDims

    def launch(self, ctx) -> None:
        import torch

        from dataflow.runtime.interop import external_stream, torch_view

        d = self.dims
        es = external_stream(ctx.stream)
        with torch.cuda.stream(es):
            dy = torch_view(ctx.inputs[ctx.task.inputs[0]],
                            (d.t, d.d), torch.bfloat16)
            x = torch_view(ctx.inputs[ctx.task.inputs[1]],
                           (d.t, d.d), torch.bfloat16)
            w_buf = ctx.inputs[ctx.task.inputs[2]]
            a_buf = ctx.inputs[ctx.task.inputs[3]]
            wl = tp_weight_layout(d)
            al = tp_saved_layout(d)
            w1 = wl.view(w_buf, "w1")
            w3 = wl.view(w_buf, "w3")
            w2 = wl.view(w_buf, "w2")
            a1 = al.view(a_buf, "a1")
            a3 = al.view(a_buf, "a3")
            h = torch.nn.functional.silu(a1) * a3
            dh = dy @ w2.T
            da1, da3 = silu_grads(a1, a3, dh)
            dw_buf = ctx.outputs[ctx.task.outputs[0].id]
            wl.view(dw_buf, "w2").copy_(h.T @ dy)
            wl.view(dw_buf, "w1").copy_(x.T @ da1)
            wl.view(dw_buf, "w3").copy_(x.T @ da3)
            dx = torch_view(ctx.outputs[ctx.task.outputs[1].id],
                            (d.t, d.d), torch.bfloat16)
            dx.copy_(da1 @ w1.T + da3 @ w3.T)   # partial, in place
            gh = (getattr(ctx, "groups", None) or {}).get(
                ctx.task.comm_groups.get("tp"))
            if gh is None:
                return
            produced = torch.cuda.Event()
            produced.record(es)
            gh.stream.wait_event(produced)
            gh.allreduce(dx)
            summed = torch.cuda.Event()
            summed.record(gh.stream)
            es.wait_event(summed)


def build_tp_mlp_resolver(dims: TpMlpDims, hyper=None, kernels=None):
    table = {"tp_fwd": TpMlpFwd(dims), "tp_bwd": TpMlpBwd(dims)}

    def resolver(task):
        key = task.compute_block_key
        if key not in table:
            raise KeyError(f"no executable for compute_block_key "
                           f"{key!r} (task {task.id!r})")
        return table[key]

    resolver.kernel_set = kernels
    return resolver



def tp_mlp_family():
    from dataflow_training.model_families.families import Family

    return Family(
        name="tp_mlp",
        config_type=TpMlpConfig,
        dims_of=dims_of_tp_mlp,
        lower=lower_tp_mlp,
        initial_values=initial_values_tp_mlp,
        build_resolver=build_tp_mlp_resolver,
    )


def register() -> None:
    from dataflow_training.model_families.families import _FAMILIES, register_family

    if "tp_mlp" not in _FAMILIES:
        register_family("tp_mlp", tp_mlp_family)


register()
