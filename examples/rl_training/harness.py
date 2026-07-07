"""Shared harness for the per-family RL training examples.

Each family subdir provides an ADAPTER (a small module-level object)
and a 3-line run.py; everything else — artifact capture, the custom
Program build, PressureFit, engine execution, the isolated autograd
reference (frozen-rollout semantics: per-layer VJPs from fixed
checkpointed inputs), and the field-level parity gate — lives here.

Adapter contract (see glm52/adapter.py for the richest example):
    name: str                       family name (families registry)
    make_cfg() -> Shaped*Config     tiny config, discrete state frozen
    make_golden(dims, n_layers, leaves) -> golden (capture/pin capable)
    capture(golden, tokens) -> (captured: dict, y_last)
        runs the rollout forward, recording per-layer block inputs in
        captured["x"] and any discrete state (selections, routing)
    meta_fields(dims, i, captured) -> dict[field_name, tensor] | None
        the layer's M payload content (None = family has no M here)
    pin(golden, captured)           put golden into pinned mode
    prep_layer(golden, i)           per-layer state before a pinned call
    block(golden, i, x) -> (y, aux) pinned layer forward (aux 0 if none)
    bias_speed: float               router noaux bias speed (0 = none)
    adamw(golden, counts_of)        the family golden's optimizer replica

Intermediates saved per family dir (Shein: users should SEE the
program): program.json (bare custom Program), plan.json (PressureFit-
annotated), rollout.pt (artifacts).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from dataflow.core import save_program
from dataflow.runtime import Engine
from dataflow.runtime.device.cuda import CudaBackend
from dataflow.runtime.device.fake import FakeBackend
from dataflow.tasks.interop import TORCH_DTYPE_BY_NAME, torch_view
from dataflow.tasks.kernels import resolve_kernels
from dataflow.tasks import ops
from dataflow.training.families import family as family_of
from dataflow.training.planning import plan_program
from dataflow.training.testing.gradcheck import rel_l2

from builder import build_rl_program
from rl_ops import RLHeadLoss, rl_loss_reference

BIAS_ATOL = {"w_router_bias": 2.5e-3,  # noaux bias: sign(count) race
             "dt_bias": 2.5e-4}       # sub-noise grads: AdamW step-1 = ±lr coin flip


def _meta_bytes(layout, fields: dict) -> torch.Tensor:
    buf = torch.zeros(layout.total_bytes, dtype=torch.uint8)
    for f in layout.fields:
        if f.name not in fields:
            continue
        dt = TORCH_DTYPE_BY_NAME[f.dtype]
        n = 1
        for s in f.shape:
            n *= s
        view = buf[f.offset_bytes:f.offset_bytes + n * dt.itemsize].view(dt)
        view.view(*f.shape).copy_(fields[f.name])
    return buf


def routing_fields(dims, ids, weights) -> dict:
    """route pack in the runtime's moe_sort convention."""
    flat = ids.reshape(-1).long()
    counts = torch.bincount(flat, minlength=dims.moe.n_experts)
    offs = torch.zeros(dims.moe.n_experts + 1, dtype=torch.int64)
    offs[1:] = torch.cumsum(counts, 0)
    return {
        "route_w": weights.to(torch.bfloat16),
        "route_ids": ids.to(torch.int32),
        "route_order": torch.argsort(flat, stable=True).to(torch.int32),
        "route_offsets": offs.to(torch.int32),
    }


def make_artifacts(adapter, out_path: Path, *, seed=11, reward_seed=100):
    cfg = adapter.make_cfg()
    fam = family_of(adapter.name)
    dims = fam.dims_of(cfg)
    backend = CudaBackend()
    prog = plan_program(fam.lower(cfg), fast_memory_capacity=1 << 30).program
    values = fam.initial_values(prog, cfg, backend, seed=seed)

    def pinned(name):
        buf = values[name]
        return torch_view(buf, (buf.size_bytes,), torch.uint8).clone().cpu()

    w_names = [o.id for o in prog.initial_objects if o.id.startswith("W_")]
    w_bytes = {n: pinned(n) for n in w_names}
    leaves = [w_bytes["W_embed"].cuda(),
              [w_bytes[f"W_{i}"].cuda() for i in range(cfg.n_layers)]]
    if "W_head" in w_bytes:
        leaves.append(w_bytes["W_head"].cuda())
    golden = adapter.make_golden(dims, cfg.n_layers, leaves)

    tokens = torch_view(values["tokens_0_0"], (dims.tokens,), torch.int32)
    tokens = tokens.long().cuda()
    with torch.no_grad():
        captured, y_last = adapter.capture(golden, tokens)
        logits = (ops.rmsnorm_reference(
            y_last, golden.w_head["final_norm_w"]) @ golden.w_head["w"].T
        ).float()
        lse = torch.logsumexp(logits, dim=-1)

    g = torch.Generator(device="cuda").manual_seed(reward_seed)
    actions = torch.randint(0, dims.vocab_size, (dims.tokens,),
                            generator=g, device="cuda", dtype=torch.int64)
    old_lp = (logits.gather(1, actions.unsqueeze(1)).squeeze(1) - lse
              + 0.1 * torch.randn(dims.tokens, generator=g, device="cuda"))
    adv = torch.randn(dims.tokens, generator=g, device="cuda")

    m_bytes = {}
    from dataflow.tasks.layouts import PackedLayout  # noqa: F401
    for i in range(cfg.n_layers):
        fields = adapter.meta_fields(dims, i, captured)
        if fields:
            m_bytes[i] = _meta_bytes(adapter.meta_layout(dims, i), fields)

    art = {
        "tokens": tokens.to(torch.int32).cpu(),
        "actions": actions.to(torch.int32).cpu(),
        "old_logprobs": old_lp.float().cpu(),
        "advantages": adv.float().cpu(),
        "x_ckpt": [t.to(torch.bfloat16).cpu() for t in captured["x"]],
        "y_last": y_last.to(torch.bfloat16).cpu(),
        "m_bytes": m_bytes,
        "w_bytes": w_bytes,
        "captured": {k: v for k, v in captured.items() if k != "x"},
    }
    torch.save(art, out_path)
    return art


def build_values(prog, fam, cfg, artifacts, backend, steps):
    values = fam.initial_values(prog, cfg, backend, seed=0)

    def load(name, tensor):
        buf = values[name]
        dst = torch_view(buf, (buf.size_bytes,), torch.uint8)
        src = tensor.contiguous().view(-1).view(torch.uint8)
        assert src.numel() == buf.size_bytes, (name, src.numel(),
                                               buf.size_bytes)
        dst.copy_(src)

    for name, tb in artifacts["w_bytes"].items():
        load(name, tb)
    for name in values:
        if name.startswith("O_"):
            torch_view(values[name], (values[name].size_bytes,),
                       torch.uint8).zero_()
    for oid, key in (("actions_0_0", "actions"),
                     ("old_logprobs_0_0", "old_logprobs"),
                     ("advantages_0_0", "advantages")):
        load(oid, artifacts[key])
    ckpts = artifacts["x_ckpt"] + [artifacts["y_last"]]
    L = cfg.n_layers
    for s in range(steps):
        load(f"tokens_{s}_0", artifacts["tokens"])
        ids = [f"y_embed_{s}_0"] + [f"y_{s}_0_{i}" for i in range(L)]
        for oid, tens in zip(ids, ckpts):
            load(oid, tens)
        for i, mb in artifacts["m_bytes"].items():
            load(f"M_{s}_0_{i}", mb)
    return values


def reference_train(adapter, artifacts, *, steps, mode):
    cfg = adapter.make_cfg()
    fam = family_of(adapter.name)
    dims = fam.dims_of(cfg)
    wb = artifacts["w_bytes"]
    leaves = [wb["W_embed"].cuda(),
              [wb[f"W_{i}"].cuda() for i in range(cfg.n_layers)]]
    if "W_head" in wb:
        leaves.append(wb["W_head"].cuda())
    golden = adapter.make_golden(dims, cfg.n_layers, leaves)
    adapter.pin(golden, artifacts["captured"])

    tokens = artifacts["tokens"].long().cuda()
    actions = artifacts["actions"].long().cuda()
    old_lp = artifacts["old_logprobs"].cuda()
    adv = artifacts["advantages"].cuda()
    x_ckpt = [c.cuda() for c in artifacts["x_ckpt"]]
    y_last = artifacts["y_last"].cuda()

    losses = []
    for _ in range(steps):
        for p in golden.parameters():
            p.grad = None
        # frozen-rollout semantics: fixed inputs, current weights,
        # explicit per-layer VJP chaining (see README "read twice")
        y_hold = y_last.clone().requires_grad_(True)
        logits = (ops.rmsnorm_reference(
            y_hold, golden.w_head["final_norm_w"]) @ golden.w_head["w"].T)
        rl = rl_loss_reference(logits, actions, old_lp, adv,
                               dims.tokens, mode)
        rl.backward()
        dy = y_hold.grad

        counts_of = {}
        for i in range(cfg.n_layers - 1, -1, -1):
            adapter.prep_layer(golden, i)
            x_i = x_ckpt[i].clone().requires_grad_(True)
            y_i, aux_i, counts = adapter.block(golden, i, x_i)
            if counts is not None:
                counts_of[i] = counts
            total = (y_i.float() * dy.float()).sum()
            if aux_i is not None:
                total = total + aux_i
            total.backward()
            dy = x_i.grad
        g32 = torch.zeros(golden.w_embed["w"].shape, dtype=torch.float32,
                          device=dy.device)
        g32.index_add_(0, tokens, dy.float())
        golden.w_embed["w"].grad = g32.to(golden.w_embed["w"].dtype)

        adapter.adamw(golden, counts_of)
        losses.append(float(rl.detach()))
    return losses, golden


def run(adapter, *, loss="ppo", steps=3, device_gib=2.0, out_dir=None):
    out = Path(out_dir or Path(sys.argv[0]).resolve().parent)
    out.mkdir(parents=True, exist_ok=True)
    art_path = out / "rollout.pt"
    if not art_path.exists():
        make_artifacts(adapter, art_path)
    artifacts = torch.load(art_path, weights_only=False)

    cfg = adapter.make_cfg()
    fam = family_of(adapter.name)
    dims = fam.dims_of(cfg)

    prog = build_rl_program(fam, cfg, steps=steps)
    save_program(prog, out / "program.json")
    planned = plan_program(prog,
                           fast_memory_capacity=int(device_gib * 2**30))
    save_program(planned.program, out / "plan.json")
    print(f"[{adapter.name}] program.json ({len(prog.task_by_id())} tasks) "
          f"+ plan.json (sim {planned.makespan_us / 1e3:.1f} ms, peak "
          f"{planned.peak_fast_bytes / 2**20:.0f} MiB) -> {out}")

    backend = CudaBackend()
    values = build_values(planned.program, fam, cfg, artifacts, backend,
                          steps)
    base_resolver = fam.build_resolver(dims)
    rl_head = RLHeadLoss(dims, resolve_kernels(), mode=loss)

    def resolver(task):
        if task.compute_block_key == "rl_head_loss":
            return rl_head
        return base_resolver(task)

    dry = Engine(FakeBackend()).execute(planned.program,
                                        initial_buffers=values)
    result = Engine(backend).execute(
        planned.program, resolver=resolver, initial_buffers=values,
        pool_prewarm=dry.pool_demand)
    engine_losses = [
        float(torch_view(result.objects.get(f"loss_{s}_0").backing.buffer,
                         (1,), torch.float32)[0])
        for s in range(steps)]

    ref_losses, golden = reference_train(adapter, artifacts,
                                         steps=steps, mode=loss)
    print(f"[{adapter.name}] engine  ({loss}): "
          + ", ".join(f"{v:.5f}" for v in engine_losses))
    print(f"[{adapter.name}] autograd ref  : "
          + ", ".join(f"{v:.5f}" for v in ref_losses))

    worst, failures = (0.0, ""), []
    w_ids = [o.id for o in planned.program.initial_objects
             if o.id.startswith("W_")]
    for oid in w_ids:
        layout, leaves = golden.final_leaves(oid)
        rec = result.objects.get(oid)
        buf = (rec.backing or rec.fast).buffer
        for f in layout.fields:
            got = torch_view(buf, f.shape, TORCH_DTYPE_BY_NAME[f.dtype],
                             offset_bytes=f.offset_bytes)
            ref = leaves[f.name].detach()
            atol = BIAS_ATOL.get(f.name)
            if atol is not None:
                gap = float((got.float().cpu() - ref.float().cpu())
                            .abs().max())
                ok, score = gap <= atol, gap / atol
            else:
                score = rel_l2(got, ref)
                ok = score <= 3e-2
            if score > worst[0]:
                worst = (score, f"{oid}.{f.name}")
            if not ok:
                failures.append((f"{oid}.{f.name}", round(score, 5)))
    loss_gap = max(abs(a - b) / max(abs(b), 1e-6)
                   for a, b in zip(engine_losses, ref_losses))
    print(f"[{adapter.name}] parity: worst {worst[1]} ({worst[0]:.2e}); "
          f"loss gap {loss_gap:.2e}")
    if failures or loss_gap > 1e-2:
        print(f"[{adapter.name}] FAIL:", failures[:5])
        sys.exit(1)
    print(f"[{adapter.name}] PASS: engine == isolated autograd, "
          f"{steps} steps ({loss}).")


def main(adapter):
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", choices=["ppo", "reinforce"], default="ppo")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--device-gib", type=float, default=2.0)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    run(adapter, loss=a.loss, steps=a.steps, device_gib=a.device_gib,
        out_dir=a.out_dir)
