"""RL post-training on the dataflow engine — end to end.

    python examples/rl_trainer/run.py [--loss ppo|reinforce] [--steps 3]
        [--device-gib 2.0] [--artifacts rollout.pt]

Pipeline (docs/extending_programs.md):
1. fake_inference: rollout forward, saves checkpoints + M payloads +
   rollout tensors + starting weights.
2. program_builder: the custom Program (no forward pass; explicit
   recompute-from-checkpoint; RL head; builtin bwd/optimizer weave).
3. PressureFit at --device-gib, then Engine.execute STEPS times — one
   annotated program replayed per optimizer step (the boundary
   invariant: every persistent object ends where it started).
4. reference_trainer: the isolated autograd witness on the same
   artifacts.
5. Parity: per-field rel_l2 of every weight object after every step
   (sign-lottery envelope for w_router_bias), plus the loss values.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from dataflow.runtime import Engine
from dataflow.runtime.device.cuda import CudaBackend
from dataflow.runtime.device.fake import FakeBackend
from dataflow.tasks.interop import TORCH_DTYPE_BY_NAME, torch_view
from dataflow.tasks.kernels import resolve_kernels
from dataflow.training.families import resolve_family
from dataflow.training.glm52 import ShapedGlm52Config
from dataflow.training.planning import plan_program
from dataflow.training.testing.gradcheck import rel_l2

import fake_inference
import reference_trainer
from program_builder import build_rl_program
from rl_ops import RLHeadLoss

BIAS_ATOL = {"w_router_bias": 2.5e-3}


def build_values(prog, cfg, artifacts, backend, steps):
    """Initial buffers: weights/opt-state via the family initializer
    conventions, rollout + checkpoints + M payloads from the artifacts."""
    fam = resolve_family(cfg)
    values = fam.initial_values(prog, cfg, backend, seed=0)

    def load(name, tensor):
        buf = values[name]
        dst = torch_view(buf, (buf.size_bytes,), torch.uint8)
        src = tensor.contiguous().view(-1).view(torch.uint8)
        assert src.numel() == buf.size_bytes, (name, src.numel(), buf.size_bytes)
        dst.copy_(src)

    for name, tb in artifacts["w_bytes"].items():
        load(name, tb)
    for name in values:
        if name.startswith("O_"):
            torch_view(values[name], (values[name].size_bytes,), torch.uint8).zero_()
    load("actions_0_0", artifacts["actions"])
    load("old_logprobs_0_0", artifacts["old_logprobs"])
    load("advantages_0_0", artifacts["advantages"])
    ckpts = artifacts["x_ckpt"] + [artifacts["y_last"]]
    for s in range(steps):
        load(f"tokens_{s}_0", artifacts["tokens"])
        ids = [f"y_embed_{s}_0"] + [f"y_{s}_0_{i}" for i in range(cfg.n_layers)]
        for oid, tens in zip(ids, ckpts):
            load(oid, tens)
        for i, mb in artifacts["m_bytes"].items():
            load(f"M_{s}_0_{i}", mb)
    return values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", choices=["ppo", "reinforce"], default="ppo")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--device-gib", type=float, default=2.0)
    ap.add_argument("--artifacts", default=None)
    args = ap.parse_args()

    art_path = args.artifacts or str(Path(__file__).parent / "rollout.pt")
    if not Path(art_path).exists():
        fake_inference.run(art_path)
    artifacts = torch.load(art_path, weights_only=False)

    cfg = replace(ShapedGlm52Config.tiny(), **artifacts["cfg_overrides"])
    fam = resolve_family(cfg)
    dims = fam.dims_of(cfg)

    prog = build_rl_program(cfg, steps=args.steps)
    planned = plan_program(prog, fast_memory_capacity=int(args.device_gib * 2**30))
    print(f"planned: {len(planned.program.task_by_id())} tasks, "
          f"sim {planned.makespan_us / 1e3:.1f} ms/step, "
          f"peak {planned.peak_fast_bytes / 2**20:.0f} MiB "
          f"<= {args.device_gib:g} GiB")

    backend = CudaBackend()
    values = build_values(planned.program, cfg, artifacts, backend, args.steps)

    base_resolver = fam.build_resolver(dims)
    rl_head = RLHeadLoss(dims, resolve_kernels(), mode=args.loss)

    def resolver(task):
        if task.compute_block_key == "rl_head_loss":
            return rl_head
        return base_resolver(task)

    dry = Engine(FakeBackend()).execute(planned.program, initial_buffers=values)
    result = Engine(backend).execute(
        planned.program, resolver=resolver, initial_buffers=values,
        pool_prewarm=dry.pool_demand,
    )
    engine_losses = []
    for s in range(args.steps):
        loss_buf = result.objects.get(f"loss_{s}_0").backing.buffer
        engine_losses.append(float(torch_view(loss_buf, (1,), torch.float32)[0]))
    print(f"engine losses ({args.loss}): "
          + ", ".join(f"{v:.5f}" for v in engine_losses))

    ref_losses, snaps, golden = reference_trainer.train(
        artifacts, steps=args.steps, mode=args.loss)
    print(f"reference losses:        "
          + ", ".join(f"{v:.5f}" for v in ref_losses))

    # ---- parity: every weight object, every field, after the run ----
    worst = (0.0, "")
    failures = []
    for oid in ["W_embed"] + [f"W_{i}" for i in range(cfg.n_layers)] + ["W_head"]:
        layout, leaves = golden.final_leaves(oid)
        rec = result.objects.get(oid)
        buf = (rec.backing or rec.fast).buffer
        for f in layout.fields:
            got = torch_view(buf, f.shape, TORCH_DTYPE_BY_NAME[f.dtype],
                             offset_bytes=f.offset_bytes)
            atol = BIAS_ATOL.get(f.name)
            if atol is not None:
                gap = float((got.float().cpu()
                             - leaves[f.name].detach().float().cpu())
                            .abs().max())
                ok = gap <= atol
                score = gap / atol
            else:
                score = rel_l2(got, leaves[f.name])
                ok = score <= 3e-2
            if score > worst[0]:
                worst = (score, f"{oid}.{f.name}")
            if not ok:
                failures.append((f"{oid}.{f.name}", score))
    loss_gap = max(abs(a - b) / max(abs(b), 1e-6)
                   for a, b in zip(engine_losses, ref_losses))
    print(f"parity after {args.steps} steps: worst field {worst[1]} "
          f"({worst[0]:.2e}); loss gap {loss_gap:.2e}")
    if failures or loss_gap > 1e-2:
        print("FAIL:", failures[:5])
        sys.exit(1)
    print("PASS: engine == isolated autograd, both trainers, "
          f"{args.steps} optimizer steps ({args.loss}).")


if __name__ == "__main__":
    main()
