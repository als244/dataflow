#!/usr/bin/env python
"""Deep correctness-compare treatment for ONE family x shape.

Family-agnostic diagnosis ladder (see docs/correctness_compare.md for
the methodology and the gotcha catalog):

  1. forward divergence by depth: engine y_{s}_{r}_{i} block outputs vs
     twin block outputs (composition recorders), with per-token hot-row
     decomposition — separates smooth precision drift from discrete
     relocations (routing flips) and localizes jumps to a block.
  2. gradient comparison in dW space (engine slabs vs twin autograd),
     grouped per block — a bug shows as a jump at one block/op; noise
     compounds smoothly with depth.
  3. MoE counts parity where the twin exposes step counters (engine
     Aux_ objects vs twin step_counts): totals must equal tokens*top_k
     exactly; per-expert |delta| bounds the number of flipped tokens.

Usage: deep_compare.py --family glm52 --shape uniform
"""
from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import replace
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


class BlockRecorder(nn.Module):
    """Composition wrapper capturing each block's output rows."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.outs = []

    def forward(self, *a, **k):
        y = self.inner(*a, **k)
        keep = y[0] if isinstance(y, tuple) else y
        self.outs.append(keep.detach().float().cpu().reshape(
            -1, keep.shape[-1]))
        return y

    def __getattr__(self, name):
        # transparent proxy: model-level walkers (load_balance_loss,
        # module scans) must still see the wrapped block's attributes
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("inner"), name)


def kinds_of(dims, cfg):
    return getattr(dims, "kinds", None) or ("?",) * cfg.n_layers


def ragged_for(cfg):
    t = cfg.seq_len * cfg.batch
    a = t // 2 + 3
    b = t // 4 + 1
    return (a, b, t - a - b)


def main() -> int:
    from dataflow.pretrain import bridges
    from dataflow.runtime import Engine
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.runtime.device.fake import FakeBackend
    from dataflow.runtime.engine import uniform_segments
    from dataflow.tasks.interop import torch_view
    from dataflow.training.families import resolve_family
    from dataflow.training.planning import plan_program
    from dataflow.training.testing.gradcheck import (
        cos_sim,
        engine_grad_state_dict,
        rel_l2,
    )
    from dataflow.tasks.base_blocks import AdamWHyper
    import dataclasses

    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--shape", default="uniform",
                    choices=("uniform", "ragged"))
    ap.add_argument("--hot-mult", type=float, default=10.0)
    ap.add_argument("--isolate", type=str, default=None,
                    help="block index: feed the ENGINE's block N-1 output "
                         "into the twin's block N (per sequence) and compare "
                         "outputs — removes upstream divergence, so per-op "
                         "math is verified even when the full-model compare "
                         "is saturated by flip cascades")
    args = ap.parse_args()

    mod = importlib.import_module(
        f"dataflow.training.models.{args.family}")
    cfg_cls = next(v for k, v in vars(mod).items()
                   if k.startswith("Shaped") and k.endswith("Config"))
    cfg = cfg_cls.tiny()
    if args.shape == "ragged":
        cfg = replace(cfg, seq_lens=ragged_for(cfg))

    fam = resolve_family(cfg)
    dims = fam.dims_of(cfg)
    program = fam.lower(cfg)
    keep = {o.id for t in program.tasks for o in t.outputs
            if o.id.startswith(("dW", "y_", "dy_"))}
    keep.update(o.id for o in program.initial_objects
                if o.id.startswith("dW"))
    program = dataclasses.replace(
        program,
        final_locations={**dict(program.final_locations),
                         **{i: "fast" for i in keep}})
    planned = plan_program(program, fast_memory_capacity=96 * 1024 * 1024)
    backend = CudaBackend()
    values = fam.initial_values(planned.program, cfg, backend, seed=0)

    # ---- twin leg: init FIRST (clean names), then wrap blocks ----
    twin = bridges.build_reference_model(cfg)
    bridges.load_reference_init(twin, cfg, dims,
                                bridges.get_bytes_from_values(values))
    param_names = {n: p for n, p in twin.named_parameters()}
    twin.train()
    recs = []
    for bi, blk in enumerate(twin.blocks):
        rec = BlockRecorder(blk)
        twin.blocks[bi] = rec
        recs.append(rec)

    tokens = torch_view(values["tokens_0_0"], (dims.tokens,),
                        torch.int32).long().cuda()
    targets = (torch_view(values["targets_0_0"], (dims.tokens,),
                          torch.int32).long().cuda()
               if "targets_0_0" in values else tokens.clone())
    lens = tuple(getattr(dims, "seq_lens", None)
                 or (dims.seq_len,) * (dims.tokens // dims.seq_len))
    aux_coef = float(getattr(cfg, "aux_coef", 0.0) or 0.0)
    aux_form = getattr(twin, "AUX_FORM", None)
    drive_aux_seq = aux_coef > 0.0 and aux_form == "sequence_wise"
    drive_aux_round = aux_coef > 0.0 and aux_form == "forward_global"
    drive_idx = (bool(getattr(cfg, "train_indexer", False))
                 and hasattr(twin, "indexer_loss"))
    if drive_idx:
        twin.enable_indexer_kl(True)
    # native varlen: one packed forward per round (per-segment positions,
    # block-diagonal attention, recurrent-state resets inside the twin)
    ce = twin.loss(tokens.view(1, -1), targets.view(1, -1), seq_lens=lens)
    total = ce
    if aux_coef > 0.0 and (drive_aux_seq or drive_aux_round):
        total = total + aux_coef * twin.load_balance_loss()
    if drive_idx:
        total = total + twin.indexer_loss()
    total.backward()
    twin_grads = {n: p.grad.detach().float().cpu()
                  for n, p in param_names.items() if p.grad is not None}

    # ---- engine leg ----
    run_args = {"segments": uniform_segments(dims, planned.program)}
    dry = Engine(FakeBackend()).execute(planned.program,
                                        initial_buffers=values)
    result = Engine(backend).execute(
        planned.program, resolver=fam.build_resolver(dims),
        initial_buffers=values, pool_prewarm=dry.pool_demand,
        run_args=run_args)

    if args.isolate is not None:
        targets_iso = [int(x) for x in args.isolate.split(",")]
        bi = targets_iso[-1]          # compare at the LAST swapped block
        offsets = [0]
        for ln in lens:
            offsets.append(offsets[-1] + ln)

        class InputSwap(nn.Module):
            """Feed the ENGINE's previous-block output into this block,
            per sequence, regardless of what the twin computed upstream."""

            def __init__(self, mod, eng_in, counter):
                super().__init__()
                self.mod = mod
                self.eng_in = eng_in
                self.counter = counter

            def forward(self, x, *rest, **k):
                s = self.counter["i"]
                self.counter["i"] += 1
                if self.eng_in is not None:
                    x = self.eng_in[
                        offsets[s]:offsets[s + 1]].view(
                        1, -1, dims.d_model).to(x.dtype)
                return self.mod(x, *rest, **k)

        for b in targets_iso:
            eng_in = None
            if b > 0:
                r = result.objects.get(f"y_0_0_{b - 1}")
                slot = r.fast or r.backing
                eng_in = torch_view(slot.buffer,
                                    (dims.tokens, dims.d_model),
                                    torch.bfloat16).clone()
            recs[b].outs.clear()
            recs[b].inner = InputSwap(recs[b].inner, eng_in, {"i": 0})
        rec_iso = recs[bi]
        with torch.no_grad():
            lo = 0
            for ln in lens:
                twin.loss(tokens[lo:lo + ln].view(1, ln),
                          targets[lo:lo + ln].view(1, ln))
                lo += ln
        r = result.objects.get(f"y_0_0_{bi}")
        slot = r.fast or r.backing
        eng = torch_view(slot.buffer, (dims.tokens, dims.d_model),
                         torch.bfloat16).float().cpu()
        twn = torch.cat(rec_iso.outs)
        rowrel = (eng - twn).norm(dim=1) / twn.norm(dim=1).clamp_min(1e-12)
        med = float(rowrel.median())
        hot = (rowrel > max(args.hot_mult * med, 0.05)
               ).nonzero().flatten().tolist()
        mask = torch.ones(dims.tokens, dtype=torch.bool)
        mask[hot] = False
        print(f"== ISOLATED block(s) {targets_iso} "
              f"[{kinds_of(dims, cfg)[bi]} compared] on engine inputs:")
        print(f"  rel={rel_l2(eng, twn):.3e} cos={cos_sim(eng, twn):.6f}")
        print(f"  row-med {med:.3e}  p90 {float(rowrel.quantile(0.9)):.3e}"
              f"  max {float(rowrel.max()):.3e}")
        print(f"  hot rows {hot}")
        print(f"  rel excl hot {rel_l2(eng[mask], twn[mask]):.3e}")
        result.close()
        dry.close()
        return 0

    # ---- 1. forward divergence by depth + hot rows ----
    kinds = kinds_of(dims, cfg)
    print(f"== {args.family}/{args.shape}: forward divergence by depth")
    hot_union: set[int] = set()
    for bi, rec in enumerate(recs):
        oid = f"y_0_0_{bi}"
        r = result.objects.get(oid)
        slot = r.fast or r.backing
        eng = torch_view(slot.buffer, (dims.tokens, dims.d_model),
                         torch.bfloat16).float().cpu()
        twn = torch.cat(rec.outs)
        rowrel = (eng - twn).norm(dim=1) / twn.norm(dim=1).clamp_min(1e-12)
        med = float(rowrel.median())
        hot = (rowrel > max(args.hot_mult * med, 0.05)
               ).nonzero().flatten().tolist()
        hot_union.update(hot)
        mask = torch.ones(eng.shape[0], dtype=torch.bool)
        mask[list(hot_union)] = False    # exclude every token hot SO FAR
        print(f"  block {bi} [{kinds[bi]}]: rel={rel_l2(eng, twn):.3e} "
              f"cos={cos_sim(eng, twn):.6f} | row-med {med:.2e} "
              f"new-hot {hot} | rel excl all-hot "
              f"{rel_l2(eng[mask], twn[mask]):.3e}")

    # ---- 2. per-block gradient medians (dW space) ----
    import statistics

    engine_grads = engine_grad_state_dict(
        cfg, fam, dims, planned.program,
        fam.build_resolver(dims, AdamWHyper()), values, result)
    per_block: dict[str, list] = {}
    rows = []
    for name, g_e in engine_grads.items():
        g_t = twin_grads.get(name)
        if g_t is None or g_t.shape != g_e.shape:
            continue
        err = rel_l2(g_e, g_t)
        rows.append((err, name, cos_sim(g_e, g_t)))
        blk = (name.split(".")[1] if name.startswith("blocks.")
               else "loose")
        per_block.setdefault(blk, []).append(err)
    print("\n== per-block gradient medians (dW space):")
    for blk in sorted(per_block, key=lambda b: (b == "loose", b)):
        v = per_block[blk]
        kind = kinds[int(blk)] if blk != "loose" else "-"
        print(f"  block {blk:>5s} [{kind}]: median {statistics.median(v):.3e}"
              f"  worst {max(v):.3e}  (n={len(v)})")
    rows.sort(reverse=True)
    print("  worst fields:")
    for err, name, cos in rows[:8]:
        print(f"    {err:10.3e}  cos={cos:.6f}  {name}")

    # ---- 3. MoE counts parity (where twin exposes step counters) ----
    aux_ids = sorted(k for k in result.objects.records
                     if k.startswith("Aux_"))
    counters = [(n, m) for n, m in twin.named_modules()
                if hasattr(m, "step_counts") and m.step_counts is not None]
    if aux_ids and counters:
        from dataflow.tasks.modules.moe.spec import moe_aux_layout

        layout = moe_aux_layout(dims, dims.moe)
        print("\n== MoE counts parity (engine Aux_ vs twin step_counts):")
        for (tname, m), oid in zip(counters, aux_ids):
            r = result.objects.get(oid)
            slot = r.fast or r.backing
            ec = layout.views(slot.buffer)[
                "expert_counts_current_step"].long().cpu()
            tc = m.step_counts.long().cpu()
            d = (ec - tc).abs()
            print(f"  {tname} <-> {oid}: totals eng={int(ec.sum())} "
                  f"twin={int(tc.sum())} | max|delta|={int(d.max())} "
                  f"flips<= {int(d.sum()) // 2}")
    elif aux_ids:
        print("\n== MoE counts parity: engine Aux_ present but twin has no "
              "step_counts counters — extend the twin to expose them")

    result.close()
    dry.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
