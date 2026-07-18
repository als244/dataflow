#!/usr/bin/env python
"""dW-space probe for the dsv3 engine-vs-twin gradient gap.

Compares the engine's accumulated gradient slabs (dW_* objects, read
straight out of the run result and unpacked per grad_layout) against
the twin's autograd par.grad, field by field in twin-name space. No
optimizer, no bf16 weight-storage rounding — this is the gradient
comparison itself.

--save FILE / --control FILE: write the engine grad dict / compare a
fresh process's engine grads against a saved one (cross-process ambient
floor: same math, pure kernel-selection noise).
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402


def engine_grad_dict(cfg, dims, result) -> dict:
    """Mirror of bridges.dsv3.to_dsv3_state_dict over dW objects."""
    from dataflow_training.model_families.dsv3.bridge import (
        mla_attention_entries,
        transposed,
    )
    from dataflow.runtime.interop import torch_view
    from dataflow_training.blocks.layouts import (
        dsv3_dense_weight_layout,
        dsv3_moe_weight_layout,
        embed_weight_layout,
        grad_layout,
        head_weight_layout,
    )

    ids = sorted(k for k in result.objects.records if k.startswith("dW"))
    print(f"dW objects: {ids}")

    def dw_bytes(oid):
        rec = result.objects.get(oid)
        slot = rec.fast or rec.backing
        return torch_view(slot.buffer, (slot.buffer.size_bytes,),
                          torch.uint8)

    def block_id(i):
        cands = [o for o in ids if o.endswith(f"_{i}")
                 and not o.startswith(("dW_embed", "dW_head"))]
        assert len(cands) == 1, (i, cands)
        return cands[0]

    op = getattr(dims, "opt_policy", None)
    gd: dict[str, torch.Tensor] = {}
    ew = grad_layout(embed_weight_layout(dims), dims.dtypes, ns="embed",
                     opt_policy=op)
    gd["embed.weight"] = ew.unpack_tensor(dw_bytes("dW_embed_0"))["w"].clone()
    hw = grad_layout(head_weight_layout(dims), dims.dtypes, ns="head",
                     opt_policy=op)
    h = hw.unpack_tensor(dw_bytes("dW_head_0"))
    gd["lm_head.weight"] = h["w"].clone()
    gd["final_norm.weight"] = h["final_norm_w"].clone()
    for i in range(cfg.n_layers):
        p = f"blocks.{i}"
        dense = dims.kinds[i] == "dense"
        wl = (dsv3_dense_weight_layout if dense
              else dsv3_moe_weight_layout)(dims, layer=i)
        w = grad_layout(wl, dims.dtypes, layer=i,
                        opt_policy=op).unpack_tensor(dw_bytes(block_id(i)))
        gd[f"{p}.attn_norm.weight"] = w["attn_norm_w"].clone()
        mla_attention_entries(gd, p, w)
        gd[f"{p}.ffn_norm.weight"] = w["ffn_norm_w"].clone()
        if dense:
            for k in ("w1", "w3", "w2"):
                gd[f"{p}.ffn.{k}.weight"] = transposed(w[k])
        else:
            gd[f"{p}.ffn.router.weight"] = transposed(w["w_router"])
            # w_router_bias is policy-frozen: no dW field exists for it
            gd[f"{p}.ffn.w13"] = w["w13_experts"].clone()
            gd[f"{p}.ffn.w2"] = w["w2_experts"].clone()
            gd[f"{p}.ffn.w_s13.weight"] = transposed(w["w_s13"])
            gd[f"{p}.ffn.w_s2.weight"] = transposed(w["w_s2"])
    return gd


def main() -> int:
    from dataflow.runtime import Engine
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.runtime.device.fake import FakeBackend
    from dataflow_training.data.segments import uniform_segments
    from dataflow_training.model_families.families import resolve_family
    from dataflow_training.model_families.dsv3 import ShapedDsv3Config
    from dataflow_training.lowering.planning import plan_program
    from dataflow_training.testing.gradcheck import (
        cos_sim,
        reference_model_step,
        rel_l2,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="ragged",
                    choices=("uniform", "ragged"))
    ap.add_argument("--save", type=Path, default=None)
    ap.add_argument("--control", type=Path, default=None)
    args = ap.parse_args()

    cfg = ShapedDsv3Config.tiny()
    if args.shape == "ragged":
        t = cfg.seq_len * cfg.batch
        a, b = t // 2 + 3, t // 4 + 1
        cfg = replace(cfg, seq_lens=(a, b, t - a - b))

    import dataclasses

    fam = resolve_family(cfg)
    dims = fam.dims_of(cfg)
    program = fam.lower(cfg)
    # drop the optimizer tasks and RETAIN the gradient slabs: dW ids in
    # final_locations are terminal placement constraints — everything
    # else is disposable after last use and gets recycled
    dw_ids = {o.id for t in program.tasks for o in t.outputs
              if o.id.startswith("dW")
              or o.id.startswith(("y_", "dy_"))}
    program = dataclasses.replace(
        program,
        tasks=tuple(t for t in program.tasks
                    if not t.id.startswith("optimizer_")),
        final_locations={**dict(program.final_locations),
                         **{i: "fast" for i in dw_ids}})
    planned = plan_program(program, fast_memory_capacity=64 * 1024 * 1024)
    backend = CudaBackend()
    values = fam.initial_values(planned.program, cfg, backend, seed=0)

    _, twin, _, _ = reference_model_step(
        cfg, values, seq_lens=getattr(dims, "seq_lens", None))
    twin_grads = {n: p.grad.detach().float().cpu()
                  for n, p in twin.named_parameters() if p.grad is not None}

    run_args = {"segments": uniform_segments(dims, planned.program)}
    dry = Engine(FakeBackend()).execute(planned.program,
                                        initial_buffers=values)
    result = Engine(backend).execute(
        planned.program, resolver=fam.build_resolver(dims),
        initial_buffers=values, pool_prewarm=dry.pool_demand,
        run_args=run_args)

    gd = {k: v.detach().float().cpu()
          for k, v in engine_grad_dict(cfg, dims, result).items()}

    if args.save:
        torch.save(gd, args.save)
        print(f"saved engine grads -> {args.save}")

    other, other_name = twin_grads, "twin-autograd"
    if args.control:
        other = torch.load(args.control, weights_only=True)
        other_name = f"saved-engine({args.control})"

    # forward-divergence profile: engine y_* objects vs twin block
    # outputs captured by composition recorders during the SAME per-seq
    # forwards (concatenated in token order)
    import torch.nn as nn
    from dataflow.runtime.interop import torch_view as tv

    class BlockRecorder(nn.Module):
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

    twin2 = __import__("dataflow_training.model_families.bridges",
                       fromlist=["bridges"]).build_reference_model(cfg)
    from dataflow_training.model_families import bridges as _b
    _b.load_reference_init(twin2, cfg, dims,
                           _b.get_bytes_from_values(values))
    twin2.eval()
    recs = []
    for bi, blk in enumerate(twin2.blocks):
        rec = BlockRecorder(blk)
        twin2.blocks[bi] = rec
        recs.append(rec)
    tok = tv(values["tokens_0_0"], (dims.tokens,), torch.int32).long().cuda()
    tgt = tv(values["targets_0_0"], (dims.tokens,),
             torch.int32).long().cuda() if "targets_0_0" in values else tok
    lens2 = getattr(dims, "seq_lens", None) or (dims.seq_len,) * (
        dims.tokens // dims.seq_len)
    with torch.no_grad():
        lo = 0
        for ln in lens2:
            twin2.loss(tok[lo:lo + ln].view(1, ln),
                       tgt[lo:lo + ln].view(1, ln))
            lo += ln
    from dataflow_training.testing.gradcheck import cos_sim as _cs
    from dataflow_training.testing.gradcheck import rel_l2 as _rl
    print("\nforward divergence by depth (engine y_* vs twin blocks):")
    for bi, rec in enumerate(recs):
        oid = f"y_0_0_{bi}"
        r = result.objects.get(oid)
        slot = r.fast or r.backing
        eng = tv(slot.buffer, (dims.tokens, dims.d_model),
                 torch.bfloat16).float().cpu()
        twn = torch.cat(rec.outs)
        rowrel = (eng - twn).norm(dim=1) / twn.norm(dim=1).clamp_min(1e-12)
        med = float(rowrel.median())
        hot = (rowrel > max(10 * med, 0.05)).nonzero().flatten().tolist()
        keep = torch.ones(eng.shape[0], dtype=torch.bool)
        keep[hot] = False
        print(f"  {oid}: rel={_rl(eng, twn):.3e} cos={_cs(eng, twn):.6f} "
              f"| row-rel median {med:.2e}; hot rows {hot} "
              f"(max {float(rowrel.max()):.2f}); "
              f"rel excl hot rows {_rl(eng[keep], twn[keep]):.3e}")
    r = result.objects.get("dy_0_0_2")
    slot = r.fast or r.backing
    eng_dy = tv(slot.buffer, (dims.tokens, dims.d_model),
                torch.bfloat16).float().cpu()
    print(f"  dy_0_0_2 (dL/dh_final): |eng|={float(eng_dy.norm()):.3e}")

    print(f"\nengine-dW vs {other_name}, worst rel_l2 first:")
    rows = []
    for name, g_e in gd.items():
        g_o = other.get(name)
        if g_o is None or g_o.shape != g_e.shape:
            continue
        rows.append((rel_l2(g_e, g_o), name, cos_sim(g_e, g_o),
                     float(g_e.norm()), float(g_o.norm())))
    rows.sort(reverse=True)
    for err, name, cos, ne, no in rows:
        print(f"  {err:11.3e}  cos={cos:.6f}  |eng|={ne:9.3e} "
              f"|other|={no:9.3e}  {name}")

    result.close()
    dry.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
