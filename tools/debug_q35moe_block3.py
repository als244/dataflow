#!/usr/bin/env python
"""Isolate qwen35moe block 3 (the gated-attention 'full' block): feed the
ENGINE's own block-2 output into the TWIN's block 3 and compare against
the engine's block-3 output. Upstream divergence is removed, so the
row profile of what remains discriminates the mechanism:

  - few hot rows, rest at the bf16 floor  -> MoE near-tie flips only
  - broadband elevation of every row      -> the mixer (gated attention)
    computes something different: formula/layout suspect, dig further

Also prints the block-3 router margin distribution (twin side) so hot
rows can be matched against near-tie flip candidates.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


def main() -> int:
    from dataflow_training.model_families import bridges
    from dataflow.runtime import Engine
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.runtime.device.fake import FakeBackend
    from dataflow.runtime.engine import uniform_segments
    from dataflow.runtime.interop import torch_view
    from dataflow_training.model_families.families import resolve_family
    from dataflow_training.model_families.qwen35moe import ShapedQwen35MoeConfig
    from dataflow_training.lowering.planning import plan_program
    from dataflow_training.testing.gradcheck import cos_sim, rel_l2
    from reference_models.qwen35moe import rope_tables

    cfg = ShapedQwen35MoeConfig.tiny()
    fam = resolve_family(cfg)
    dims = fam.dims_of(cfg)
    program = fam.lower(cfg)
    keep = {o.id for t in program.tasks for o in t.outputs
            if o.id.startswith("y_")}
    program = dataclasses.replace(
        program,
        final_locations={**dict(program.final_locations),
                         **{i: "fast" for i in keep}})
    planned = plan_program(program, fast_memory_capacity=96 * 1024 * 1024)
    backend = CudaBackend()
    values = fam.initial_values(planned.program, cfg, backend, seed=0)

    twin = bridges.build_reference_model(cfg)
    bridges.load_reference_init(twin, cfg, dims,
                                bridges.get_bytes_from_values(values))
    twin.eval()

    run_args = {"segments": uniform_segments(dims, planned.program)}
    dry = Engine(FakeBackend()).execute(planned.program,
                                        initial_buffers=values)
    result = Engine(backend).execute(
        planned.program, resolver=fam.build_resolver(dims),
        initial_buffers=values, pool_prewarm=dry.pool_demand,
        run_args=run_args)

    def y_of(i):
        rec = result.objects.get(f"y_0_0_{i}")
        slot = rec.fast or rec.backing
        return torch_view(slot.buffer, (dims.tokens, dims.d_model),
                          torch.bfloat16).clone()

    eng_y2 = y_of(2)
    eng_y3 = y_of(3).float().cpu()

    class RecordingLinear(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.rows = []

        def forward(self, x):
            y = self.inner(x)
            self.rows.append(y.detach().float().cpu())
            return y

    blk = twin.blocks[3]
    rec = RecordingLinear(blk.moe.router)
    blk.moe.router = rec

    seq = dims.seq_len
    nseq = dims.tokens // seq
    outs = []
    with torch.no_grad():
        cos_t, sin_t = rope_tables(seq, twin.cfg.rot_dim, twin.cfg.rope_base, "cuda")
        for s in range(nseq):
            x = eng_y2[s * seq:(s + 1) * seq].view(1, seq, dims.d_model)
            outs.append(blk(x, cos_t, sin_t).float().cpu().view(
                seq, dims.d_model))
    twn_y3 = torch.cat(outs)

    rowrel = (eng_y3 - twn_y3).norm(dim=1) / twn_y3.norm(dim=1).clamp_min(
        1e-12)
    med = float(rowrel.median())
    hot = (rowrel > max(10 * med, 0.05)).nonzero().flatten().tolist()
    mask = torch.ones(dims.tokens, dtype=torch.bool)
    mask[hot] = False
    print(f"isolated block 3 on ENGINE inputs:")
    print(f"  rel={rel_l2(eng_y3, twn_y3):.3e} cos={cos_sim(eng_y3, twn_y3):.6f}")
    print(f"  row-rel median {med:.3e}  p90 "
          f"{float(rowrel.quantile(0.9)):.3e}  max {float(rowrel.max()):.3e}")
    print(f"  hot rows {hot}")
    print(f"  rel excl hot {rel_l2(eng_y3[mask], twn_y3[mask]):.3e}")

    logits = torch.cat([r.reshape(-1, twin.cfg.n_experts) for r in rec.rows])
    top = torch.topk(torch.softmax(logits, dim=-1), k=min(
        twin.cfg.top_k + 1, twin.cfg.n_experts), dim=-1).values
    margin = top[:, twin.cfg.top_k - 1] - top[:, twin.cfg.top_k]
    order = torch.argsort(margin)
    print(f"\nblock-3 router margins (kth - (k+1)th softmax prob):")
    print(f"  median {float(margin.median()):.3e}; tightest tokens "
          f"{[int(i) for i in order[:6]]} -> "
          f"{[f'{float(margin[i]):.1e}' for i in order[:6]]}")

    result.close()
    dry.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
