#!/usr/bin/env python
"""Minimal repro/instrumentation for the dsv3 ragged router_bias
disagreement (engine vs isolated twin).

Question under test: the noaux balance rule is b += speed*sign(mean(c)-c)
on BOTH sides with identical tie semantics (torch.sign), so any bias
disagreement must come from the COUNTS c. This script runs the exact
failing gate case (dsv3 tiny, ragged (259,129,124)) and prints, per MoE
layer:

  1. twin per-sequence counts + step aggregate
  2. engine expert_counts_current_step (read from the Aux_ object)
  3. count totals (must be tokens*top_k on both sides — a total mismatch
     means double/under-counting, a REAL bug)
  4. per-expert count deltas and where sign(mean-c) disagrees
  5. twin-side top-k selection margins (kth selected vs best rejected,
     biased scores) — near-ties at bf16-ulp scale are flip candidates

Verdict logic: totals equal + deltas of +-1..2 concentrated on
near-tie tokens => selection near-tie lottery (invariance gap, not math
error). Totals unequal or structured deltas => engine counting bug.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402


def ragged_partition(cfg):
    t = cfg.seq_len * cfg.batch
    a = t // 2 + 3
    b = t // 4 + 1
    return (a, b, t - a - b)


def main() -> int:
    from dataflow_training.model_families import bridges
    from dataflow.runtime import Engine
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.runtime.device.fake import FakeBackend
    from dataflow_training.data.segments import uniform_segments
    from dataflow.runtime.interop import torch_view
    from dataflow_training.blocks.modules.moe.spec import moe_aux_layout
    from dataflow_training.model_families.families import resolve_family
    from dataflow_training.model_families.dsv3 import ShapedDsv3Config
    from dataflow_training.lowering.planning import plan_program

    cfg = ShapedDsv3Config.tiny()
    lens = ragged_partition(cfg)
    cfg = replace(cfg, seq_lens=lens)
    print(f"cfg: tokens={cfg.seq_len * cfg.batch} lens={lens} "
          f"E={cfg.n_experts} top_k={cfg.top_k} n_group={cfg.n_group} "
          f"topk_group={cfg.topk_group} speed={cfg.bias_update_speed} "
          f"aux_coef={cfg.aux_coef}")

    fam = resolve_family(cfg)
    dims = fam.dims_of(cfg)
    print(f"dims: tokens={dims.tokens} "
          f"seq_len={getattr(dims, 'seq_len', '<absent>')} "
          f"seq_lens={getattr(dims, 'seq_lens', '<absent>')}")
    program = fam.lower(cfg)
    planned = plan_program(program, fast_memory_capacity=64 * 1024 * 1024)
    backend = CudaBackend()
    values = fam.initial_values(planned.program, cfg, backend, seed=0)

    # ---- twin leg (instrumented copy of reference_model_step) ----
    model = bridges.build_reference_model(cfg)
    bridges.load_reference_init(model, cfg, dims,
                                bridges.get_bytes_from_values(values))
    model.train()
    tokens = torch_view(values["tokens_0_0"], (dims.tokens,),
                        torch.int32).long().cuda()
    targets = (torch_view(values["targets_0_0"], (dims.tokens,),
                          torch.int32).long().cuda()
               if "targets_0_0" in values else tokens.clone())
    moe_layers = [(name, mod) for name, mod in model.named_modules()
                  if hasattr(mod, "apply_bias_update")]
    print(f"twin MoE layers: {[n for n, _ in moe_layers]}")

    # twin lens: the FIXED harness semantics — the config's static ragged
    # partition, one independent row per sequence (twin attention is
    # per-row causal, never boundary-crossing)
    twin_lens = tuple(getattr(dims, "seq_lens", None) or (dims.seq_len,))
    print(f"twin lens: {twin_lens}")

    step_valid = int((targets >= 0).sum())
    margins_all: dict[str, list] = {n: [] for n, _ in moe_layers}
    per_seq_counts: dict[str, list] = {n: [] for n, _ in moe_layers}

    # margin probe: recompute selection outside the model per forward,
    # from each layer's captured router input? Simpler: monkey-free —
    # after each per-seq forward read layer.last_counts, and compute
    # margins from a separate manual pass below using the twin weights.
    lo = 0
    for ln in twin_lens:
        t = tokens[lo:lo + ln].view(1, ln)
        g = targets[lo:lo + ln].view(1, ln)
        valid = int((g >= 0).sum())
        ce = model.loss(t, g)
        (ce * (valid / step_valid)).backward()
        for n, mod in moe_layers:
            per_seq_counts[n].append(mod.last_counts.clone().cpu())
        lo += ln
    torch.cuda.synchronize()

    twin_counts = {n: mod.step_counts.clone().cpu()
                   for n, mod in moe_layers}

    # margin probe: replay the FIRST MoE layer's routing per sequence
    # via a fresh forward hookless pass is invasive; instead compute
    # margins from the model's routing math on the recorded logits is
    # not directly available — so approximate at the SELECTION level:
    # rerun model.loss per sequence with no_grad and ask each layer to
    # report margins by temporarily setting a flag the twin honors?
    # The twin has no such flag; skip in-model margins and measure the
    # DELTA structure instead (step counts + engine counts + totals),
    # which is sufficient to separate "few +-1 flips" from "structural".

    # twin bias after rule (fresh copy so we can also print sign vec)
    twin_bias = {}
    twin_sign = {}
    speed = float(cfg.bias_update_speed)
    for n, mod in moe_layers:
        c = mod.step_counts.float()
        twin_sign[n] = torch.sign(c.mean() - c).cpu()
        mod.apply_bias_update(speed)
        twin_bias[n] = mod.router_bias.clone().cpu()

    # margin probe: replay layer-1 routing on captured logits. The
    # router input is captured by swapping the layer's nn.Linear for a
    # recording composition (explicit, no hooks); a fresh no_grad replay
    # of the per-seq forwards collects every token's logits.
    import torch.nn as nn

    class RecordingLinear(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.rows = []

        def forward(self, x):
            y = self.inner(x)
            self.rows.append(y.detach().float().cpu())
            return y

    probe_model = bridges.build_reference_model(cfg)
    bridges.load_reference_init(probe_model, cfg, dims,
                                bridges.get_bytes_from_values(values))
    probe_model.eval()
    layer1 = dict(probe_model.named_modules())["blocks.1.ffn"]
    rec = RecordingLinear(layer1.router)
    layer1.router = rec
    with torch.no_grad():
        lo = 0
        for ln in twin_lens:
            probe_model.loss(tokens[lo:lo + ln].view(1, ln),
                             targets[lo:lo + ln].view(1, ln))
            lo += ln
    logits = torch.cat([r.reshape(-1, cfg.n_experts) for r in rec.rows])
    scores = torch.sigmoid(logits)          # bias = 0 at step 0
    s67 = (scores[:, 6] - scores[:, 7]).abs()
    order = torch.argsort(s67)
    print("\nlayer-1 margin probe (|sigmoid_6 - sigmoid_7| per token):")
    print(f"  tokens: {scores.shape[0]}; min gaps: "
          f"{[f'{s67[i]:.2e}' for i in order[:5]]}")
    print(f"  median gap: {s67.median():.3e}")
    t0 = int(order[0])
    print(f"  closest token {t0}: s6={scores[t0, 6]:.9f} "
          f"s7={scores[t0, 7]:.9f} logit6={logits[t0, 6]:.6f} "
          f"logit7={logits[t0, 7]:.6f}")

    # ---- engine leg (exact check_model_step invocation) ----
    from dataflow_training.data.segments import uniform_segments
    run_args = {"segments": uniform_segments(dims, planned.program)}
    seg0 = next(iter(run_args["segments"].values()))
    print(f"engine segments: {seg0}")
    dry = Engine(FakeBackend()).execute(planned.program,
                                        initial_buffers=values)
    result = Engine(backend).execute(
        planned.program,
        resolver=fam.build_resolver(dims),
        initial_buffers=values,
        pool_prewarm=dry.pool_demand,
        run_args=run_args,
    )

    aux_ids = sorted(k for k in result.objects.records if k.startswith("Aux_"))
    print(f"engine Aux objects: {aux_ids}")
    layout = moe_aux_layout(dims, dims.moe)
    engine_counts = {}
    for oid in aux_ids:
        rec = result.objects.get(oid)
        slot = rec.backing or rec.fast
        views = layout.views(slot.buffer)
        engine_counts[oid] = views["expert_counts_current_step"].clone().cpu()

    engine_state = bridges.to_reference_state_dict(
        cfg, __import__("dataflow_training.testing.gradcheck",
                        fromlist=["EngineFinalBytes"]).EngineFinalBytes(result))
    engine_bias = {k: v.clone().float().cpu() for k, v in engine_state.items()
                   if k.endswith("router_bias")}

    # ---- report ----
    expect_total = dims.tokens * cfg.top_k
    print(f"\nexpected count total per layer: tokens*top_k = {expect_total}")
    aux_sorted = list(engine_counts.items())
    for (twin_name, tc), (oid, ec) in zip(sorted(twin_counts.items()),
                                          aux_sorted):
        tc_i = tc.long()
        ec_i = ec.long()
        d = (ec_i - tc_i)
        sign_t = twin_sign[twin_name].long()
        sign_e = torch.sign(ec_i.float().mean() - ec_i.float()).long()
        print(f"\n== twin {twin_name}  <->  engine {oid}")
        print(f"  twin per-seq counts: "
              f"{[c.tolist() for c in per_seq_counts[twin_name]]}")
        print(f"  twin step counts:   {tc_i.tolist()}  "
              f"(total {int(tc_i.sum())})")
        print(f"  engine step counts: {ec_i.tolist()}  "
              f"(total {int(ec_i.sum())})")
        print(f"  delta (eng-twin):   {d.tolist()}")
        print(f"  twin sign(mean-c):  {sign_t.tolist()}")
        print(f"  eng  sign(mean-c):  {sign_e.tolist()}")
        print(f"  sign disagreements: "
              f"{int((sign_t != sign_e).sum())} of {len(sign_t)}")
        eb = engine_bias.get(twin_name + ".router_bias")
        tb = twin_bias[twin_name]
        if eb is not None:
            print(f"  engine final bias:  {eb.tolist()}")
            print(f"  twin   final bias:  {tb.tolist()}")

    result.close()
    dry.close()
    from dataflow.runtime.interop import clear_view_cache
    clear_view_cache()
    for buf in values.values():
        backend.free(buf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
