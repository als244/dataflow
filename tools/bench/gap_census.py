"""Production inter-task overhead, measured in situ — no profiler.

Runs a real preset cell through a fresh out-of-process daemon for N
steps and computes, from each run's OWN trace intervals (event-resolved
device times the engine records unconditionally), the device-timeline
gap between consecutive compute tasks — the same formula the floor
benchmark uses, applied to the production path. The per-task gap
distribution IS the inter-task overhead: resolver work, retire and
dispatch bookkeeping, transfer-token handling, and enqueue-tail bleed
all land in it, while step time, kernel time, and genuine transfer
waits do not pollute it (waits show as large outliers; the p50/p25 of
block-class gaps track the overhead floor).

Usage: python tools/bench/gap_census.py [--preset gpt2_124m]
    [--steps 10] [--batch 32] [--seq 2048] [--fast-gib 8.4]
    [--backing-gib 24] [--json PATH]
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.getcwd(), "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="gpt2_124m")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--fast-gib", type=float, default=8.4)
    ap.add_argument("--backing-gib", type=float, default=24.0)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    from dataclasses import fields, replace

    import torch

    from dataflow.core.jsonio import program_to_dict
    from dataflow_training.lowering.planning import plan_program
    from dataflow_training.model_families.families import family
    from dataflow_training.run.driver import init_model
    from dataflow_training.run.presets import (cfg_dict, resolve_preset,
                                               resolver_family)
    from dataflow_training.testing.server_process import out_of_process_server

    cfg = resolve_preset(args.preset)
    names = {f.name for f in fields(cfg)}
    kw = {"seq_len": args.seq, "batch": args.batch}
    if "max_seq_len" in names:
        kw["max_seq_len"] = args.seq
    cfg = replace(cfg, **kw)
    fam = family(resolver_family(cfg))
    planned = plan_program(
        fam.lower(cfg),
        fast_memory_capacity=int(args.fast_gib * (1 << 30)))
    resolver_spec = {"kind": "model_family",
                     "family": resolver_family(cfg),
                     "cfg": cfg_dict(cfg)}

    g = torch.Generator().manual_seed(11)
    tokens = torch.randint(0, cfg.vocab_size, (args.batch * args.seq,),
                           generator=g, dtype=torch.int32)
    targets = tokens.clone()
    boundaries = [i * args.seq for i in range(args.batch + 1)]

    per_step = []
    with out_of_process_server(backing_gib=args.backing_gib) as client:
        init_model(client, resolver_family(cfg), cfg_dict(cfg), seed=11)
        client.put_object("tokens_0_0", tokens.numpy().tobytes())
        client.put_object("targets_0_0", targets.numpy().tobytes())
        reg = client.register_program(program_to_dict(planned.program),
                                      resolver=resolver_spec)
        if reg["bindings"]["missing_inputs"]:
            raise RuntimeError(f"unbound inputs: {reg['bindings']}")
        for step in range(args.steps):
            out = client.run(
                reg["prog_id"],
                args={"step": step,
                      "valid_rows": int((targets >= 0).sum()),
                      "seq_lens": {"0": boundaries}},
                trace=True)
            if out.get("state") != "done":
                raise RuntimeError(f"step {step}: {out.get('state')}: "
                                   f"{out.get('outcome')}")
            # wire rows: [task_id, track, start, end]
            ivs = sorted((iv for iv in out["trace"]["intervals"]
                          if iv[1] == "compute"), key=lambda r: r[2])
            gaps = [nxt[2] - prev[3] for prev, nxt in zip(ivs, ivs[1:])]
            per_step.append(gaps)

    steady = [g for gaps in per_step[2:] for g in gaps]  # skip warm steps
    q = statistics.quantiles(steady, n=100)
    print(f"gap census ({args.preset} b{args.fast_gib} "
          f"s{args.seq} x{args.batch}, {args.steps} steps, "
          f"{len(steady)} steady gaps):")
    print(f"  inter-task device gap: p25 {q[24]:7.1f}  p50 {q[49]:7.1f}  "
          f"p75 {q[74]:7.1f}  p90 {q[89]:7.1f} us")
    print(f"  per-step total gap: "
          f"{sum(steady) / max(1, len(per_step) - 2) / 1e3:8.2f} ms")
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump({"preset": args.preset,
                       "gap_p25_us": q[24], "gap_p50_us": q[49],
                       "gap_p75_us": q[74], "gap_p90_us": q[89],
                       "per_step_total_gap_ms":
                           sum(steady) / max(1, len(per_step) - 2) / 1e3},
                      fh, indent=1)


main()
