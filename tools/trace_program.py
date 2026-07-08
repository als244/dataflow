"""Trace an arbitrary dataflow program's runtime event timeline.

    python tools/trace_program.py --program plan.annotated.json
    python tools/trace_program.py --program cells/<cell>/plan.json \\
        --kinds reserve,transfer_reserve,pressure_evict --out events.jsonl

Executes the program on the FAKE backend — no GPU, no kernels, no
initial values needed; tasks and transfers advance the clock by their
PLANNED durations, and the engine walks the exact residency/admission
machinery the real run uses (reserves, transfer charges, releases,
pressure evictions, placement escapes). The emitted event stream is
the runtime's own RunTrace, i.e. what the engine-vs-sim parity gate
compares — the right instrument for debugging a PLANNED training
program's memory behavior before (or without) running it for real.

Prints a table (or writes JSONL with --out) and a summary line:
pressure_evictions / placement_escapes / peak fast bytes. A healthy
plan shows ZERO evictions and escapes — nonzero means realized-order
divergence machinery engaged (see the reserve-order-inversion design
note for the taxonomy).

Event kinds: reserve (task output charge), transfer_reserve (transfer
destination charge — the transfer-lane twin of a task reserve),
transfer_enqueue / transfer_end, release, offload_*, pressure_evict,
placement_escape.

For a REAL-GPU trace of a training run, use tools/gap_analysis.py
(trace.json) — that path fills real weights and measures real timings.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataflow.core import load_program
from dataflow.runtime.device.fake import FakeBackend
from dataflow.runtime.engine import Engine


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--program", type=Path, required=True,
                    help="a Program JSON (annotated plan.json / "
                         "*.annotated.json / bare program.json)")
    ap.add_argument("--kinds", default=None,
                    help="comma list to filter event kinds (default: all)")
    ap.add_argument("--intervals", action="store_true",
                    help="also print task/transfer intervals")
    ap.add_argument("--out", type=Path, default=None,
                    help="write events as JSONL instead of a table")
    args = ap.parse_args()

    program = load_program(args.program)
    if program.metadata:
        print(f"# metadata: {json.dumps(dict(program.metadata))[:300]}")

    result = Engine(FakeBackend()).execute(program)
    try:
        kinds = set(args.kinds.split(",")) if args.kinds else None
        events = [e for e in result.trace.events
                  if kinds is None or e.kind in kinds]

        if args.out:
            with open(args.out, "w") as f:
                for e in events:
                    f.write(json.dumps({
                        "t_us": e.t, "kind": e.kind,
                        "object_id": e.object_id, "task_id": e.task_id,
                        "detail": getattr(e, "detail", None),
                    }) + "\n")
            print(f"wrote {len(events)} events -> {args.out}")
        else:
            print(f"{'t_us':>14}  {'kind':<18} {'object':<22} {'task':<26} detail")
            for e in events:
                print(f"{e.t:>14.1f}  {e.kind:<18} {str(e.object_id or ''):<22} "
                      f"{str(e.task_id or ''):<26} {getattr(e, 'detail', '') or ''}")

        if args.intervals:
            print(f"\n{'start':>14} {'end':>14}  interval")
            for iv in result.trace.intervals:
                print(f"{iv.start:>14.1f} {iv.end:>14.1f}  {iv.task_id}")

        print(f"\nsummary: events={len(events)} "
              f"pressure_evictions={result.pressure_evictions} "
              f"placement_escapes={getattr(result, 'placement_escapes', 0)} "
              f"peak_fast={result.peak_fast_bytes:,}B"
              if hasattr(result, "peak_fast_bytes") else
              f"\nsummary: events={len(events)} "
              f"pressure_evictions={result.pressure_evictions}")
    finally:
        result.close()


if __name__ == "__main__":
    main()
