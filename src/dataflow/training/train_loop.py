"""Multi-step training driver.

One annotated chain, replayed once per optimizer step:

- persistent state (W_*, O_*) lives in caller-provided pinned buffers; the
  plan's final offloads overwrite them in place, so step N+1's initial
  objects ARE step N's results — no copies, no replanning;
- per-step inputs (tokens/targets) are refreshed in their pinned buffers
  between steps;
- a runtime Session keeps the device slab + pinned pool alive across steps
  (steady-state steps perform zero vendor allocations).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from dataflow.core import Program
from dataflow.runtime import Engine
from dataflow.runtime.device.fake import FakeBackend
from dataflow.runtime.engine import Session
from dataflow.tasks.interop import torch_view
from dataflow.tasks.llama3_blocks import AdamWHyper, build_resolver
from dataflow.training.llama3_lowering import dims_of, initial_values
from dataflow.training.shaped_llama3 import ShapedLlamaConfig


@dataclass
class TrainReport:
    losses: list[float] = field(default_factory=list)
    step_wall_s: list[float] = field(default_factory=list)
    step_makespan_us: list[float] = field(default_factory=list)
    peak_fast_bytes: int = 0
    # per-step NEW overflow allocations (cumulative pool counter, diffed);
    # step 0 may overflow at tight budgets (headroom heuristic), steady
    # state must not — overflowed buffers join the free lists and get reused
    step_slab_overflows: list[int] = field(default_factory=list)
    last_trace: object = None  # RunTrace of the final step (replay-gap analysis)

    @property
    def steady_state_makespan_us(self) -> float:
        tail = self.step_makespan_us[1:] or self.step_makespan_us
        return sum(tail) / len(tail)


def train(
    annotated: Program,
    cfg: ShapedLlamaConfig,
    backend,
    *,
    steps: int,
    seed: int = 0,
    hyper: AdamWHyper = AdamWHyper(),
    token_stream=None,
    values: dict | None = None,
) -> TrainReport:
    """Run `steps` optimizer steps of the (single-step) annotated program.

    `token_stream(step) -> (tokens, targets) int32 cpu tensors` supplies data;
    defaults to seeded random tokens. When `values` is None, initial pinned
    buffers are created AND freed here (tens of GB at 8B scale — leaking them
    across sweep budgets exhausts host RAM); pass your own dict to keep final
    state readable afterwards.
    """
    dims = dims_of(cfg)
    owns_values = values is None
    if values is None:
        values = initial_values(annotated, cfg, backend, seed=seed)
    resolver = build_resolver(dims, hyper)
    dry = Engine(FakeBackend()).execute(annotated, initial_buffers=values)
    session = Session(backend=backend)
    gen = torch.Generator().manual_seed(seed + 1)

    def default_stream(_step: int):
        tokens = torch.randint(0, dims.vocab_size, (dims.tokens,), generator=gen, dtype=torch.int32)
        targets = torch.randint(0, dims.vocab_size, (dims.tokens,), generator=gen, dtype=torch.int32)
        return tokens, targets

    token_stream = token_stream or default_stream
    report = TrainReport()
    tokens_view = torch_view(values["tokens_0_0"], (dims.tokens,), torch.int32)
    targets_view = torch_view(values["targets_0_0"], (dims.tokens,), torch.int32)

    try:
        for step in range(steps):
            tokens, targets = token_stream(step)
            tokens_view.copy_(tokens)
            targets_view.copy_(targets)
            # optimizer bias correction advances with the global step
            stepped = _with_step(annotated, step)

            t0 = time.perf_counter()
            result = Engine(backend, session=session).execute(
                stepped, resolver=resolver, initial_buffers=values, pool_prewarm=dry.pool_demand,
            )
            report.step_wall_s.append(time.perf_counter() - t0)
            report.step_makespan_us.append(result.makespan_us)
            report.peak_fast_bytes = max(report.peak_fast_bytes, result.peak_fast_bytes)
            prior = sum(report.step_slab_overflows)
            report.step_slab_overflows.append(result.slab_overflows - prior)

            loss_rec = result.objects.get("loss_0_0")
            slot = loss_rec.backing or loss_rec.fast
            report.losses.append(float(torch_view(slot.buffer, (1,), torch.float32)[0]))
            report.last_trace = result.trace
    finally:
        session.close()
        if owns_values:
            for buf in values.values():
                backend.free(buf)
    return report


def _with_step(program: Program, step: int) -> Program:
    from dataclasses import replace

    if step == 0:
        return program
    new_tasks = tuple(
        replace(t, block_params={**t.block_params, "step": step}) if t.group == "optimizer" else t
        for t in program.tasks
    )
    return replace(program, tasks=new_tasks)
