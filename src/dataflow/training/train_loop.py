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
    placement_escapes: int = 0  # quiescent-deadlock escapes (0 in healthy runs)
    last_trace: object = None  # RunTrace of the final step (replay-gap analysis)
    placement_extent_bytes: int = 0
    placement_overhead: float = 1.0  # extent / peak load (contiguity geometry tax)
    peak_backing_bytes: int = 0      # ledger peak of plan-managed host bytes
    pinned_host_bytes: int = 0       # physically registered pinned memory

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
    physical_limit_bytes: int = 27 * 1024**3,
    placement_mode: str = "static",
    placement=None,
) -> TrainReport:
    """Run `steps` optimizer steps of the (single-step) annotated program.

    `token_stream(step) -> (tokens, targets) int32 cpu tensors` supplies data;
    defaults to seeded random tokens. When `values` is None, initial pinned
    buffers are created AND freed here (tens of GB at 8B scale — leaking them
    across sweep budgets exhausts host RAM); pass your own dict to keep final
    state readable afterwards.
    """
    from dataflow.runtime.placement import PlacementRecorder, compute_placement

    dims = dims_of(cfg)
    owns_values = values is None
    if values is None:
        values = initial_values(annotated, cfg, backend, seed=seed)
    resolver = build_resolver(dims, hyper)
    if placement is None and placement_mode == "static":
        # static placement (default): packing proven against physical VRAM at
        # planning time. placement_mode="dynamic" keeps the online slab+arena
        # path — required for shape-UNstable programs (variable-length
        # sequences change object sizes per round/step, invalidating any
        # recorded instance stream). A pre-computed placement (e.g. from an
        # extent-budget search) can also be injected directly.
        recorder = PlacementRecorder()
        dry = Engine(FakeBackend()).execute(
            annotated, initial_buffers=values, record_placement=recorder
        )
        placement = compute_placement(recorder, physical_limit_bytes=physical_limit_bytes)
    else:
        dry = Engine(FakeBackend()).execute(annotated, initial_buffers=values)
    session = Session(backend=backend)
    gen = torch.Generator().manual_seed(seed + 1)

    def default_stream(_step: int):
        tokens = torch.randint(0, dims.vocab_size, (dims.tokens,), generator=gen, dtype=torch.int32)
        targets = torch.randint(0, dims.vocab_size, (dims.tokens,), generator=gen, dtype=torch.int32)
        return tokens, targets

    token_stream = token_stream or default_stream
    annotator = getattr(backend, "annotator", None)
    if annotator is not None and annotator.enabled:
        annotator.range_push("train_steps")  # nsys --capture-range=nvtx target
    report = TrainReport()
    if placement is not None:
        report.placement_extent_bytes = placement.extent_bytes
        report.placement_overhead = placement.overhead
    rounds = cfg.grad_accum_rounds
    round_views = [
        (
            torch_view(values[f"tokens_0_{r}"], (dims.tokens,), torch.int32),
            torch_view(values[f"targets_0_{r}"], (dims.tokens,), torch.int32),
        )
        for r in range(rounds)
    ]

    try:
        for step in range(steps):
            for r, (tokens_view, targets_view) in enumerate(round_views):
                tokens, targets = token_stream(step * rounds + r)
                tokens_view.copy_(tokens)
                targets_view.copy_(targets)
            # optimizer bias correction advances with the global step
            stepped = _with_step(annotated, step)

            t0 = time.perf_counter()
            if annotator is not None and annotator.enabled:
                annotator.range_push(f"step:{step}")
            try:
                result = Engine(backend, session=session).execute(
                    stepped, resolver=resolver, initial_buffers=values,
                    pool_prewarm=dry.pool_demand, placement=placement,
                )
            finally:
                if annotator is not None and annotator.enabled:
                    annotator.range_pop()
            report.step_wall_s.append(time.perf_counter() - t0)
            report.step_makespan_us.append(result.makespan_us)
            report.peak_fast_bytes = max(report.peak_fast_bytes, result.peak_fast_bytes)
            report.peak_backing_bytes = max(report.peak_backing_bytes, result.peak_backing_bytes)
            report.pinned_host_bytes = max(
                report.pinned_host_bytes, getattr(backend, "pinned_peak", 0)
            )
            prior = sum(report.step_slab_overflows)
            report.step_slab_overflows.append(result.slab_overflows - prior)
            report.placement_escapes = result.placement_escapes

            loss_rec = result.objects.get("loss_0_0")
            slot = loss_rec.backing or loss_rec.fast
            report.losses.append(float(torch_view(slot.buffer, (1,), torch.float32)[0]))
            report.last_trace = result.trace
    finally:
        if annotator is not None and annotator.enabled:
            annotator.range_pop()  # train_steps
        session.close()
        dry.close()
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
