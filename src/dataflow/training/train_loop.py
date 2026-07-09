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

import re
import time
from dataclasses import dataclass, field

import torch

from dataflow.core import Program
from dataflow.runtime import Engine
from dataflow.runtime.device.fake import FakeBackend
from dataflow.runtime.engine import Session
from dataflow.tasks.interop import torch_view
from dataflow.tasks.base_blocks import AdamWHyper
from dataflow.training.families import resolve_family
from dataflow.training.models.llama3 import ShapedLlamaConfig


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
    pressure_evictions: int = 0  # ledger-inversion evictions (0 in healthy runs)
    last_trace: object = None  # RunTrace of the final step (replay-gap analysis)
    placement_extent_bytes: int = 0
    placement_overhead: float = 1.0  # extent / peak load (contiguity geometry tax)
    vmm_stats: dict | None = None    # arena counters when placement_mode="vmm"
    peak_backing_bytes: int = 0      # ledger peak of plan-managed host bytes
    pinned_host_bytes: int = 0       # physically registered pinned memory

    @property
    def steady_state_makespan_us(self) -> float:
        tail = self.step_makespan_us[1:] or self.step_makespan_us
        return sum(tail) / len(tail)


def _make_default_stream_probe(*, vocab: int, tokens: int, seq_len: int,
                               rounds: int, seed: int,
                               decoupled: bool = False):
    """Test seam: the bench default_stream's data contract, minus the
    engine. Returns (stream, seq_len); see test_bench_default_stream_
    semantics."""
    import torch as _torch

    gen = _torch.Generator().manual_seed(seed + 1)
    fixed: dict[int, tuple] = {}

    def stream(k: int):
        r = k % max(1, rounds)
        if r not in fixed:
            toks = _torch.randint(0, vocab, (tokens,), generator=gen,
                                  dtype=_torch.int32)
            if decoupled:
                tgts = _torch.randint(0, vocab, (tokens,), generator=gen,
                                      dtype=_torch.int32)
            else:
                tgts = (toks.view(-1, seq_len).roll(-1, dims=1)
                        .reshape(-1).contiguous())
            fixed[r] = (toks, tgts)
        return fixed[r]

    return stream, seq_len


def train(
    annotated: Program,
    cfg: ShapedLlamaConfig,
    backend,
    *,
    steps: int,
    seed: int = 0,
    hyper: AdamWHyper = AdamWHyper(),
    token_stream=None,
    decoupled_targets: bool = False,
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

    fam = resolve_family(cfg)
    dims = fam.dims_of(cfg)
    owns_values = values is None
    if values is None:
        values = fam.initial_values(annotated, cfg, backend, seed=seed)
    resolver = fam.build_resolver(dims, hyper)
    use_vmm = placement_mode == "vmm"
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
        # placement_mode="vmm": no packing problem exists — per-object stable
        # VAs are backed by pooled physical extents sized to the LEDGER, so
        # physical tracks logical by construction (device/vmm.py).
        # placement_mode="dynamic": online slab+arena (shape-unstable programs).
        dry = Engine(FakeBackend()).execute(annotated, initial_buffers=values)
    session = Session(backend=backend)
    gen = torch.Generator().manual_seed(seed + 1)

    # Bench data semantics (Shein): random token sequences with SHIFTED
    # targets (next-token, mimicking pretraining), unique across the
    # batch and ga rounds WITHIN a step, and REPEATED across steps — so
    # a healthy run shows a real learning signal: step-0 loss ~ ln(V)
    # (+ init logit variance), then a slow decline as the model
    # memorizes the fixed set. Targets shift within each sequence; the
    # last position wraps to the sequence's first token.
    _fixed_rounds: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def default_stream(k: int):
        rounds_ = max(1, cfg.grad_accum_rounds)
        r = k % rounds_
        if r not in _fixed_rounds:
            tokens = torch.randint(0, dims.vocab_size, (dims.tokens,),
                                   generator=gen, dtype=torch.int32)
            if decoupled_targets:
                # independently drawn targets (still fixed across steps):
                # memorization signal with NO causal/sequence structure —
                # copy-from-context shortcuts cannot help here
                targets = torch.randint(0, dims.vocab_size, (dims.tokens,),
                                        generator=gen, dtype=torch.int32)
            else:
                seq = dims.seq_len
                targets = (tokens.view(-1, seq).roll(-1, dims=1)
                           .reshape(-1).contiguous())
            _fixed_rounds[r] = (tokens, targets)
        return _fixed_rounds[r]

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
            # indexer-only warm-up programs carry no targets (no head/CE)
            (torch_view(values[f"targets_0_{r}"], (dims.tokens,), torch.int32)
             if f"targets_0_{r}" in values else None),
        )
        for r in range(rounds)
    ]
    # every run provides segments: the fixed-shape training path builds the
    # uniform per-round descriptor from dims (round keys are stable across
    # steps, so one run_args serves every step; the prologue materializes it)
    from dataflow.runtime.engine import uniform_segments

    run_args = {"segments": uniform_segments(dims, annotated)}

    try:
        for step in range(steps):
            t0 = time.perf_counter()  # full-step wall: fill + execute + readback
            for r, (tokens_view, targets_view) in enumerate(round_views):
                tokens, targets = token_stream(step * rounds + r)
                tokens_view.copy_(tokens)
                if targets_view is not None:
                    targets_view.copy_(targets)
            # optimizer bias correction advances with the global step
            stepped = _with_step(annotated, step)

            if annotator is not None and annotator.enabled:
                annotator.range_push(f"step:{step}")
            try:
                result = Engine(backend, session=session).execute(
                    stepped, resolver=resolver, initial_buffers=values,
                    pool_prewarm=dry.pool_demand, placement=placement,
                    vmm=use_vmm, annotate_rename=_annotate_step(step),
                    run_args=run_args,
                )
            finally:
                if annotator is not None and annotator.enabled:
                    annotator.range_pop()
            report.step_makespan_us.append(result.makespan_us)
            report.peak_fast_bytes = max(report.peak_fast_bytes, result.peak_fast_bytes)
            report.peak_backing_bytes = max(report.peak_backing_bytes, result.peak_backing_bytes)
            report.pinned_host_bytes = max(
                report.pinned_host_bytes, getattr(backend, "pinned_peak", 0)
            )
            prior = sum(report.step_slab_overflows)
            report.step_slab_overflows.append(result.slab_overflows - prior)
            report.placement_escapes = result.placement_escapes
            report.pressure_evictions += result.pressure_evictions

            report.step_wall_s.append(time.perf_counter() - t0)
            loss_rec = result.objects.get("loss_0_0")
            slot = loss_rec.backing or loss_rec.fast
            report.losses.append(float(torch_view(slot.buffer, (1,), torch.float32)[0]))
            report.last_trace = result.trace
        if use_vmm and session.pool is not None and session.pool.vmm is not None:
            arena = session.pool.vmm
            # the physical reservation plays the extent role in device-peak
            # accounting; overhead vs ledger peak is pure rounding+headroom
            report.placement_extent_bytes = arena.pool_bytes
            report.placement_overhead = arena.pool_bytes / max(report.peak_fast_bytes, 1)
            report.vmm_stats = {
                "maps": arena.maps,
                "handle_creates": arena.handle_creates,
                "handle_reflows": arena.handle_reflows,
                "prewarmed": arena.prewarmed,
                "slot_adoptions": arena.slot_adoptions,
                "t_create_s": round(arena.t_create_s, 4),
                "t_destroy_s": round(arena.t_destroy_s, 4),
                "t_map_s": round(arena.t_map_s, 4),
                "t_unmap_s": round(arena.t_unmap_s, 4),
                "reclaim_drains": arena.reclaim_drains,
                "peak_mapped_bytes": arena.peak_used_bytes,
                "peak_physical_bytes": arena.peak_created_bytes,
                "pool_bytes": arena.pool_bytes,
            }
    finally:
        if annotator is not None and annotator.enabled:
            annotator.range_pop()  # train_steps
        from dataflow.tasks.interop import clear_view_cache

        clear_view_cache()  # cached views must not outlive the pool's buffers
        session.close()
        dry.close()
        if owns_values:
            for buf in values.values():
                backend.free(buf)
    return report


# Step-scoped id families from the lowering, longest-prefix first. A replayed
# single-step plan bakes step 0 into these; NVTX display names substitute the
# GLOBAL step so profiler rows read block_fwd_{step}_{round}_{layer}. W_{i} /
# O_{i} carry a LAYER index, not a step - deliberately absent from this list.
_STEP0_ID = re.compile(
    r"^(embed_fwd|block_fwd|head_loss|block_recompute|block_bwd"
    r"|embed_bwd|optimizer_embed|optimizer_head|optimizer"
    r"|tokens|targets|y_embed|y|A|dy_embed|dy|loss"
    r"|dW_embed|dW_head|dW)_0(?=_|$)"
)


def _annotate_step(step: int):
    '''NVTX-only renamer: rewrite the baked step-0 field to the global step.'''
    if step == 0:
        return None
    return lambda name: _STEP0_ID.sub(rf"\g<1>_{step}", name, count=1)


def _with_step(program: Program, step: int) -> Program:
    from dataclasses import replace

    if step == 0:
        return program
    new_tasks = tuple(
        replace(t, block_params={**t.block_params, "step": step}) if t.group == "optimizer" else t
        for t in program.tasks
    )
    return replace(program, tasks=new_tasks)
