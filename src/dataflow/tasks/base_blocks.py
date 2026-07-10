"""Family-NEUTRAL task executables and plumbing, shared by every model
family: the executable base class, embed forward/backward, the fused
head+loss, and the per-field-policy optimizer step (plus AdamWHyper and
the profiler token-fill). Family block modules subclass the Block*
templates from llama3_blocks and bind these classes directly in their
resolvers.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from dataflow.core import TaskSpec
from dataflow.runtime.executable import TaskContext

from . import ops
from .interop import TORCH_DTYPE_BY_NAME, external_stream, torch_view
from .kernels import KernelCtx, KernelSet, resolve_kernels
from .layouts import (
    LlamaDims,
    PackedLayout,
    activation_layout,
    embed_weight_layout,
    grad_layout,
    head_weight_layout,
    opt_state_layout,
    weight_layout,
)



@dataclass(frozen=True)
class RoundPrologue:
    """The round-boundary task: publishes the CURRENT ROUND both as an
    object-backed value (its 4-byte int32 output — tasks that need the
    round DEPEND on it, so ordering and recompute-safety ride the
    dataflow) and into the engine's mutable ``run_values`` channel
    (``ctx.run_values["current_round"]``, the ergonomic read; ``run_args``
    stays immutable). Round 0 will additionally zero each aux layer's
    per-step expert counts once the persistent Aux objects land."""

    dims: object = None
    kernels: object = None

    def launch(self, ctx) -> None:
        from .interop import torch_view
        from .modules.moe.spec import moe_aux_layout

        r = int(ctx.task.block_params["round"])
        if ctx.run_values is not None:
            ctx.run_values["current_round"] = r
        for buf in ctx.outputs.values():
            torch_view(buf, (1,), torch.int32).fill_(r)
        if r == 0 and ctx.task.mutates:
            # round 0 zeroes every aux layer's per-step counts (the
            # all-of-training aggregate is untouched)
            layout = moe_aux_layout(self.dims, self.dims.moe)
            for m in ctx.task.mutates:
                if m.startswith("Aux_"):
                    layout.views(ctx.mutates[m])["expert_counts_current_step"].zero_()


@dataclass(frozen=True)
class AdamWHyper:
    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.0
    momentum: float = 0.95      # sgdm/muon (tasks/optim.py); unused by adamw
    muon_lr: float | None = None  # muon fields use this when set (else lr)
    # lr(step) schedule (tasks/optim.py LRSchedule). None (default) =
    # no scaling; LRSchedule() = constant (debug-consistent); use
    # LRSchedule("wsd", warmup_steps=W, total_steps=T) for training.
    schedule: object = None


@dataclass(frozen=True)
class _Base:
    dims: LlamaDims
    kernels: KernelSet = None  # type: ignore[assignment]

    def _weight_layout(self, layer: int | None = None) -> PackedLayout:
        """Family/kind weight layout; subclasses override. ``layer`` selects
        a depth-dependent dtype sub-policy (None = base policy)."""
        return weight_layout(self.dims, layer=layer)

    @property
    def wl(self) -> PackedLayout:
        return self._weight_layout()

    @property
    def gl(self) -> PackedLayout:
        """dW layout: mirrors wl field-by-field at the policy's grad dtypes
        (identical to wl under the default all-bf16 policy)."""
        return grad_layout(self.wl, self.dims.dtypes)

    @staticmethod
    def layer_of(task) -> int | None:
        """Layer index of a block-scoped task, derived from its W_{i}
        object (inputs for fwd/recompute/bwd, mutates for the optimizer).
        None for loose tasks (embed/head/loss)."""
        for oid in tuple(task.inputs) + tuple(task.mutates):
            if oid.startswith("W_") and oid not in ("W_embed", "W_head"):
                return int(oid.split("_")[1])
        return None

    def wl_for(self, task) -> PackedLayout:
        return self._weight_layout(self.layer_of(task))

    def gl_for(self, task) -> PackedLayout:
        layer = self.layer_of(task)
        return grad_layout(self._weight_layout(layer), self.dims.dtypes,
                           layer=layer,
                           opt_policy=getattr(self.dims, "opt_policy", None))

    @property
    def cl(self) -> PackedLayout:
        return activation_layout(self.dims)

    def _stream_ctx(self, ctx: TaskContext):
        es = external_stream(ctx.stream)
        return es, KernelCtx(stream_handle=es.cuda_stream, torch_stream=es)

    def _in(self, ctx: TaskContext, i: int) -> object:
        return ctx.inputs[ctx.task.inputs[i]]

    def _out(self, ctx: TaskContext, i: int) -> object:
        return ctx.outputs[ctx.task.outputs[i].id]

    # --- shared backward closures ---------------------------------------------
    # Every family backward re-created these two closures verbatim; they are
    # the create-vs-accumulate grad writer and the rmsnorm backward step.

    def _acc_fn(self, dw, accum: bool):
        """create-vs-accumulate grad writer. FROZEN fields simply are
        not in ``dw`` (the grad layout is policy-filtered) and their
        writes SKIP — the layout is the freeze switch; a fully frozen
        pass-through layer has no dW object at all (dw is None) and
        every write skips. The freeze-plan verifier gate and the
        per-family ladders own the no-silent-typo guarantee."""
        if dw is None:
            noop = lambda name, value: None  # noqa: E731
            noop.wanted = lambda name: False
            return noop

        def acc(name: str, value: torch.Tensor) -> None:
            if name not in dw:
                return                      # frozen field: no storage
            if accum:
                dw[name].add_(value.to(dw[name].dtype))
            else:
                dw[name].copy_(value.to(dw[name].dtype))

        # expensive call sites guard the wgrad GEMM itself:
        #   if acc.wanted("wq"): acc("wq", h1.T @ dq)
        # frozen fields then skip the COMPUTATION, not just the write
        acc.wanted = lambda name: name in dw
        return acc

    def _round_of(self, ctx) -> str:
        parts = ctx.task.id.rsplit("_", 3)
        return parts[2] if len(parts) >= 3 else "0"

    def _attn_meta(self, ctx):
        """The round's ``Segments`` — the SINGLE varlen descriptor every
        family's fwd/bwd attention + rope reads (models ALWAYS run varlen).
        Already materialized (``seg.cu`` / ``seg.positions`` device fields,
        ``seg.max_len`` host int) by the engine's run prologue; stages read
        them as attributes, never rebuilding a device tensor mid-round.
        Round parsed from the task id ({s}_{r})."""
        ra = ctx.run_args or {}
        r = self._round_of(ctx)
        segs = ra.get("segments")
        if not segs or r not in segs:
            raise RuntimeError(
                f"packed metadata missing for round {r!r}: every run must "
                f"provide run_args['seq_lens'] (the engine prologue then "
                f"derives run_args['segments'] = {{round: Segments}})")
        return segs[r]

    def _aux_counts_state(self, ctx) -> dict | None:
        """Views of the layer's PERSISTENT Aux object (the per-step +
        all-of-training expert counts): the forward finds it in mutates
        (accumulate), the LAST round's backward in inputs (read the step
        aggregate). None when the family/task has no Aux edge — recompute
        never does, which IS the double-count guard."""
        if getattr(self.dims, "moe", None) is None:
            return None
        from .modules.moe.spec import moe_aux_layout

        layout = moe_aux_layout(self.dims, self.dims.moe)
        for m in ctx.task.mutates:
            if m.startswith("Aux_"):
                return layout.views(ctx.mutates[m])
        for j, oid in enumerate(ctx.task.inputs):
            if oid.startswith("Aux_"):
                return layout.views(self._in(ctx, j))
        return None

    def _aux_temp_state(self, ctx) -> dict | None:
        """Family hook: st entries for METADATA objects (M_{s}_{r}_{i} —
        never-recompute forward artifacts: routing packs, selections).
        None = family has no aux pack. Implementations inspect
        ctx.task.compute_block_key to set aux_temp_ready for recompute."""
        return None

    def _norm_bwd_fn(self, kctx):
        K = self.kernels

        def norm_bwd(dyv, xv, rstd, wv):
            dxv = torch.empty_like(xv)
            dwv = torch.empty(wv.numel(), dtype=torch.float32, device=xv.device)
            K.rmsnorm_bwd(kctx, dyv, xv, rstd, wv, dxv, dwv)
            return dxv, dwv

        return norm_bwd


def _fill_tokens(ex, ctx: TaskContext) -> None:
    d = ex.dims
    for oid, b in ctx.inputs.items():
        if "tokens" in oid:
            g = torch.Generator(device="cuda")
            g.manual_seed(0xE_B_ED ^ d.tokens)
            v = torch_view(b, (d.tokens,), torch.int32)
            v.copy_(torch.randint(0, d.vocab_size, (d.tokens,),
                                  generator=g, dtype=torch.int32,
                                  device=v.device))


@dataclass(frozen=True)
class EmbedFwd(_Base):
    """Token-embedding lookup: tokens + W_embed -> the first hidden
    state."""

    def profile_fill(self, ctx: TaskContext) -> None:
        _fill_tokens(self, ctx)
    def launch(self, ctx: TaskContext) -> None:
        d = self.dims
        with torch.cuda.stream(external_stream(ctx.stream)):
            tokens = torch_view(self._in(ctx, 0), (d.tokens,), torch.int32)
            w = torch_view(self._in(ctx, 1), (d.vocab_size, d.d_model), torch.bfloat16)
            y = torch_view(self._out(ctx, 0), (d.tokens, d.d_model), torch.bfloat16)
            ops.embed_fwd(tokens, w, y)



HEAD_CHUNK_SCRATCH_BYTES = 512 << 20  # per (chunk, vocab) bf16 buffer


def head_chunk_rows(vocab_size: int) -> int:
    """Token-chunk size for the fused head: bounds each internal
    (chunk, vocab) bf16 buffer (logits, dlogits) to
    ~HEAD_CHUNK_SCRATCH_BYTES. Deterministic in the dims, so profiled
    costs are stable."""
    rows = HEAD_CHUNK_SCRATCH_BYTES // (2 * vocab_size)
    return max(256, (rows // 256) * 256)


@dataclass(frozen=True)
class HeadLoss(_Base):
    """Fused final-norm + LM head + CE loss + head backward, micro-chunked
    over tokens (flextrain-inspired). HARD INVARIANT — the memory model's
    point: no (tokens, vocab) tensor is EVER materialized; logits/dlogits
    exist only as (chunk, vocab) scratch inside one loop iteration.

    Buffer contract: inputs (y_last, targets, W_head-or-tied-W_embed
    [, dW accum rounds]); outputs (dy_last, loss [, dW on round 0]).
    Chunking numerics: per-row CE math is exact (rows are independent;
    dlogits normalize by the FULL token count via total_rows); the head
    dW accumulates across chunks in its grad STORAGE dtype — the same
    convention as grad-accum rounds.
    """

    @property
    def hl(self) -> PackedLayout:
        return head_weight_layout(self.dims)

    @property
    def hgl(self) -> PackedLayout:
        return grad_layout(self.hl, self.dims.dtypes, ns="head",
                           opt_policy=getattr(self.dims, "opt_policy", None))

    def launch(self, ctx: TaskContext) -> None:
        d = self.dims
        es, kctx = self._stream_ctx(ctx)
        with torch.cuda.stream(es):
            K = self.kernels
            y = torch_view(self._in(ctx, 0), (d.tokens, d.d_model), torch.bfloat16)
            targets = torch_view(self._in(ctx, 1), (d.tokens,), torch.int32)
            wh = self.hl.views(self._in(ctx, 2))
            accum = bool(ctx.task.mutates)
            dy = torch_view(self._out(ctx, 0), (d.tokens, d.d_model), torch.bfloat16)
            loss = torch_view(self._out(ctx, 1), (1,), torch.float32)
            # frozen head: the dW_head object does not exist (policy-
            # filtered layout scrubbed it) — CE fwd + dy still run,
            # head wgrads skip entirely
            if accum:
                dwh = self.hgl.views(ctx.mutates[ctx.task.mutates[0]])
            elif len(ctx.task.outputs) > 2:
                dwh = self.hgl.views(self._out(ctx, 2))
            else:
                dwh = None
            # packed batches: normalization = VALID rows (pads carry
            # IGNORE_INDEX targets). Host int via run_args — never a
            # device count (hidden-sync rule). ROUND-KEYED like
            # seq_lens (uniform args schema); round derived from the
            # loss output id "loss_{s}_{r}". Absent => all rows
            # (bit-identical legacy).
            ra_h = ctx.run_args or {}
            vr = ra_h.get("valid_rows")
            norm_rows = d.tokens
            if vr is not None:
                if isinstance(vr, dict):
                    _r = (ctx.task.outputs[1].id.rsplit("_", 1)[-1]
                          if len(ctx.task.outputs) > 1 else "0")
                    if _r in vr:
                        norm_rows = int(vr[_r])
                else:
                    norm_rows = int(vr)
            chunk = head_chunk_rows(d.vocab_size)
            loss_acc = torch.zeros(1, dtype=torch.float32, device=y.device)
            part = torch.empty(1, dtype=torch.float32, device=y.device)
            dnorm_acc = torch.zeros(d.d_model, dtype=torch.float32, device=y.device)
            dnorm_c = torch.empty(d.d_model, dtype=torch.float32, device=y.device)
            for lo in range(0, d.tokens, chunk):
                hi = min(lo + chunk, d.tokens)
                y_c = y[lo:hi]
                yn = torch.empty_like(y_c)
                rstd = torch.empty(hi - lo, dtype=torch.float32, device=y.device)
                K.rmsnorm_fwd(kctx, y_c, wh["final_norm_w"], yn, rstd)
                logits = yn @ wh["w"].T                      # (c, V) — chunk scratch
                dlogits = torch.empty_like(logits)
                K.ce_loss_fwd_bwd(
                    kctx, logits, targets[lo:hi], part, dlogits,
                    total_rows=norm_rows,
                )
                loss_acc += part
                if dwh is not None and "w" in dwh:
                    dw_c = dlogits.T @ yn                    # (V, d)
                    if accum or lo > 0:
                        dwh["w"].add_(dw_c.to(dwh["w"].dtype))
                    else:
                        dwh["w"].copy_(dw_c.to(dwh["w"].dtype))
                    del dw_c
                dyn = dlogits @ wh["w"]                      # (c, d)
                del logits, dlogits
                K.rmsnorm_bwd(kctx, dyn, y_c, rstd, wh["final_norm_w"], dy[lo:hi], dnorm_c)
                dnorm_acc += dnorm_c
                # del before the next iteration's allocs: Python rebinding
                # would otherwise keep the previous chunk's buffers live
                # while the new ones allocate (2x peak)
                del yn, rstd, dyn
            if dwh is not None and "final_norm_w" in dwh:
                if accum:
                    dwh["final_norm_w"].add_(
                        dnorm_acc.to(dwh["final_norm_w"].dtype))
                else:
                    dwh["final_norm_w"].copy_(
                        dnorm_acc.to(dwh["final_norm_w"].dtype))
            loss.copy_(loss_acc)


@dataclass(frozen=True)
class EmbedBwd(_Base):
    """Embedding backward: deterministic per-row scatter of dy into
    dW_embed (sorted single-writer, never atomics)."""
    @property
    def egl(self) -> PackedLayout:
        return grad_layout(embed_weight_layout(self.dims), self.dims.dtypes, ns="embed")

    def profile_fill(self, ctx: TaskContext) -> None:
        # The profiler's generic int32 zero-fill sends every token to
        # vocab row 0, collapsing the sorted segment reduction into ONE
        # t-long segment — measured 30 ms vs the ~1 ms real runs see
        # with spread tokens (40x cost bias in the plan). Seed a
        # reproducible uniform token draw instead.
        _fill_tokens(self, ctx)

    def launch(self, ctx: TaskContext) -> None:
        d = self.dims
        es, kctx = self._stream_ctx(ctx)
        with torch.cuda.stream(es):
            dy = torch_view(self._in(ctx, 0), (d.tokens, d.d_model), torch.bfloat16)
            tokens = torch_view(self._in(ctx, 1), (d.tokens,), torch.int32)
            accum = bool(ctx.task.mutates)
            buf = ctx.mutates[ctx.task.mutates[0]] if accum else self._out(ctx, 0)
            dwe = self.egl.view(buf, "w")
            self.kernels.embed_bwd_accum(kctx, tokens, dy, dwe,
                                         zero_first=not accum)


@dataclass(frozen=True)
class GradReduceStep(_Base):
    """Data-parallel gradient exchange as its OWN task: reads the
    local dW input, lands the group SUM in the dWg output. The
    exchange is enqueue-only on the GROUP stream, so it overlaps the
    remaining backward; the compute stream stalls ONLY until the
    staging read of dW completes (keeping dW's lifetime stream-ordered
    while the wire flies). The tail optimizer consumes dWg and merely
    waits. Absent group (sim dry-runs / standalone engines) it
    degrades to a plain copy — the correct world-1 semantics."""

    hyper: object = None       # ctor parity with optimizer kinds; unused
    kind: str = "block"

    def grad_layout_of(self, task):
        d = self.dims
        layer = self.layer_of(task)
        if self.kind == "embed":
            wl_, ns = embed_weight_layout(d), "embed"
        elif self.kind == "head":
            wl_, ns = head_weight_layout(d), "head"
        else:
            wl_, ns = weight_layout(d, layer=layer), None
        return grad_layout(wl_, d.dtypes, ns=ns, layer=layer,
                           opt_policy=getattr(d, "opt_policy", None))

    def launch(self, ctx: TaskContext) -> None:
        es, kctx = self._stream_ctx(ctx)
        with torch.cuda.stream(es):
            g_local = self._in(ctx, 0)
            g_global = self._out(ctx, 0)
            gl_ = self.grad_layout_of(ctx.task)
            if gl_.total_bytes != g_local.size_bytes:
                raise ValueError(
                    f"grad_reduce layout mismatch for {ctx.task.id!r}: "
                    f"layout {gl_.total_bytes} bytes vs dW buffer "
                    f"{g_local.size_bytes} bytes")
            views = []
            dtypes = {f.dtype for f in gl_.fields}
            if len(dtypes) == 1:
                total = 0
                for f in gl_.fields:
                    n = 1
                    for s in f.shape:
                        n *= s
                    dt_f = TORCH_DTYPE_BY_NAME[f.dtype]
                    end = f.offset_bytes + n * dt_f.itemsize
                    if end > total:
                        total = end
                dt_all = TORCH_DTYPE_BY_NAME[gl_.fields[0].dtype]
                count = total // dt_all.itemsize
                views.append((torch_view(g_local, (count,), dt_all),
                              torch_view(g_global, (count,), dt_all)))
            else:
                for f in gl_.fields:
                    dt_f = TORCH_DTYPE_BY_NAME[f.dtype]
                    views.append(
                        (torch_view(g_local, f.shape, dt_f,
                                    offset_bytes=f.offset_bytes),
                         torch_view(g_global, f.shape, dt_f,
                                    offset_bytes=f.offset_bytes)))
            dp_name = ctx.task.block_params.get("dp_group")
            gh = (getattr(ctx, "groups", None) or {}).get(dp_name) \
                if dp_name else None
            if gh is None:
                for lv, gv in views:
                    gv.copy_(lv)
                return
            produced = torch.cuda.Event()
            produced.record(es)
            gh.stream.wait_event(produced)      # producer edge es->gh
            staged = None
            for lv, gv in views:
                staged = gh.allreduce(lv, out=gv)
            if staged is not None:
                es.wait_event(staged)   # es resumes once dW's bytes
                                        # are STAGED (input reusable);
                                        # the exchange itself overlaps


@dataclass(frozen=True)
class AdamWStep(_Base):  # name kept for resolver back-compat; see OptimizerStep alias
    """Per-FIELD optimizer step over one packed weight object —
    dispatches each field's rule (adamw/sgd/sgdm/muon/custom) and
    hyperparameters through the config's optimizer policy
    (tasks/optim.py); AdamW remains the all-fields default.

    Each field updates through its own w/g/m/v views at the dtype policy's
    storage dtypes (fp32 in registers regardless — the kernels are
    dtype-generic); padding gaps are never touched. ``kind`` selects the
    llama-shaped default layout ("block" | "embed" | "head");
    ``layout_for(dims, task, w_size_bytes) -> (PackedLayout, ns)`` overrides
    it for families whose optimizer key spans several layouts (qwen3's own
    block layout, qwen3.5's per-kind blocks / tied embed). The chosen
    layout's total_bytes must equal the W buffer size — a mismatch is a
    loud error, never a silent misview.
    """

    hyper: AdamWHyper = AdamWHyper()
    kind: str = "block"
    layout_for: object = None
    # per-field update overrides: {field_name: fn(kctx, kernels, w_view,
    # g_view)} — the field SKIPS AdamW math entirely (no m/v read, no
    # decay). First customer: DeepSeek-V3's non-gradient router bias,
    # whose dW slot carries expert counts and whose update is the
    # balance sign rule (tasks/moe/stages.moe_bias_update).
    update_specials: object = None

    def _layouts(self, task, w_size: int):
        d = self.dims
        layer = self.layer_of(task)
        if self.layout_for is not None:
            wl_, ns = self.layout_for(d, task, w_size)
        elif self.kind == "embed":
            wl_, ns = embed_weight_layout(d), "embed"
        elif self.kind == "head":
            wl_, ns = head_weight_layout(d), "head"
        else:
            wl_, ns = weight_layout(d, layer=layer), None
        if wl_.total_bytes != w_size:
            raise ValueError(
                f"optimizer layout mismatch for {task.id!r}: layout "
                f"{wl_.total_bytes} bytes vs W buffer {w_size} bytes"
            )
        p = d.dtypes
        op = getattr(d, "opt_policy", None)
        return (wl_, grad_layout(wl_, p, ns=ns, layer=layer, opt_policy=op),
                opt_state_layout(wl_, p, ns=ns, layer=layer, opt_policy=op),
                ns)

    def launch(self, ctx: TaskContext) -> None:
        from dataclasses import replace as dc_replace

        from .optim import OPTIMIZERS, hyper_for, resolve_opt_policy

        es, kctx = self._stream_ctx(ctx)
        with torch.cuda.stream(es):
            w_buf = ctx.mutates[ctx.task.mutates[0]]
            g_buf = ctx.inputs[ctx.task.inputs[1]]
            # a fully-stateless assignment (all-sgd layer) has NO O
            # object — the lowering scrubbed it (apply_exact_sizes)
            o_buf = (ctx.mutates[ctx.task.mutates[1]]
                     if len(ctx.task.mutates) > 1 else None)
            wl_, gl_, ol_, ns = self._layouts(ctx.task, w_buf.size_bytes)
            # data-parallel gradient sum (P4a): the group ROLE name is
            # baked in block_params; the HANDLE arrives per run via
            # ctx.groups. Present => per-field allreduce(dW) on the
            # GROUP stream with event edges both ways (enqueue-only);
            # absent => standalone run of the same artifact (valid
            # exactly because plain-allreduce DP is rank-complete).
            dp_name = ctx.task.block_params.get("dp_group")
            gh = (getattr(ctx, "groups", None) or {}).get(dp_name) \
                if dp_name else None
            if gh is not None:
                grads_ready = torch.cuda.Event()
                grads_ready.record(es)
                gh.stream.wait_event(grads_ready)
                dtypes = {f.dtype for f in gl_.fields}
                if len(dtypes) == 1:
                    # contiguous grad layout, uniform dtype: ONE fused
                    # exchange for the whole layer's grads (wire-floor
                    # sized) instead of a round trip per field —
                    # elementwise sums, so bitwise-identical results
                    total = 0
                    for f in gl_.fields:
                        n = 1
                        for s in f.shape:
                            n *= s
                        dt_f = TORCH_DTYPE_BY_NAME[f.dtype]
                        end = f.offset_bytes + n * dt_f.itemsize
                        if end > total:
                            total = end
                    dt_all = TORCH_DTYPE_BY_NAME[gl_.fields[0].dtype]
                    fused = torch_view(g_buf, (total // dt_all.itemsize,),
                                       dt_all)
                    gh.allreduce(fused)
                else:
                    for f in gl_.fields:
                        gview = torch_view(g_buf, f.shape,
                                           TORCH_DTYPE_BY_NAME[f.dtype],
                                           offset_bytes=f.offset_bytes)
                        gh.allreduce(gview)
                summed = torch.cuda.Event()
                summed.record(gh.stream)
                es.wait_event(summed)
            wait_name = ctx.task.block_params.get("dp_wait_group")
            gw = (getattr(ctx, "groups", None) or {}).get(wait_name) \
                if wait_name else None
            if gw is not None:
                # overlap lowering: grads arrive PRE-REDUCED in the
                # dWg input, summed on the GROUP stream by an earlier
                # grad_reduce task — order es behind its current tail
                # (FIFO stream => covers this layer's sum)
                summed = torch.cuda.Event()
                summed.record(gw.stream)
                es.wait_event(summed)
            ra = getattr(ctx, "run_args", None) or {}
            # service runs bind the global step per run (run_args);
            # in-process paths keep baking it into block_params
            step = int(ra.get("step", ctx.task.block_params.get("step", 0))) + 1
            op = resolve_opt_policy(getattr(self.dims, "opt_policy", None))
            layer = self.layer_of(ctx.task)
            key = (lambda n: f"{ns}.{n}") if ns else (lambda n: n)
            # lr schedule: pure function of the step index; scales lr
            # AND muon_lr, applied AFTER per-field hyper overrides
            sched = self.hyper.schedule
            sched_scale = sched.scale(step) if sched is not None else 1.0
            for f in wl_.fields:
                if self.update_specials is not None and f.name in self.update_specials:
                    # highest-priority per-field override (noaux bias
                    # rule, frozen fields) — skips policy AND state
                    self.update_specials[f.name](
                        kctx, self.kernels,
                        wl_.view(w_buf, f.name).view(-1),
                        gl_.view(g_buf, f.name).view(-1),
                    )
                    continue
                opt = OPTIMIZERS[op.for_field(key(f.name), layer, f.shape)]
                if opt.name == "frozen":
                    continue        # frozen: no grad storage, no update
                hp = hyper_for(op, key(f.name), layer, self.hyper)
                if sched_scale != 1.0:
                    hp = dc_replace(
                        hp, lr=hp.lr * sched_scale,
                        muon_lr=(hp.muon_lr * sched_scale
                                 if hp.muon_lr else hp.muon_lr))
                if opt.slots and o_buf is None:
                    raise ValueError(
                        f"{ctx.task.id}: field {f.name!r} wants "
                        f"{opt.name!r} state but the task has no O object")
                states = {slot: ol_.view(o_buf, f"{slot}_{f.name}").view(-1)
                          for slot in opt.slots}
                opt.step(kctx, self.kernels, hp, step,
                         wl_.view(w_buf, f.name).view(-1),
                         gl_.view(g_buf, f.name).view(-1),
                         states, f.shape)



OptimizerStep = AdamWStep  # the general per-field-policy optimizer executable


