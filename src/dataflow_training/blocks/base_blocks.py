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
from dataflow.runtime.interop import TORCH_DTYPE_BY_NAME, external_stream, torch_view
from ..kernels import KernelCtx, KernelSet, resolve_kernels
from .layouts import (
    LlamaDims,
    PackedLayout,
    activation_layout,
    embed_weight_layout,
    grad_layout,
    head_weight_layout,
    opt_state_layout,
    opt_state_slice_layout,
    sliced_layout,
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
        from dataflow.runtime.interop import torch_view
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

    # saved activations tied to a tensor-parallel weight slice: the
    # dense-MLP convention (x1/x3 are the w1/w3 up-projections and
    # share their d_ff column range)
    TP_ACT_OF_WEIGHT = {"w1": "x1", "w3": "x3"}

    @staticmethod
    def tp_slices(task) -> dict | None:
        """Weight-field shard slices from the task's tensor-parallel
        block param: {field: (dim, lo, hi)}, or None when the task is
        not tensor-parallel."""
        tp = task.block_params.get("tp_slices")
        if tp is None:
            return None
        return {name: tuple(int(x) for x in sl)
                for name, sl in tp.items()}

    def wl_for(self, task) -> PackedLayout:
        wl = self._weight_layout(self.layer_of(task))
        slices = self.tp_slices(task)
        return sliced_layout(wl, slices) if slices else wl

    def gl_for(self, task) -> PackedLayout:
        layer = self.layer_of(task)
        return grad_layout(self.wl_for(task), self.dims.dtypes,
                           layer=layer,
                           opt_policy=getattr(self.dims, "opt_policy", None))

    @property
    def cl(self) -> PackedLayout:
        return activation_layout(self.dims)

    def cl_for(self, task) -> PackedLayout:
        """The task's saved-context layout: under a tensor-parallel
        block param the activations tied to sharded weights narrow to
        the same slice."""
        slices = self.tp_slices(task)
        if not slices:
            return self.cl
        act = {}
        for weight_field, act_field in self.TP_ACT_OF_WEIGHT.items():
            if weight_field in slices:
                act[act_field] = slices[weight_field]
        return sliced_layout(self.cl, act) if act else self.cl

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
        slices = self.tp_slices(task)
        if slices:
            wl_ = sliced_layout(wl_, slices)   # per-rank shard shapes
        if wl_.total_bytes != w_size:
            raise ValueError(
                f"optimizer layout mismatch for {task.id!r}: layout "
                f"{wl_.total_bytes} bytes vs W buffer {w_size} bytes"
            )
        p = d.dtypes
        op = getattr(d, "opt_policy", None)
        # sharded optimizer state: the task's shard block_param narrows
        # O to the regions THIS RANK updates — the same update_regions
        # call the lowering sized the O object with (exact match by
        # construction, so the buffer/layout equality below still holds)
        sh = task.block_params.get("shard")
        if sh is not None and sh.get("mode") == "rs":
            # byte-equal shard: flat slice+tail state, sized by the
            # SAME parameters the lowering used
            ol_ = opt_state_slice_layout(int(sh["n_slice"]),
                                         int(sh["n_tail"]),
                                         sh["opt_dtype"])
            return (wl_,
                    grad_layout(wl_, p, ns=ns, layer=layer,
                                opt_policy=op),
                    ol_, ns)
        regions = None
        if sh is not None:
            regions = {name: (tuple(rows) if rows else None)
                       for name, rows in sh["update"].items()}
        return (wl_, grad_layout(wl_, p, ns=ns, layer=layer, opt_policy=op),
                opt_state_layout(wl_, p, ns=ns, layer=layer, opt_policy=op,
                                 update_regions=regions),
                ns)

    def launch(self, ctx: TaskContext) -> None:
        from .adamw import dp, rs, shards

        es, kctx = self._stream_ctx(ctx)
        with torch.cuda.stream(es):
            w_buf = ctx.mutates[ctx.task.mutates[0]]
            g_buf = ctx.inputs[ctx.task.inputs[1]]
            # a fully-stateless assignment (all-sgd layer) has NO O
            # object — the lowering scrubbed it (apply_exact_sizes)
            o_buf = (ctx.mutates[ctx.task.mutates[1]]
                     if len(ctx.task.mutates) > 1 else None)
            wl_, gl_, ol_, ns = self._layouts(ctx.task, w_buf.size_bytes)
            # comm participation: task.comm_groups maps a purpose to
            # a peer-group NAME (pure data, set by the conductor's
            # lowering); the HANDLES arrive per run via ctx.groups. An
            # absent handle means a standalone run of the same artifact
            # — the fleet warm-up path — and each variant degrades to
            # local-only work.
            groups = getattr(ctx, "groups", None) or {}
            dp_name = ctx.task.comm_groups.get("dp")
            gh = groups.get(dp_name) if dp_name else None
            sh = ctx.task.block_params.get("shard")
            if sh is not None and self.update_specials:
                raise NotImplementedError(
                    f"{ctx.task.id}: sharded optimizer + update_specials "
                    f"— the special's inputs are rank-local (MoE counts) "
                    f"and would diverge across replicas")
            if sh is not None and sh.get("mode") == "rs":
                rs.launch(self, ctx, es, kctx, wl_, gl_, ol_, ns,
                          w_buf, g_buf, o_buf, gh, sh)
            elif sh is not None:
                shards.launch(self, ctx, es, kctx, wl_, gl_, ol_, ns,
                              w_buf, g_buf, o_buf, gh, sh)
            else:
                dp.launch(self, ctx, es, kctx, wl_, gl_, ol_, ns,
                          w_buf, g_buf, o_buf, gh)


OptimizerStep = AdamWStep  # the general per-field-policy optimizer executable


