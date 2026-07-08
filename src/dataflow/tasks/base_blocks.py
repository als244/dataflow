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
from .interop import external_stream, torch_view
from .kernels import KernelCtx, KernelSet, resolve_kernels
from .layouts import (
    LlamaDims,
    PackedLayout,
    context_layout,
    embed_weight_layout,
    grad_layout,
    head_weight_layout,
    opt_state_layout,
    weight_layout,
)



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
        return context_layout(self.dims)

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
            return lambda name, value: None

        def acc(name: str, value: torch.Tensor) -> None:
            if name not in dw:
                return                      # frozen field: no storage
            if accum:
                dw[name].add_(value.to(dw[name].dtype))
            else:
                dw[name].copy_(value.to(dw[name].dtype))

        return acc

    def _meta_state(self, ctx) -> dict | None:
        """Family hook: st entries for METADATA objects (M_{s}_{r}_{i} —
        never-recompute forward artifacts: routing packs, selections).
        None = family has no metadata. Implementations inspect
        ctx.task.compute_block_key to set meta_ready for recompute."""
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
        return grad_layout(self.hl, self.dims.dtypes, ns="head")

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
            dwh = self.hgl.views(
                ctx.mutates[ctx.task.mutates[0]] if accum else self._out(ctx, 2)
            )
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
                    kctx, logits, targets[lo:hi], part, dlogits, total_rows=d.tokens,
                )
                loss_acc += part
                dw_c = dlogits.T @ yn                        # (V, d)
                if accum or lo > 0:
                    dwh["w"].add_(dw_c.to(dwh["w"].dtype))
                else:
                    dwh["w"].copy_(dw_c.to(dwh["w"].dtype))
                dyn = dlogits @ wh["w"]                      # (c, d)
                del logits, dlogits, dw_c
                K.rmsnorm_bwd(kctx, dyn, y_c, rstd, wh["final_norm_w"], dy[lo:hi], dnorm_c)
                dnorm_acc += dnorm_c
                # del before the next iteration's allocs: Python rebinding
                # would otherwise keep the previous chunk's buffers live
                # while the new ones allocate (2x peak)
                del yn, rstd, dyn
            if accum:
                dwh["final_norm_w"].add_(dnorm_acc.to(dwh["final_norm_w"].dtype))
            else:
                dwh["final_norm_w"].copy_(dnorm_acc.to(dwh["final_norm_w"].dtype))
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
            step = int(ctx.task.block_params.get("step", 0)) + 1
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


