"""RL head loss for the custom-Program example: fused final-norm + LM
head + policy-gradient loss + head backward, micro-chunked over tokens.

Mirrors the builtin ``HeadLoss`` skeleton (same buffer conventions, same
no-(tokens,vocab)-materialization invariant, same dW accumulation dtype
rules) with the CE piece swapped for an RL objective:

- ``ppo``:  L = -mean_i  min(r_i A_i, clamp(r_i, 1-eps, 1+eps) A_i),
  r_i = exp(logprob_i - old_logprob_i), formulated with an explicit
  ``where`` mask (NOT torch.minimum) so the tie/branch derivative is
  unambiguous and the isolated autograd reference (which uses the SAME
  where-form) matches bit-for-branch.
- ``reinforce``: L = -mean_i A_i * logprob_i.

The math lives in ``rl_head_loss_math`` (plain tensors in/out) so it is
pinned against autograd without any engine machinery; the executable is
a thin view-wrapper obeying the task contract (no host syncs, chunk
scratch only).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from dataflow.runtime.interop import torch_view
from dataflow_training.blocks.layouts import PackedLayout, grad_layout, head_weight_layout
from dataflow_training.blocks.base_blocks import _Base, head_chunk_rows

PPO_CLIP_EPS = 0.2


def rl_loss_reference(logits: torch.Tensor, actions: torch.Tensor,
                      old_logprobs: torch.Tensor, advantages: torch.Tensor,
                      total_rows: int, mode: str) -> torch.Tensor:
    """Autograd form on a logits CHUNK; contributes chunk_rows/total_rows
    of the mean. The where-form derivative convention is THE contract —
    reference_trainer.py uses this exact function."""
    lse = torch.logsumexp(logits.float(), dim=-1)
    lp = logits.float().gather(1, actions.long().unsqueeze(1)).squeeze(1) - lse
    a = advantages.float()
    if mode == "reinforce":
        obj = a * lp
    else:
        ratio = torch.exp(lp - old_logprobs.float())
        surr1 = ratio * a
        surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP_EPS, 1.0 + PPO_CLIP_EPS) * a
        obj = torch.where(surr1 <= surr2, surr1, surr2)
    return -obj.sum() / total_rows


def rl_head_loss_math(y, actions, old_lp, adv, w, norm_w, *, mode, K, kctx,
                      total_rows=None, chunk=None):
    """Hand-written fwd+bwd, chunked. Returns (loss, dy, dw, dnorm) —
    the executable and the pin test both call this."""
    t, d_model = y.shape
    vocab = w.shape[0]
    total = total_rows or t
    chunk = chunk or head_chunk_rows(vocab)
    dy = torch.empty_like(y)
    dw = torch.zeros(vocab, d_model, dtype=torch.float32, device=y.device)
    dnorm = torch.zeros(d_model, dtype=torch.float32, device=y.device)
    loss = torch.zeros(1, dtype=torch.float32, device=y.device)
    for lo in range(0, t, chunk):
        hi = min(lo + chunk, t)
        y_c = y[lo:hi]
        yn = torch.empty_like(y_c)
        rstd = torch.empty(hi - lo, dtype=torch.float32, device=y.device)
        K.rmsnorm_fwd(kctx, y_c, norm_w, yn, rstd)
        logits = (yn @ w.T).float()                    # (c, V) chunk scratch
        lse = torch.logsumexp(logits, dim=-1)
        act = actions[lo:hi].long()
        lp = logits.gather(1, act.unsqueeze(1)).squeeze(1) - lse
        a = adv[lo:hi].float()
        if mode == "reinforce":
            obj = a * lp
            glp = -a / total                            # dL/dlp
        else:
            ratio = torch.exp(lp - old_lp[lo:hi].float())
            surr1 = ratio * a
            clip_r = torch.clamp(ratio, 1.0 - PPO_CLIP_EPS, 1.0 + PPO_CLIP_EPS)
            surr2 = clip_r * a
            pick1 = surr1 <= surr2
            obj = torch.where(pick1, surr1, surr2)
            in_band = (ratio > 1.0 - PPO_CLIP_EPS) & (ratio < 1.0 + PPO_CLIP_EPS)
            active = pick1 | in_band                    # where the r-branch grads flow
            glp = torch.where(active, -a * ratio / total,
                              torch.zeros_like(a))
        loss += -obj.sum().reshape(1) / total
        # dlogits = glp * (onehot(action) - softmax)
        soft = torch.exp(logits - lse.unsqueeze(1))
        dlogits = soft * (-glp).unsqueeze(1)
        dlogits.scatter_add_(1, act.unsqueeze(1), glp.unsqueeze(1))
        dlogits = dlogits.to(yn.dtype)
        dw += (dlogits.T @ yn).float()
        dyn = dlogits @ w
        del logits, soft, dlogits
        dnorm_c = torch.empty(d_model, dtype=torch.float32, device=y.device)
        K.rmsnorm_bwd(kctx, dyn, y_c, rstd, norm_w, dy[lo:hi], dnorm_c)
        dnorm += dnorm_c
        del yn, rstd, dyn
    return loss, dy, dw, dnorm


@dataclass(frozen=True)
class RLHeadLoss(_Base):
    """Buffer contract: inputs (y_last, actions i32, old_logprobs f32,
    advantages f32, W_head) ; outputs (dy_last, loss, dW_head)."""

    mode: str = "ppo"

    @property
    def hl(self) -> PackedLayout:
        return head_weight_layout(self.dims)

    @property
    def hgl(self) -> PackedLayout:
        return grad_layout(self.hl, self.dims.dtypes, ns="head")

    def launch(self, ctx) -> None:
        d = self.dims
        es, kctx = self._stream_ctx(ctx)
        with torch.cuda.stream(es):
            K = self.kernels
            y = torch_view(self._in(ctx, 0), (d.max_tokens, d.d_model), torch.bfloat16)
            actions = torch_view(self._in(ctx, 1), (d.max_tokens,), torch.int32)
            old_lp = torch_view(self._in(ctx, 2), (d.max_tokens,), torch.float32)
            adv = torch_view(self._in(ctx, 3), (d.max_tokens,), torch.float32)
            wh = self.hl.views(self._in(ctx, 4))
            dy = torch_view(self._out(ctx, 0), (d.max_tokens, d.d_model), torch.bfloat16)
            loss = torch_view(self._out(ctx, 1), (1,), torch.float32)
            dwh = self.hgl.views(self._out(ctx, 2))
            l, dy_v, dw, dnorm = rl_head_loss_math(
                y, actions, old_lp, adv, wh["w"], wh["final_norm_w"],
                mode=self.mode, K=K, kctx=kctx, total_rows=d.max_tokens)
            dy.copy_(dy_v)
            dwh["w"].copy_(dw.to(dwh["w"].dtype))
            dwh["final_norm_w"].copy_(dnorm.to(dwh["final_norm_w"].dtype))
            loss.copy_(l)
