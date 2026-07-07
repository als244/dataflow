"""Isolated autograd RL trainer: the parity witness.

Plain torch + autograd end to end — no engine, no planner, no custom
kernels. Rebuilds the model from the SAME packed weight bytes, replays
the SAME rollout with selections and routing PINNED to the saved
payloads, computes the SAME RL objective (rl_ops.rl_loss_reference —
the where-form contract), backprops with autograd, and applies the
golden AdamW replica (storage-dtype round-trips included). After each
step it snapshots every parameter field; run.py compares these against
the engine-side buffers field by field.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from dataflow.tasks import ops
from dataflow.training.families import resolve_family
from dataflow.training.glm52 import ShapedGlm52Config

from pinned_golden import RlGlm52
from rl_ops import rl_loss_reference


def train(artifacts: dict, *, steps: int = 3, mode: str = "ppo"):
    cfg = replace(ShapedGlm52Config.tiny(), **artifacts["cfg_overrides"])
    dims = resolve_family(cfg).dims_of(cfg)

    wb = artifacts["w_bytes"]
    golden = RlGlm52.from_packed_bytes(
        dims, cfg.n_layers, wb["W_embed"].cuda(),
        [wb[f"W_{i}"].cuda() for i in range(cfg.n_layers)],
        wb["W_head"].cuda(),
    )
    golden.saved = {"sel": artifacts["sel"],
                    "route_ids": artifacts["route_ids"]}

    tokens = artifacts["tokens"].long().cuda()
    actions = artifacts["actions"].long().cuda()
    old_lp = artifacts["old_logprobs"].cuda()
    adv = artifacts["advantages"].cuda()

    losses, snapshots = [], []
    x_ckpt = [c.cuda() for c in artifacts["x_ckpt"]]
    y_last = artifacts["y_last"].cuda()
    for _ in range(steps):
        golden._pending_counts = []
        for p in golden.parameters():
            p.grad = None

        # ---- the engine's EXACT semantics: no fresh forward pass.
        # Each layer forwards from its FIXED checkpointed input with the
        # CURRENT weights; the head reads the FIXED last-block output;
        # gradients chain across layers through explicit VJPs. (A live
        # re-forward would let step s>1 weight updates ripple through
        # activations — a different, on-policy-ish trainer.)
        y_hold = y_last.clone().requires_grad_(True)
        logits = (ops.rmsnorm_reference(y_hold, golden.w_head["final_norm_w"])
                  @ golden.w_head["w"].T)
        rl = rl_loss_reference(logits, actions, old_lp, adv,
                               dims.tokens, mode)
        rl.backward()
        dy = y_hold.grad

        counts_of = {}
        for i in range(cfg.n_layers - 1, -1, -1):
            golden._layer_ptr = i
            if dims.role_of(i) != "full":
                # pinned mask comes from the layer's LEADER
                lead = dims.leader_of(i)
                golden._group_mask = None
                golden._layer_ptr = lead
                # rebuild leader mask only (cheap): pinned sel
                from dataflow.tasks.dsa_reference import dsa_mask_from_idx

                golden._group_mask = dsa_mask_from_idx(
                    golden.saved["sel"][lead].to(dy.device), dims,
                    dims.tokens)
                golden._layer_ptr = i
            x_i = x_ckpt[i].clone().requires_grad_(True)
            n_before = len(golden._pending_counts)
            y_i, aux_i = golden.block_forward(x_i, golden.w_blocks[i])
            if len(golden._pending_counts) > n_before:
                counts_of[i] = golden._pending_counts[-1]
            ((y_i.float() * dy.float()).sum() + aux_i).backward()
            dy = x_i.grad
        # embed: dW_embed[token] += dy rows (grad in storage dtype,
        # accumulated in fp32 first — the golden AdamW convention)
        g32 = torch.zeros(golden.w_embed["w"].shape,
                          dtype=torch.float32, device=dy.device)
        g32.index_add_(0, tokens, dy.float())
        golden.w_embed["w"].grad = g32.to(golden.w_embed["w"].dtype)

        golden.step_count += 1
        golden._adamw_obj("embed", golden.w_embed)
        speed = dims.moe.bias_update_speed
        for i, leaves in enumerate(golden.w_blocks):
            golden._adamw_obj(f"block_{i}", leaves)
            if "w_router_bias" in leaves and speed and i in counts_of:
                c = counts_of[i]
                b = leaves["w_router_bias"]
                b.data.add_(torch.sign(c.mean() - c).to(b.dtype), alpha=speed)
        golden._adamw_obj("head", golden.w_head)

        losses.append(float(rl.detach()))
        snapshots.append({
            "W_embed": {k: v.detach().clone() for k, v in golden.w_embed.items()},
            **{f"W_{i}": {k: v.detach().clone() for k, v in leaves.items()}
               for i, leaves in enumerate(golden.w_blocks)},
            "W_head": {k: v.detach().clone() for k, v in golden.w_head.items()},
        })
    return losses, snapshots, golden
