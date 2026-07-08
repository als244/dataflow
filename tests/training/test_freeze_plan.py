"""FreezePlan: spec composer, derivation, surgery, and E2E parity.

The freeze feature's contract (docs/notes/handling_frozen_plan.md):
frozen params are SPECIFIED once (the optimizer policy — the freeze()
composer is its front door) and every structural consequence DERIVES:
dW/O shrink to trainable fields, fully-frozen layers lose their
backward tasks when nothing below trains (truncation) or keep a
dgrad-only pass-through (guards-first), A saves only where a backward
will read it, and the dy chain stops at the deepest trainable depth.

E2E gates run the REAL engine against the policy-dispatched golden —
the golden freezes the same params through the same policy, so
check_model_step certifies loss + trainable-param parity while frozen
params must remain bit-identical on both sides by construction.
"""
from __future__ import annotations

import dataclasses

import pytest
import torch  # noqa: F401

from dataflow.tasks.optim import freeze
from dataflow.training.freeze_plan import FreezePlan, derive_freeze_plan
from dataflow.training.models.llama3 import (
    ShapedLlamaConfig,
    dims_of,
    lower_llama3,
)
from dataflow.training.testing.gradcheck import check_model_step

FIELDS = ("attn_norm_w", "wq", "wk", "wv", "wo", "ffn_norm_w",
          "w1", "w3", "w2")


def _plan(cfg):
    d = dims_of(cfg)
    return derive_freeze_plan(
        d, cfg.n_layers, lambda i: FIELDS,
        tied_embeddings=bool(getattr(cfg, "tied_embeddings", False)))


def _tiny(**over):
    return dataclasses.replace(ShapedLlamaConfig.tiny(), **over)


# ---------------------------------------------------------------- analyzer

def test_no_freeze_derives_none():
    assert _plan(_tiny()) is None
    # partial freezes are structurally invisible too: dW shrinks via
    # layouts, no surgery — the byte-identity fast path
    assert _plan(_tiny(opt_policy=freeze(fields=("wq",)))) is None
    assert _plan(_tiny(opt_policy=freeze(pairs=(("wo", 1),)))) is None


def test_truncated_prefix_plan():
    plan = _plan(_tiny(opt_policy=freeze(layers=(0,), embed=True)))
    assert plan.regimes == ("truncated", "train")
    assert plan.emit_bwd == (False, True)
    assert plan.produce_dy == (False, False)   # nothing below layer 1 trains
    assert plan.recv_dy == (False, True)
    assert plan.save_ctx == (False, True)
    assert not plan.embed_trainable and plan.head_trainable


def test_passthrough_plan():
    plan = _plan(_tiny(opt_policy=freeze(layers=(0,))))
    assert plan.regimes == ("passthrough", "train")
    assert plan.emit_bwd == (True, True)       # dgrads must reach embed
    assert plan.produce_dy == (True, True)   # embed trains below layer 1
    assert plan.save_ctx == (True, True)
    assert plan.embed_trainable


def test_all_frozen_ce_rejected():
    with pytest.raises(ValueError, match="nothing"):
        FreezePlan(n_layers=1, regimes=("truncated",), emit_bwd=(False,),
                   recv_dy=(False,), produce_dy=(False,),
                   save_ctx=(False,), embed_trainable=False,
                   head_trainable=False)


def test_composer_semantics():
    pol = freeze(base="muon", layers=(0,), fields=("wo",),
                 pairs=(("w1", 1),), embed=True)
    assert pol.for_field("wq", 0, (8, 8)) == "frozen"      # layer freeze
    assert pol.for_field("wo", 1, (8, 8)) == "frozen"      # field freeze
    assert pol.for_field("w1", 1, (8, 8)) == "frozen"      # pair freeze
    assert pol.for_field("embed.w", None, (16, 8)) == "frozen"
    assert pol.for_field("wq", 1, (8, 8)) == "muon"        # base rules
    assert pol.for_field("attn_norm_w", 1, (8,)) == "adamw"


def test_plan_repr_compact():
    plan = _plan(_tiny(opt_policy=freeze(layers=(0,), embed=True)))
    r = repr(plan)
    assert "truncated=0" in r and "train=1" in r and "obj=ce" in r


# ---------------------------------------------------------------- E2E (GPU)

_CAP = 64 * 1024 * 1024


def test_model_step_truncated_prefix():
    """Layer 0 + embedding frozen: layer 0 has NO backward, NO A, NO
    dW/O; no dy below the head->layer-1 edge; no embed_bwd. Engine
    matches the policy-dispatched golden."""
    cfg = _tiny(opt_policy=freeze(layers=(0,), embed=True))
    prog = lower_llama3(cfg)
    ids = set(prog.task_by_id())
    sizes = prog.object_sizes()
    assert "block_bwd_0_0_0" not in ids
    assert "A_0_0_0" not in sizes and "dW_0_0" not in sizes
    # the boundary backward keeps its dy output (positional contract);
    # it is consumer-less and disposable. dy_embed's PRODUCER (layer 0's
    # backward) is gone, so it does not exist at all.
    assert "dy_0_0_0" in sizes and "dy_embed_0_0" not in sizes
    assert not any("dy_0_0_0" in tt.inputs
                   for tt in prog.task_by_id().values())
    assert not any(t.startswith("embed_bwd") for t in ids)
    check_model_step(cfg, fast_memory_capacity=_CAP, tol=3e-2).assert_ok()


def test_model_step_passthrough():
    """Layer 0 frozen, embedding trainable: layer 0's backward runs
    dgrad-only (dw=None tolerated; wgrads skip via the freeze-aware
    acc), dy reaches embed_bwd, dW_embed exists."""
    cfg = _tiny(opt_policy=freeze(layers=(0,)))
    prog = lower_llama3(cfg)
    sizes = prog.object_sizes()
    assert "dW_0_0" not in sizes and "dW_embed_0" in sizes
    check_model_step(cfg, fast_memory_capacity=_CAP, tol=3e-2).assert_ok()


def test_model_step_partial_fields():
    """wq/wk frozen fleet-wide: no surgery (plan None), dW/O shrink to
    the trainable fields, the frozen fields' wgrads never compute
    (absent from dw -> acc skips), and every trainable field still
    matches the golden."""
    cfg = _tiny(opt_policy=freeze(fields=("wq", "wk")))
    check_model_step(cfg, fast_memory_capacity=_CAP, tol=3e-2).assert_ok()


def test_model_step_truncated_ga2():
    """Grad accumulation across the truncated program: create/accumulate
    rounds on the shrunken dW set."""
    cfg = _tiny(grad_accum_rounds=2,
                opt_policy=freeze(layers=(0,), embed=True))
    check_model_step(cfg, fast_memory_capacity=_CAP, tol=3e-2).assert_ok()
