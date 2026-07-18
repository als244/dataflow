"""llama3 adapter: the SIMPLEST case — dense model, no discrete state,
no M objects. The pinned forward IS the plain golden forward (fixed
inputs are all the pinning a dense block needs)."""
from dataclasses import replace

import torch

from dataflow_training.model_families.llama3 import ShapedLlamaConfig

from golden_base import GoldenLlama3

name = "llama3"
bias_speed = 0.0


def make_cfg():
    return ShapedLlamaConfig.tiny()


def make_golden(dims, n_layers, leaves):
    return GoldenLlama3.from_packed_bytes(dims, n_layers, *leaves)


def capture(golden, tokens):
    captured = {"x": []}
    x = golden.w_embed["w"][tokens]
    for w in golden.w_blocks:
        captured["x"].append(x.detach().clone())
        x = golden.block_forward(x, w)
    return captured, x.detach().clone()


def meta_layout(dims, i):
    return None


def meta_fields(dims, i, captured):
    return None


def pin(golden, captured):
    pass


def prep_layer(golden, i):
    pass


def block(golden, i, x):
    y = golden.block_forward(x, golden.w_blocks[i])
    return y, None, None


def adamw(golden, counts_of):
    golden.step_count += 1
    golden._opt_obj("embed", golden.w_embed)
    for i, leaves in enumerate(golden.w_blocks):
        golden._opt_obj(f"block_{i}", leaves)
    golden._opt_obj("head", golden.w_head)
