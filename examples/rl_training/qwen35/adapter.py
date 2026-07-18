"""qwen35 adapter: hybrid DeltaNet/full-attention kinds, dense MLP —
no discrete state at all, so like llama3 the pinned forward is just
the plain per-kind golden forward from fixed inputs. Shows the harness
handling heterogeneous layer kinds and a standalone golden class."""
import torch

from dataflow.training.models.qwen35 import ShapedQwen35Config

from golden_base import GoldenQwen35

name = "qwen35"


def make_cfg():
    return ShapedQwen35Config.tiny()


def make_golden(dims, n_layers, leaves):
    return GoldenQwen35.from_packed_bytes(dims, n_layers, *leaves)


def _fwd(golden, i, x):
    w = golden.w_blocks[i]
    if golden.dims.kinds[i] == "lin":
        return golden.lin_block_forward(x, w)
    return golden.full_block_forward(x, w)


def capture(golden, tokens):
    captured = {"x": []}
    x = golden.w_embed["w"][tokens.long()]
    for i in range(len(golden.w_blocks)):
        captured["x"].append(x.detach().clone())
        x = _fwd(golden, i, x)
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
    return _fwd(golden, i, x), None, None


def adamw(golden, counts_of):
    golden.step_count += 1
    golden._opt_obj("embed", "head" if golden.tied else "embed",
                      golden.w_embed)
    if golden.w_head is not None:
        golden._opt_obj("head", "head", golden.w_head)
    for i, leaves in enumerate(golden.w_blocks):
        golden._opt_obj(f"block_{i}", None, leaves)
