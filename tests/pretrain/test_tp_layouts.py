"""T1 gates: per-rank tensor-parallel family layouts + init parity.

The tp_view layout transform must (a) materialize MLP weight fields
and their tied saved activations at shard shape while leaving
replicated fields untouched, and (b) keep init CERTIFICATION-GRADE:
shard fields draw at the full single-GPU shape and slice, so every
rank's bytes are exact slices of the single-GPU init and the
generator stream stays aligned across later fields and layers."""
import pytest

torch = pytest.importorskip("torch")

from dataflow_training.distributed.sharding import (
    layer_fields_by_root,
    tp_mlp_shards,
    tp_view,
)
from dataflow_training.lowering.emit import fill_weight_fields
from dataflow_training.model_families.llama3 import (
    ShapedLlamaConfig,
    family_layouts,
    tp_fill_slices,
)

CFG = ShapedLlamaConfig.tiny()
SEED = 7


def build_plan():
    plan = tp_mlp_shards(layer_fields_by_root(CFG), "tp", 2)
    plan.validate()
    plan.consumable("tp")
    return plan


def test_per_rank_layout_shapes_and_sizes():
    plan = build_plan()
    dims, full = family_layouts(CFG)
    _, fl0 = family_layouts(CFG, tp_view=tp_view(plan, 0))
    ffs = CFG.d_ff // 2
    w = {f.name: f.shape for f in fl0.layers[0].weights.fields}
    assert w["w1"] == (CFG.d_model, ffs)
    assert w["w3"] == (CFG.d_model, ffs)
    assert w["w2"] == (ffs, CFG.d_model)
    assert w["wq"] == full.layers[0].weights.field("wq").shape
    a = {f.name: f.shape for f in fl0.layers[0].activations.fields}
    assert a["x1"] == (dims.tokens, ffs)
    assert a["x3"] == (dims.tokens, ffs)
    assert a["q"] == full.layers[0].activations.field("q").shape
    # mlp field bytes halve exactly (whole-field sums, no padding)
    mlp = ("w1", "w3", "w2")
    full_bytes = sum(f.nbytes for f in full.layers[0].weights.fields
                     if f.name in mlp)
    rank_bytes = sum(f.nbytes for f in fl0.layers[0].weights.fields
                     if f.name in mlp)
    assert rank_bytes * 2 == full_bytes
    # embed/head untouched (replicated roots)
    assert fl0.embed.total_bytes == full.embed.total_bytes
    assert fl0.head.total_bytes == full.head.total_bytes


def test_init_parity_shards_are_single_gpu_slices():
    if not torch.cuda.is_available():
        pytest.skip("pinned-host init needs a CUDA context")
    from dataflow.runtime.device.cuda import CudaBackend

    plan = build_plan()
    dims, fl_full = family_layouts(CFG)
    backend = CudaBackend()
    layers = min(2, CFG.n_layers)   # >=2 proves cross-layer stream
    ffs = CFG.d_ff // 2             # alignment through sliced draws

    gen = torch.Generator().manual_seed(SEED)
    full_bufs = []
    for i in range(layers):
        wl = fl_full.layers[i].weights
        buf = backend.alloc("backing", wl.total_bytes)
        fill_weight_fields(buf, wl, gen)
        full_bufs.append(buf)

    rank_layouts = []
    rank_bufs = []
    for rank in (0, 1):
        view = tp_view(plan, rank)
        _, fl_r = family_layouts(CFG, tp_view=view)
        slices = tp_fill_slices(CFG, view)
        gen_r = torch.Generator().manual_seed(SEED)
        bufs = []
        for i in range(layers):
            wl = fl_r.layers[i].weights
            buf = backend.alloc("backing", wl.total_bytes)
            fill_weight_fields(buf, wl, gen_r,
                               tp_slices=slices.get(f"W_{i}"))
            bufs.append(buf)
        rank_layouts.append(fl_r)
        rank_bufs.append(bufs)

    for i in range(layers):
        wl_full = fl_full.layers[i].weights
        for rank in (0, 1):
            wl_r = rank_layouts[rank].layers[i].weights
            buf_r = rank_bufs[rank][i]
            lo, hi = rank * ffs, (rank + 1) * ffs
            # sharded fields == exact single-GPU slices
            assert torch.equal(wl_r.view(buf_r, "w1"),
                               wl_full.view(full_bufs[i], "w1")[:, lo:hi])
            assert torch.equal(wl_r.view(buf_r, "w3"),
                               wl_full.view(full_bufs[i], "w3")[:, lo:hi])
            assert torch.equal(wl_r.view(buf_r, "w2"),
                               wl_full.view(full_bufs[i], "w2")[lo:hi, :])
            # replicated fields == the single-GPU bytes on BOTH ranks
            # (also proves the draw stream stayed aligned after the
            # sliced draws — wq of layer i+1 draws after layer i's mlp)
            assert torch.equal(wl_r.view(buf_r, "wq"),
                               wl_full.view(full_bufs[i], "wq"))
            assert torch.equal(wl_r.view(buf_r, "wo"),
                               wl_full.view(full_bufs[i], "wo"))
