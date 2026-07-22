"""Kernel registry: mechanics (CPU) + fused-vs-eager-vs-autograd (GPU).

Every fused implementation must match the eager ops.py form at bf16 output
tolerance (identical fp32 math modulo transcendental/FMA ulps) and the
autograd gradient of the shared reference. Odd sizes exercise masking.

Tests:
- test_registry_selection_override_and_gating: kernel resolution picks the highest-priority available impl, an override beats priority, a capability gate enables an impl and flips all_deterministic, and a missing override or duplicate registration raises.
- test_builtin_ops_all_have_eager_fallback: every builtin op registers an eager impl that requires neither CUDA nor triton.
- test_fused_set_is_fully_fused_and_deterministic: the triton-overridden kernel set resolves to triton for each fused op and reports all-deterministic.
- test_swiglu_fused: fused swiglu forward and backward match the eager op and the autograd reference gradient.
- test_rope_fused: fused rope forward and backward match eager and autograd across single, uniform-pair, and ragged packings.
- test_rmsnorm_fused: fused rmsnorm forward, apply, noweight, and backward match the eager ops, the rstd, and the autograd reference.
- test_ce_fused: fused cross-entropy loss and dlogits match the eager op and the autograd reference.
- test_ce_fused_past_int32_elements: cross-entropy over more than 2^31 elements stays finite and rows past the int32 boundary match a per-row fp32 reference (int64 row-offset regression).
- test_adamw_fused: the fused AdamW step matches the eager step for weights and moments across small and large sizes.
- test_fused_steady_state_no_torch_allocation: a warmed-up steady-state launch of the fused swiglu/rope/rmsnorm/adamw ops triggers no torch allocator growth.
"""
import pytest

from dataflow_training.kernels import registry as reg

# --- registry mechanics (no device needed) -----------------------------------


def test_registry_selection_override_and_gating():
    ns = "test_sel_op"
    reg.register(ns, "a", deterministic=True, workspace=reg.none(),
                 priority=1, fn=lambda kctx: "a")
    reg.register(ns, "b", deterministic=True, workspace=reg.none(),
                 priority=5, fn=lambda kctx: "b")
    reg.register(ns, "gated", deterministic=False, workspace=reg.none(),
                 priority=99, requires=lambda caps: caps.get("magic", False),
                 fn=lambda kctx: "gated")

    caps = {"cuda": False, "triton": False}
    kset = reg.resolve_kernels(caps)
    assert kset.entry(ns).impl_id == "b"          # priority among available
    assert kset.describe()[ns] == "b"

    kset = reg.resolve_kernels(caps, overrides={ns: "a"})
    assert kset.entry(ns).impl_id == "a"          # override beats priority

    kset = reg.resolve_kernels({"magic": True})
    assert kset.entry(ns).impl_id == "gated"      # gate opens -> top priority
    assert not kset.all_deterministic()           # gated is nondeterministic

    with pytest.raises(KeyError):
        reg.resolve_kernels(caps, overrides={ns: "missing"})
    with pytest.raises(ValueError):
        reg.register(ns, "a", deterministic=True, workspace=reg.none(),
                     fn=lambda kctx: "dup")


def test_builtin_ops_all_have_eager_fallback():
    for op in ("swiglu_fwd_out", "swiglu_bwd", "rope_fwd", "rope_bwd",
               "rmsnorm_fwd", "rmsnorm_apply", "rmsnorm_noweight",
               "rmsnorm_bwd", "ce_loss_fwd_bwd", "adamw_step"):
        impls = reg.registered(op)
        assert "eager" in impls, op
        assert impls["eager"].requires({"cuda": False, "triton": False})


# --- fused vs eager vs autograd (GPU) -----------------------------------------

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no GPU", allow_module_level=True)
pytest.importorskip("triton")

from dataflow_training.blocks import ops  # noqa: E402
from dataflow_training.kernels import KernelCtx, resolve_kernels  # noqa: E402
from dataflow_training.testing.gradcheck import rel_l2  # noqa: E402

pytestmark = pytest.mark.gpu

KCTX = KernelCtx()
# explicit overrides: these tests target the triton impls even when the
# environment forces DATAFLOW_KERNELS=eager for the rest of the suite
FUSED = resolve_kernels(
    {"cuda": True, "triton": True},
    overrides={op: "triton" for op in (
        "swiglu_fwd_out", "swiglu_bwd", "rope_fwd", "rope_bwd",
        "rmsnorm_fwd", "rmsnorm_apply", "rmsnorm_noweight", "rmsnorm_bwd",
        "ce_loss_fwd_bwd", "adamw_step",
    )},
)


def _rand(*shape, dtype=torch.bfloat16, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(*shape, device="cuda", generator=gen, dtype=dtype)


def test_fused_set_is_fully_fused_and_deterministic():
    desc = FUSED.describe()
    for op in ("swiglu_bwd", "rope_fwd", "rope_bwd", "rmsnorm_fwd",
               "rmsnorm_bwd", "ce_loss_fwd_bwd", "adamw_step"):
        assert desc[op] == "triton", desc
    assert FUSED.all_deterministic()


@pytest.mark.parametrize("rows,dff", [(129, 517), (2048, 1024), (4099, 14336)])
def test_swiglu_fused(rows, dff):
    x1, x3, ds = _rand(rows, dff, seed=1), _rand(rows, dff, seed=2), _rand(rows, dff, seed=3)
    e_out = torch.empty_like(x1)
    ops.swiglu_fwd_out(x1, x3, e_out)
    f_out = torch.empty_like(x1)
    FUSED.swiglu_fwd_out(KCTX, x1, x3, f_out)
    assert rel_l2(f_out.float(), e_out.float()) < 2e-3

    e_dx1, e_dx3 = ops.swiglu_bwd(ds, x1, x3)
    f_dx1, f_dx3 = torch.empty_like(x1), torch.empty_like(x3)
    FUSED.swiglu_bwd(KCTX, ds, x1, x3, f_dx1, f_dx3)
    assert rel_l2(f_dx1.float(), e_dx1.float()) < 2e-3
    assert rel_l2(f_dx3.float(), e_dx3.float()) < 2e-3

    a1 = x1.detach().clone().requires_grad_(True)
    a3 = x3.detach().clone().requires_grad_(True)
    ops.swiglu_fwd(a1, a3).backward(ds)
    assert rel_l2(f_dx1.float(), a1.grad.float()) < 3e-2
    assert rel_l2(f_dx3.float(), a3.grad.float()) < 3e-2


@pytest.mark.parametrize("seq,heads,hd", [
    (128, 8, 64),                # single sequence
    ((64, 64), 8, 64),           # uniform pair (tuple spec)
    ((96, 96), 6, 128),
    ((80, 29, 19), 8, 64),       # RAGGED packing: positions reset per seq
])
def test_rope_fused(seq, heads, hd):
    t = seq if isinstance(seq, int) else sum(seq)
    x = _rand(t, heads * hd, seed=4)
    dy = _rand(t, heads * hd, seed=5)
    base = 500000.0
    seg = (ops.Segments.uniform(seq, t // seq) if isinstance(seq, int)
           else ops.Segments(tuple(seq)))
    pos = seg.on(x.device).positions

    e_fwd = ops.rope_fwd(x, pos, heads, hd, base)
    f_fwd = torch.empty_like(x)
    FUSED.rope_fwd(KCTX, x, f_fwd, pos, heads, hd, base)
    assert rel_l2(f_fwd.float(), e_fwd.float()) < 2e-3

    e_bwd = ops.rope_bwd(dy, pos, heads, hd, base)
    f_bwd = torch.empty_like(dy)
    FUSED.rope_bwd(KCTX, dy, f_bwd, pos, heads, hd, base)
    assert rel_l2(f_bwd.float(), e_bwd.float()) < 2e-3

    ax = x.detach().clone().requires_grad_(True)
    ops.rope_fwd(ax, pos, heads, hd, base).backward(dy)
    assert rel_l2(f_bwd.float(), ax.grad.float()) < 3e-2


@pytest.mark.parametrize("rows,d", [(129, 517), (2048, 4096), (4099, 1024)])
def test_rmsnorm_fused(rows, d):
    x, w, dy = _rand(rows, d, seed=6), _rand(d, seed=7), _rand(rows, d, seed=8)

    e_out = torch.empty_like(x)
    e_rstd = torch.empty(rows, device="cuda", dtype=torch.float32)
    ops.rmsnorm_fwd(x, w, e_out, e_rstd)
    f_out = torch.empty_like(x)
    f_rstd = torch.empty_like(e_rstd)
    FUSED.rmsnorm_fwd(KCTX, x, w, f_out, f_rstd)
    assert rel_l2(f_out.float(), e_out.float()) < 2e-3
    assert rel_l2(f_rstd, e_rstd) < 1e-5

    f_apply = torch.empty_like(x)
    FUSED.rmsnorm_apply(KCTX, x, f_rstd, w, f_apply)
    assert rel_l2(f_apply.float(), e_out.float()) < 2e-3

    e_nw, e_nw_rstd = ops.rmsnorm_noweight(x)
    f_nw = torch.empty_like(x)
    f_nw_rstd = torch.empty_like(e_nw_rstd)
    FUSED.rmsnorm_noweight(KCTX, x, f_nw, f_nw_rstd)
    assert rel_l2(f_nw.float(), e_nw.float()) < 2e-3

    e_dx, e_dw = ops.rmsnorm_bwd(dy, x, e_rstd, w)
    f_dx = torch.empty_like(x)
    f_dw = torch.empty(d, device="cuda", dtype=torch.float32)
    FUSED.rmsnorm_bwd(KCTX, dy, x, f_rstd, w, f_dx, f_dw)
    assert rel_l2(f_dx.float(), e_dx.float()) < 2e-3
    assert rel_l2(f_dw.float(), e_dw.float()) < 2e-3

    ax = x.detach().clone().float().requires_grad_(True)
    aw = w.detach().clone().float().requires_grad_(True)
    ops.rmsnorm_reference(ax, aw).backward(dy.float())
    assert rel_l2(f_dx.float(), ax.grad) < 3e-2
    assert rel_l2(f_dw.float(), aw.grad) < 3e-2


@pytest.mark.parametrize("rows,vocab", [(128, 517), (1024, 32003), (513, 128256)])
def test_ce_fused(rows, vocab):
    logits = _rand(rows, vocab, seed=9)
    gen = torch.Generator(device="cuda").manual_seed(10)
    targets = torch.randint(0, vocab, (rows,), device="cuda", generator=gen, dtype=torch.int32)

    e_loss = torch.empty(1, device="cuda", dtype=torch.float32)
    e_dl = torch.empty_like(logits)
    ops.ce_loss_fwd_bwd(logits, targets, e_loss, e_dl)
    f_loss = torch.empty_like(e_loss)
    f_dl = torch.empty_like(logits)
    FUSED.ce_loss_fwd_bwd(KCTX, logits, targets, f_loss, f_dl)
    assert abs(f_loss.item() - e_loss.item()) / e_loss.item() < 1e-4
    assert rel_l2(f_dl.float(), e_dl.float()) < 2e-3

    al = logits.detach().clone().float().requires_grad_(True)
    ops.ce_loss_reference(al, targets).backward()
    assert rel_l2(f_dl.float(), al.grad) < 3e-2


def test_ce_fused_past_int32_elements():
    """rows x vocab > 2^31: the row offset must be computed in int64.

    Regression for the bs16 crash — qwen3.5's 248,320 vocab overflows
    int32 pointer math at row 8,650 (illegal memory access pre-fix). Needs
    ~8.7 GB free VRAM for logits + dlogits; verifies rows past the boundary
    against a per-row fp32 reference.
    """
    rows, vocab = 8704, 248320
    assert rows * vocab > 2**31
    free, _total = torch.cuda.mem_get_info()
    if free < 10 * 2**30:
        pytest.skip("needs ~10 GB free VRAM")
    logits = _rand(rows, vocab, seed=21)
    gen = torch.Generator(device="cuda").manual_seed(22)
    targets = torch.randint(0, vocab, (rows,), device="cuda", generator=gen,
                            dtype=torch.int32)
    loss = torch.empty(1, device="cuda", dtype=torch.float32)
    dl = torch.empty_like(logits)
    FUSED.ce_loss_fwd_bwd(KCTX, logits, targets, loss, dl)
    assert torch.isfinite(loss).all()
    # rows beyond the int32 line: compare against per-row reference math
    for r in (8650, rows - 1):
        row = logits[r].float()
        soft = torch.softmax(row, dim=-1)
        soft[targets[r].long()] -= 1.0
        assert rel_l2(dl[r].float(), soft / rows) < 2e-3
    del logits, dl
    torch.cuda.empty_cache()


@pytest.mark.parametrize("n", [129, 1 << 20, (1 << 24) + 3])
def test_adamw_fused(n):
    def fresh(seed):
        return (_rand(n, seed=seed), _rand(n, seed=seed + 1),
                _rand(n, seed=seed + 2) * 0.01, _rand(n, seed=seed + 3).abs() * 0.01)

    hyper = dict(lr=3e-4, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.1, step=3)
    ew, eg, em, ev = fresh(11)
    ops.adamw_step(ew, eg, em, ev, **hyper)
    fw, fg, fm, fv = fresh(11)
    FUSED.adamw_step(KCTX, fw, fg, fm, fv, **hyper)
    # same fp32 structure; FMA contraction can flip single bf16 ulps, and at
    # tiny n one flip dominates the norm (real math errors are orders larger)
    assert rel_l2(fw.float(), ew.float()) < 5e-3
    assert rel_l2(fm.float(), em.float()) < 5e-3
    assert rel_l2(fv.float(), ev.float()) < 5e-3


def test_fused_steady_state_no_torch_allocation():
    """Post-JIT swiglu/rope/rmsnorm-fwd/adamw launches touch no allocator.
    (ce + rmsnorm_bwd declare small internal buffers — cache-served.)"""
    x1, x3 = _rand(512, 1024, seed=20), _rand(512, 1024, seed=21)
    m, v = _rand(512, 1024, seed=22) * 0.01, _rand(512, 1024, seed=23).abs() * 0.01
    out = torch.empty_like(x1)
    rstd = torch.empty(512, device="cuda", dtype=torch.float32)
    # positions materialized ONCE (the Segments discipline) — read as a field
    # in the steady-state loop, so no per-launch device allocation
    pos = ops.Segments.uniform(128, x1.shape[0] // 128).on(x1.device).positions

    def run():
        FUSED.swiglu_fwd_out(KCTX, x1, x3, out)
        FUSED.rope_fwd(KCTX, x1, out, pos, 8, 128, 500000.0)
        FUSED.rmsnorm_fwd(KCTX, x1, x3[0], out, rstd)
        FUSED.adamw_step(KCTX, x1, x3, m, v,
                         lr=1e-4, beta1=0.9, beta2=0.95, eps=1e-8,
                         weight_decay=0.1, step=1)

    run()  # JIT warm-up
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    run()
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
