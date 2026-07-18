"""Kernel audit battery: write-coverage + degenerate inputs, every op.

Two latent kernel bugs survived every model-level gate for months because
nothing exercised their trigger geometry directly: an online-softmax NaN on
rows whose first live tile comes late, and a workspace consumed where the
kernel early-returned without storing (stale bytes from the PREVIOUS
launch). This battery attacks both classes at the KERNEL REGISTRY level,
op by op:

- WRITE COVERAGE (poison invariance): run the same implementation twice on
  identical inputs with every output buffer pre-filled with two different
  byte poisons (and, best-effort, the torch caching allocator primed with
  the same poison so internal scratch starts dirty — the exact technique
  that diagnosed the stale-workspace bug). Any output byte that differs
  between the runs was READ THROUGH, not written: with slab reuse, that
  byte is another task's garbage in production.
- DEGENERATE INPUTS: per-op edge geometry (odd/unaligned sizes, single
  rows, saturated values, ties, empty groups) checked implementation vs
  implementation (eager is the anchor when present) and for finiteness.
- COMPLETENESS TRIPWIRE: every op in the registry must carry audit cases
  or an explicit reasoned exemption — a new kernel cannot land unaudited.

Cases are plain data (module-level builder functions, no closures): each
returns fresh args from a seeded generator so the two poison runs see
bit-identical inputs.
"""
from dataclasses import dataclass, field
from typing import Callable

import pytest

from dataflow_training.kernels import registry as reg

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

pytestmark = pytest.mark.gpu

KCTX = reg.KernelCtx()
CAPS = reg.device_caps()
POISONS = (0x55, 0xAA)
PRIME_BYTES = 128 << 20   # allocator priming block for internal scratch


def gen(seed: int) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(seed)


def rand(g, *shape, dtype=torch.bfloat16, scale=1.0):
    return torch.randn(*shape, device="cuda", generator=g,
                       dtype=torch.float32).mul(scale).to(dtype)


def fill_poison(t: torch.Tensor, byte: int) -> None:
    t.view(torch.uint8).fill_(byte)


def prime_allocator(byte: int) -> None:
    """Best-effort scratch poisoning: dirty the caching allocator's blocks
    so an implementation's internal torch.empty starts on poisoned bytes.
    Heuristic (bin reuse), but it is the probe that caught the real
    stale-workspace bug."""
    blocks = [torch.empty(PRIME_BYTES, device="cuda", dtype=torch.uint8)]
    for i in range(4):
        blocks.append(torch.empty(PRIME_BYTES >> (2 * i + 2),
                                  device="cuda", dtype=torch.uint8))
    for b in blocks:
        b.fill_(byte)
    del blocks
    torch.cuda.synchronize()


@dataclass
class AuditCase:
    """One audited invocation shape for one op.

    make(generator) -> (args, kwargs): fresh tensors every call, identical
    across calls with equal seeds. outputs = positions in args the kernel
    must FULLY write (poisoned before launch). inout = positions the kernel
    reads AND writes (seeded, never poisoned, still compared). tol = cross-
    implementation tolerance (rel L2) for float outputs; integer outputs
    always compare exact. finite = outputs must contain no NaN/Inf.
    """

    name: str
    make: Callable
    outputs: tuple = ()
    inout: tuple = ()
    tol: float = 5e-2
    finite: bool = True
    # outputs with DOCUMENTED -inf semantics (e.g. index scores above the
    # diagonal): finiteness skips them; cross-impl compares the finiteness
    # PATTERN exactly and values on the finite subset
    inf_ok: bool = False


CASES: dict[str, list[AuditCase]] = {}
# ops with NO audit cases, each with the reason the audit does not apply
EXEMPT: dict[str, str] = {}


def add_cases(op: str, *cases: AuditCase) -> None:
    CASES.setdefault(op, []).extend(cases)


# =========================== runner ==========================================

def available_impls(op: str) -> dict[str, reg.KernelEntry]:
    impls = {}
    for impl_id, entry in reg.registered(op).items():
        if entry.requires(CAPS):
            impls[impl_id] = entry
    return impls


def snapshot(args, positions) -> list:
    out = []
    for i in positions:
        out.append(args[i].detach().clone())
    return out


def run_once(entry: reg.KernelEntry, case: AuditCase, seed: int,
             poison: int) -> list:
    """Build fresh args, poison outputs (+ scratch), launch, snapshot the
    output and inout tensors."""
    args, kwargs = case.make(gen(seed))
    for i in case.outputs:
        fill_poison(args[i], poison)
    if entry.workspace.style == "internal":
        prime_allocator(poison)
    entry.fn(KCTX, *args, **kwargs)
    torch.cuda.synchronize()
    return snapshot(args, tuple(case.outputs) + tuple(case.inout))


def bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    af, bf = a.float(), b.float()
    denom = bf.norm().item()
    if denom == 0.0:
        return af.norm().item()
    return (af - bf).norm().item() / denom


def poison_report(a: torch.Tensor, b: torch.Tensor) -> str:
    diff = a.view(torch.uint8) != b.view(torch.uint8)
    n = int(diff.sum())
    first = int(diff.view(-1).nonzero()[0]) if n else -1
    return f"{n} byte(s) poison-dependent (first at byte {first} of {a.numel() * a.element_size()})"


# ====================== case builders: tranche 1 =============================
# swiglu / rmsnorm / rope / adamw — flat elementwise + rowwise families.

def make_swiglu_fwd_odd(g):
    n = 3 * 1021          # deliberately not a multiple of any block
    return (rand(g, n), rand(g, n), torch.empty(n, device="cuda", dtype=torch.bfloat16)), {}


def make_swiglu_fwd_saturated(g):
    n = 4096
    x1 = rand(g, n, scale=40.0)     # sigmoid fully saturated both sides
    x3 = rand(g, n, scale=40.0)
    return (x1, x3, torch.empty(n, device="cuda", dtype=torch.bfloat16)), {}


def make_swiglu_fwd_single(g):
    return (rand(g, 1), rand(g, 1), torch.empty(1, device="cuda", dtype=torch.bfloat16)), {}


def make_swiglu_bwd_odd(g):
    n = 3 * 1021
    e = torch.empty(n, device="cuda", dtype=torch.bfloat16)
    return (rand(g, n), rand(g, n), rand(g, n), e, torch.empty_like(e)), {}


def make_swiglu_bwd_saturated(g):
    n = 4096
    e = torch.empty(n, device="cuda", dtype=torch.bfloat16)
    return (rand(g, n, scale=8.0), rand(g, n, scale=40.0), rand(g, n, scale=40.0),
            e, torch.empty_like(e)), {}


def make_swiglu_packed_fwd_odd(g):
    rows, f = 37, 511
    h13 = rand(g, rows, 2 * f)
    return (h13, torch.empty(rows, f, device="cuda", dtype=torch.bfloat16)), {}


def make_swiglu_packed_fwd_single(g):
    h13 = rand(g, 1, 2)
    return (h13, torch.empty(1, 1, device="cuda", dtype=torch.bfloat16)), {}


def make_swiglu_packed_bwd_odd(g):
    rows, f = 37, 511
    return (rand(g, rows, f), rand(g, rows, 2 * f),
            torch.empty(rows, 2 * f, device="cuda", dtype=torch.bfloat16)), {}


add_cases("swiglu_fwd_out",
          AuditCase("odd_n", make_swiglu_fwd_odd, outputs=(2,)),
          AuditCase("saturated", make_swiglu_fwd_saturated, outputs=(2,)),
          AuditCase("single", make_swiglu_fwd_single, outputs=(2,)))
add_cases("swiglu_bwd",
          AuditCase("odd_n", make_swiglu_bwd_odd, outputs=(3, 4)),
          AuditCase("saturated", make_swiglu_bwd_saturated, outputs=(3, 4)))
add_cases("swiglu_packed_fwd",
          AuditCase("odd_shape", make_swiglu_packed_fwd_odd, outputs=(1,)),
          AuditCase("single", make_swiglu_packed_fwd_single, outputs=(1,)))
add_cases("swiglu_packed_bwd",
          AuditCase("odd_shape", make_swiglu_packed_bwd_odd, outputs=(2,)))


def make_rmsnorm_fwd_odd(g):
    t, d = 33, 415
    return (rand(g, t, d), rand(g, d),
            torch.empty(t, d, device="cuda", dtype=torch.bfloat16),
            torch.empty(t, device="cuda", dtype=torch.float32)), {}


def make_rmsnorm_fwd_zero_rows(g):
    t, d = 8, 256
    x = rand(g, t, d)
    x[3].zero_()                    # rstd = 1/sqrt(eps): large, finite
    return (x, rand(g, d),
            torch.empty(t, d, device="cuda", dtype=torch.bfloat16),
            torch.empty(t, device="cuda", dtype=torch.float32)), {}


def make_rmsnorm_apply_odd(g):
    t, d = 33, 415
    x = rand(g, t, d)
    rstd = torch.rsqrt(x.float().pow(2).mean(-1) + 1e-5)
    return (x, rstd, rand(g, d),
            torch.empty(t, d, device="cuda", dtype=torch.bfloat16)), {}


def make_rmsnorm_noweight_odd(g):
    t, d = 33, 415
    return (rand(g, t, d),
            torch.empty(t, d, device="cuda", dtype=torch.bfloat16),
            torch.empty(t, device="cuda", dtype=torch.float32)), {}


def make_rmsnorm_bwd_odd(g):
    t, d = 33, 415
    x = rand(g, t, d)
    rstd = torch.rsqrt(x.float().pow(2).mean(-1) + 1e-5)
    return (rand(g, t, d), x, rstd, rand(g, d),
            torch.empty(t, d, device="cuda", dtype=torch.bfloat16),
            torch.empty(d, device="cuda", dtype=torch.bfloat16)), {}


def make_rmsnorm_bwd_single_row(g):
    t, d = 1, 512
    x = rand(g, t, d)
    rstd = torch.rsqrt(x.float().pow(2).mean(-1) + 1e-5)
    return (rand(g, t, d), x, rstd, rand(g, d),
            torch.empty(t, d, device="cuda", dtype=torch.bfloat16),
            torch.empty(d, device="cuda", dtype=torch.bfloat16)), {}


add_cases("rmsnorm_fwd",
          AuditCase("odd_shape", make_rmsnorm_fwd_odd, outputs=(2, 3)),
          AuditCase("zero_row", make_rmsnorm_fwd_zero_rows, outputs=(2, 3)))
add_cases("rmsnorm_apply",
          AuditCase("odd_shape", make_rmsnorm_apply_odd, outputs=(3,)))
add_cases("rmsnorm_noweight",
          AuditCase("odd_shape", make_rmsnorm_noweight_odd, outputs=(1, 2)))
add_cases("rmsnorm_bwd",
          AuditCase("odd_shape", make_rmsnorm_bwd_odd, outputs=(4, 5), tol=8e-2),
          AuditCase("single_row", make_rmsnorm_bwd_single_row, outputs=(4, 5), tol=8e-2))


def _ln_stats(x):
    xf = x.float()
    mean = xf.mean(-1)
    rstd = torch.rsqrt((xf - mean.unsqueeze(-1)).pow(2).mean(-1) + 1e-5)
    return mean, rstd


def make_layernorm_fwd_odd(g):
    t, d = 33, 415
    return (rand(g, t, d), rand(g, d), rand(g, d),
            torch.empty(t, d, device="cuda", dtype=torch.bfloat16),
            torch.empty(t, device="cuda", dtype=torch.float32),
            torch.empty(t, device="cuda", dtype=torch.float32)), {}


def make_layernorm_fwd_constant_row(g):
    t, d = 8, 256
    x = rand(g, t, d)
    x[3].fill_(2.0)                 # var 0: rstd = 1/sqrt(eps), large finite
    return (x, rand(g, d), rand(g, d),
            torch.empty(t, d, device="cuda", dtype=torch.bfloat16),
            torch.empty(t, device="cuda", dtype=torch.float32),
            torch.empty(t, device="cuda", dtype=torch.float32)), {}


def make_layernorm_apply_odd(g):
    t, d = 33, 415
    x = rand(g, t, d)
    mean, rstd = _ln_stats(x)
    return (x, mean, rstd, rand(g, d), rand(g, d),
            torch.empty(t, d, device="cuda", dtype=torch.bfloat16)), {}


def make_layernorm_bwd_odd(g):
    t, d = 33, 415
    x = rand(g, t, d)
    mean, rstd = _ln_stats(x)
    return (rand(g, t, d), x, mean, rstd, rand(g, d),
            torch.empty(t, d, device="cuda", dtype=torch.bfloat16),
            torch.empty(d, device="cuda", dtype=torch.float32),
            torch.empty(d, device="cuda", dtype=torch.float32)), {}


def make_layernorm_bwd_single_row(g):
    t, d = 1, 512
    x = rand(g, t, d)
    mean, rstd = _ln_stats(x)
    return (rand(g, t, d), x, mean, rstd, rand(g, d),
            torch.empty(t, d, device="cuda", dtype=torch.bfloat16),
            torch.empty(d, device="cuda", dtype=torch.float32),
            torch.empty(d, device="cuda", dtype=torch.float32)), {}


add_cases("layernorm_fwd",
          AuditCase("odd_shape", make_layernorm_fwd_odd, outputs=(3, 4, 5)),
          AuditCase("constant_row", make_layernorm_fwd_constant_row,
                    outputs=(3, 4, 5)))
add_cases("layernorm_apply",
          AuditCase("odd_shape", make_layernorm_apply_odd, outputs=(5,)))
add_cases("layernorm_bwd",
          AuditCase("odd_shape", make_layernorm_bwd_odd, outputs=(5, 6, 7),
                    tol=8e-2),
          AuditCase("single_row", make_layernorm_bwd_single_row,
                    outputs=(5, 6, 7), tol=8e-2))


def make_gelu_fwd_odd(g):
    t, f = 33, 415
    return (rand(g, t, f),
            torch.empty(t, f, device="cuda", dtype=torch.bfloat16)), {}


def make_gelu_fwd_saturated(g):
    n = 4096
    return (rand(g, 1, n, scale=40.0),      # tanh fully saturated both sides
            torch.empty(1, n, device="cuda", dtype=torch.bfloat16)), {}


def make_gelu_bwd_odd(g):
    t, f = 33, 415
    return (rand(g, t, f), rand(g, t, f),
            torch.empty(t, f, device="cuda", dtype=torch.bfloat16)), {}


def make_gelu_bwd_saturated(g):
    n = 4096
    return (rand(g, 1, n, scale=8.0), rand(g, 1, n, scale=40.0),
            torch.empty(1, n, device="cuda", dtype=torch.bfloat16)), {}


add_cases("gelu_fwd_out",
          AuditCase("odd_shape", make_gelu_fwd_odd, outputs=(1,)),
          AuditCase("saturated", make_gelu_fwd_saturated, outputs=(1,)))
add_cases("gelu_bwd",
          AuditCase("odd_shape", make_gelu_bwd_odd, outputs=(2,)),
          AuditCase("saturated", make_gelu_bwd_saturated, outputs=(2,)))


def make_rope_fwd_odd(g):
    t, h, hd = 33, 5, 64
    pos = torch.randint(0, 4096, (t,), device="cuda", dtype=torch.int32,
                        generator=g)
    return (rand(g, t, h * hd),
            torch.empty(t, h * hd, device="cuda", dtype=torch.bfloat16),
            pos, h, hd, 10000.0), {}


def make_rope_fwd_huge_pos(g):
    t, h, hd = 16, 2, 32
    pos = torch.full((t,), 10_000_000, device="cuda", dtype=torch.int32)
    return (rand(g, t, h * hd),
            torch.empty(t, h * hd, device="cuda", dtype=torch.bfloat16),
            pos, h, hd, 10000.0), {}


def make_rope_bwd_odd(g):
    t, h, hd = 33, 5, 64
    pos = torch.randint(0, 4096, (t,), device="cuda", dtype=torch.int32,
                        generator=g)
    return (rand(g, t, h * hd),
            torch.empty(t, h * hd, device="cuda", dtype=torch.bfloat16),
            pos, h, hd, 10000.0), {}


add_cases("rope_fwd",
          AuditCase("odd_rows", make_rope_fwd_odd, outputs=(1,)),
          AuditCase("huge_positions", make_rope_fwd_huge_pos, outputs=(1,), tol=2e-1))
add_cases("rope_bwd",
          AuditCase("odd_rows", make_rope_bwd_odd, outputs=(1,)))


def make_adamw_first_step(g):
    n = 3 * 1021
    w = rand(g, n)
    gr = rand(g, n)
    m = torch.zeros(n, device="cuda", dtype=torch.bfloat16)
    v = torch.zeros(n, device="cuda", dtype=torch.bfloat16)
    kw = dict(lr=1e-3, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.1, step=1)
    return (w, gr, m, v), kw


def make_adamw_huge_grad(g):
    n = 4096
    w = rand(g, n)
    gr = rand(g, n, scale=1e4)
    m = rand(g, n, scale=1e-2)
    v = rand(g, n, scale=1e-4).abs()
    kw = dict(lr=1e-3, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.1, step=7)
    return (w, gr, m, v), kw


add_cases("adamw_step",
          AuditCase("first_step", make_adamw_first_step, inout=(0, 2, 3)),
          AuditCase("huge_grad", make_adamw_huge_grad, inout=(0, 2, 3)))


def make_ce_odd(g):
    t, v = 33, 4099
    logits = rand(g, t, v, scale=4.0)
    targets = torch.randint(0, v, (t,), device="cuda", dtype=torch.int32,
                            generator=g)
    return (logits, targets,
            torch.empty(1, device="cuda", dtype=torch.float32),
            torch.empty(t, v, device="cuda", dtype=torch.bfloat16)), {}


def make_ce_extreme_logits(g):
    t, v = 16, 512
    logits = rand(g, t, v, scale=80.0)      # online-max must absorb this
    targets = torch.randint(0, v, (t,), device="cuda", dtype=torch.int32,
                            generator=g)
    return (logits, targets,
            torch.empty(1, device="cuda", dtype=torch.float32),
            torch.empty(t, v, device="cuda", dtype=torch.bfloat16)), {}


def make_ce_some_ignored(g):
    t, v = 32, 1024
    logits = rand(g, t, v, scale=4.0)
    targets = torch.randint(0, v, (t,), device="cuda", dtype=torch.int32,
                            generator=g)
    targets[5:9] = -1                        # ignored rows: zero nll + zero row
    return (logits, targets,
            torch.empty(1, device="cuda", dtype=torch.float32),
            torch.empty(t, v, device="cuda", dtype=torch.bfloat16)), \
        {"total_rows": 28}


def make_ce_all_ignored(g):
    t, v = 8, 256
    logits = rand(g, t, v, scale=4.0)
    targets = torch.full((t,), -1, device="cuda", dtype=torch.int32)
    return (logits, targets,
            torch.empty(1, device="cuda", dtype=torch.float32),
            torch.empty(t, v, device="cuda", dtype=torch.bfloat16)), \
        {"total_rows": 1}


def make_ce_boundary_targets(g):
    t, v = 16, 1000
    logits = rand(g, t, v, scale=4.0)
    targets = torch.zeros(t, device="cuda", dtype=torch.int32)
    targets[::2] = v - 1                     # first/last vocab columns
    return (logits, targets,
            torch.empty(1, device="cuda", dtype=torch.float32),
            torch.empty(t, v, device="cuda", dtype=torch.bfloat16)), {}


add_cases("ce_loss_fwd_bwd",
          AuditCase("odd_vocab", make_ce_odd, outputs=(2, 3), tol=2e-3),
          AuditCase("extreme_logits", make_ce_extreme_logits, outputs=(2, 3), tol=2e-3),
          AuditCase("some_ignored", make_ce_some_ignored, outputs=(2, 3), tol=2e-3),
          AuditCase("all_ignored", make_ce_all_ignored, outputs=(2, 3), tol=2e-3),
          AuditCase("boundary_targets", make_ce_boundary_targets, outputs=(2, 3), tol=2e-3))


def make_embed_bwd_fresh(g):
    t, v, d = 64, 97, 33
    tokens = torch.randint(0, v, (t,), device="cuda", dtype=torch.int32,
                           generator=g)
    return (tokens, rand(g, t, d),
            torch.empty(v, d, device="cuda", dtype=torch.bfloat16)), \
        {"zero_first": True}


def make_embed_bwd_all_same_token(g):
    t, v, d = 128, 97, 33
    tokens = torch.full((t,), 41, device="cuda", dtype=torch.int32)
    return (tokens, rand(g, t, d),
            torch.empty(v, d, device="cuda", dtype=torch.bfloat16)), \
        {"zero_first": True}


def make_embed_bwd_accumulate(g):
    t, v, d = 64, 97, 33
    tokens = torch.randint(0, v, (t,), device="cuda", dtype=torch.int32,
                           generator=g)
    dw = rand(g, v, d, scale=0.1)
    tokens[0] = 0
    tokens[1] = v - 1                        # boundary vocab rows
    return (tokens, rand(g, t, d), dw), {"zero_first": False}


add_cases("embed_bwd_accum",
          AuditCase("fresh", make_embed_bwd_fresh, outputs=(2,), tol=2e-2),
          AuditCase("all_same_token", make_embed_bwd_all_same_token, outputs=(2,), tol=2e-2),
          AuditCase("accumulate", make_embed_bwd_accumulate, inout=(2,), tol=2e-2))


# ====================== case builders: MoE family ============================

def route_from_logits(g, t, e, k):
    logits = rand(g, t, e)
    p = torch.softmax(logits.float(), dim=-1)
    w, ids = torch.topk(p, k, dim=-1)
    return logits, (w / w.sum(-1, keepdim=True)).to(torch.bfloat16), \
        ids.to(torch.int32)


def slot_of_from_ids(ids, k):
    flat = ids.reshape(-1).long()
    order = torch.argsort(flat, stable=True)
    slot = torch.empty_like(order)
    slot[order] = torch.arange(order.numel(), device=order.device)
    return slot.view(-1, k).to(torch.int32)


def make_topk_softmax_odd(g):
    t, e, k = 37, 24, 3
    logits = rand(g, t, e)
    return (logits, torch.empty(t, k, device="cuda", dtype=torch.bfloat16),
            torch.empty(t, k, device="cuda", dtype=torch.int32)), \
        {"top_k": k, "mode": "softmax_then_topk"}


def make_topk_softmax_mode0(g):
    t, e, k = 37, 24, 3
    logits = rand(g, t, e)
    return (logits, torch.empty(t, k, device="cuda", dtype=torch.bfloat16),
            torch.empty(t, k, device="cuda", dtype=torch.int32)), \
        {"top_k": k, "mode": "topk_then_softmax"}


def make_topk_softmax_ties(g):
    t, e, k = 16, 16, 4
    logits = torch.zeros(t, e, device="cuda", dtype=torch.bfloat16)
    return (logits, torch.empty(t, k, device="cuda", dtype=torch.bfloat16),
            torch.empty(t, k, device="cuda", dtype=torch.int32)), \
        {"top_k": k, "mode": "softmax_then_topk"}


def make_topk_softmax_full_k(g):
    t, e = 16, 8
    logits = rand(g, t, e)
    return (logits, torch.empty(t, e, device="cuda", dtype=torch.bfloat16),
            torch.empty(t, e, device="cuda", dtype=torch.int32)), \
        {"top_k": e, "mode": "softmax_then_topk"}


add_cases("moe_topk_softmax",
          AuditCase("odd_shape", make_topk_softmax_odd, outputs=(1, 2), tol=2e-2),
          AuditCase("mode_topk_then_softmax", make_topk_softmax_mode0, outputs=(1, 2), tol=2e-2),
          AuditCase("total_ties", make_topk_softmax_ties, outputs=(1, 2), tol=2e-2),
          AuditCase("k_equals_e", make_topk_softmax_full_k, outputs=(1, 2), tol=2e-2))


def make_router_bwd(mode):
    def build(g):
        t, e, k = 37, 24, 3
        logits, w, ids = route_from_logits(g, t, e, k)
        dprob = rand(g, t, k, dtype=torch.float32)
        return (dprob, w, ids, logits,
                torch.empty(t, e, device="cuda", dtype=torch.float32)), \
            {"mode": mode}
    return build


make_router_bwd_mode0 = make_router_bwd("topk_then_softmax")
make_router_bwd_mode1 = make_router_bwd("softmax_then_topk")


add_cases("moe_router_bwd",
          AuditCase("topk_then_softmax", make_router_bwd_mode0, outputs=(4,), tol=2e-2),
          AuditCase("softmax_then_topk", make_router_bwd_mode1, outputs=(4,), tol=2e-2))


def make_router_bwd_sigmoid(g):
    t, e, k = 37, 24, 3
    logits = rand(g, t, e)
    s = torch.sigmoid(logits.float())
    w, ids = torch.topk(s, k, dim=-1)
    w = (2.5 * w / w.sum(-1, keepdim=True)).to(torch.bfloat16)
    dprob = rand(g, t, k, dtype=torch.float32)
    return (dprob, w, ids.to(torch.int32), logits,
            torch.empty(t, e, device="cuda", dtype=torch.float32)), {}


add_cases("moe_router_bwd_sigmoid",
          AuditCase("basic", make_router_bwd_sigmoid, outputs=(4,), tol=2e-2))


def make_sigmoid_noaux(g):
    t, e, k = 37, 32, 4
    logits = rand(g, t, e)
    bias = rand(g, e, dtype=torch.float32, scale=0.5)
    return (logits, bias,
            torch.empty(t, k, device="cuda", dtype=torch.bfloat16),
            torch.empty(t, k, device="cuda", dtype=torch.int32)), \
        {"top_k": k, "n_group": 8, "topk_group": 3, "routed_scaling": 2.5}


def make_sigmoid_noaux_extreme_bias(g):
    t, e, k = 16, 32, 4
    logits = rand(g, t, e)
    bias = torch.zeros(e, device="cuda", dtype=torch.float32)
    bias[:16] = 100.0                    # selection forced into 4 groups
    return (logits, bias,
            torch.empty(t, k, device="cuda", dtype=torch.bfloat16),
            torch.empty(t, k, device="cuda", dtype=torch.int32)), \
        {"top_k": k, "n_group": 8, "topk_group": 3, "routed_scaling": 2.5}


add_cases("moe_topk_sigmoid_noaux",
          AuditCase("basic", make_sigmoid_noaux, outputs=(2, 3), tol=2e-2),
          AuditCase("extreme_bias", make_sigmoid_noaux_extreme_bias, outputs=(2, 3), tol=2e-2))


def make_aux_lb_grad(g):
    t, e, k = 64, 16, 2
    logits = rand(g, t, e)
    counts = torch.zeros(e, device="cuda", dtype=torch.int32)
    counts[0] = t * k                     # maximal imbalance
    dlogits = rand(g, t, e, dtype=torch.float32, scale=0.1)
    return (logits, counts, dlogits), {"alpha": 0.02, "top_k": k}


def make_aux_lb_grad_single_token(g):
    t, e, k = 1, 16, 2
    logits = rand(g, t, e)
    counts = torch.zeros(e, device="cuda", dtype=torch.int32)
    counts[3] = 1
    counts[7] = 1
    dlogits = torch.zeros(t, e, device="cuda", dtype=torch.float32)
    return (logits, counts, dlogits), {"alpha": 0.02, "top_k": k}


add_cases("moe_aux_lb_grad",
          AuditCase("max_imbalance", make_aux_lb_grad, inout=(2,), tol=2e-2),
          AuditCase("single_token", make_aux_lb_grad_single_token, inout=(2,), tol=2e-2))


def make_seq_aux_grad(g):
    t, e, k = 32, 16, 2
    logits = rand(g, t, e)
    ids = torch.randint(0, e, (t, k), device="cuda", dtype=torch.int32,
                        generator=g)
    dlogits = rand(g, t, e, dtype=torch.float32, scale=0.1)
    return (logits, ids, dlogits), \
        {"alpha": 0.02, "top_k": k, "seq_bounds": ((0, 12), (12, 31), (31, 32))}


add_cases("moe_seq_aux_grad",
          AuditCase("ragged_with_len1_seq", make_seq_aux_grad, inout=(2,), tol=2e-2))


def make_sort_case(ids_fn):
    def build(g):
        t, e, k = 37, 8, 2
        ids = ids_fn(g, t, e, k)
        return (ids, torch.empty(t * k, device="cuda", dtype=torch.int32),
                torch.empty(e + 1, device="cuda", dtype=torch.int32)), \
            {"n_experts": e}
    return build


def ids_uniform(g, t, e, k):
    return torch.randint(0, e, (t, k), device="cuda", dtype=torch.int32,
                         generator=g)


def ids_one_expert(g, t, e, k):
    return torch.full((t, k), e - 1, device="cuda", dtype=torch.int32)


make_sort_uniform = make_sort_case(ids_uniform)
make_sort_one_expert = make_sort_case(ids_one_expert)

add_cases("moe_sort",
          AuditCase("uniform", make_sort_uniform, outputs=(1, 2)),
          AuditCase("all_one_expert_rest_empty", make_sort_one_expert, outputs=(1, 2)))


def make_dispatch_fwd(g):
    t, d, e, k = 37, 33, 8, 2
    ids = torch.randint(0, e, (t, k), device="cuda", dtype=torch.int32,
                        generator=g)
    order = torch.argsort(ids.reshape(-1).long(), stable=True).to(torch.int32)
    return (rand(g, t, d), order,
            torch.empty(t * k, d, device="cuda", dtype=torch.bfloat16)), \
        {"top_k": k}


add_cases("moe_dispatch_fwd",
          AuditCase("odd_shape", make_dispatch_fwd, outputs=(2,)))


def make_dispatch_bwd(g):
    t, d, k = 37, 33, 2
    ids = torch.randint(0, 8, (t, k), device="cuda", dtype=torch.int32,
                        generator=g)
    slot = slot_of_from_ids(ids, k)
    return (rand(g, t * k, d), slot,
            torch.empty(t, d, device="cuda", dtype=torch.float32)), {}


add_cases("moe_dispatch_bwd",
          AuditCase("odd_shape", make_dispatch_bwd, outputs=(2,), tol=2e-2))


def make_combine_fwd(g):
    t, d, k = 37, 33, 2
    ids = torch.randint(0, 8, (t, k), device="cuda", dtype=torch.int32,
                        generator=g)
    slot = slot_of_from_ids(ids, k)
    w = torch.softmax(rand(g, t, k).float(), -1).to(torch.bfloat16)
    return (rand(g, t * k, d), slot, w, rand(g, t, d),
            torch.empty(t, d, device="cuda", dtype=torch.bfloat16)), {}


add_cases("moe_combine_fwd",
          AuditCase("odd_shape", make_combine_fwd, outputs=(4,), tol=2e-2))


def make_scale_rows(g):
    rows, n = 37, 511
    srw = rand(g, rows, dtype=torch.float32).abs().add(0.01)
    return (rand(g, rows, n), srw), {}


add_cases("moe_scale_rows",
          AuditCase("odd_shape", make_scale_rows, inout=(0,), tol=2e-2))


def make_rowdot(g):
    rows, n = 37, 511
    return (rand(g, rows, n), rand(g, rows, n),
            torch.empty(rows, device="cuda", dtype=torch.float32)), {}


add_cases("moe_rowdot",
          AuditCase("odd_shape", make_rowdot, outputs=(2,), tol=2e-2))


def grouped_inputs(g, counts, kd, n):
    e = len(counts)
    m = sum(counts)
    offs = torch.tensor([0] + list(torch.tensor(counts).cumsum(0)),
                        device="cuda", dtype=torch.int32)
    x = rand(g, m, kd, scale=0.5)
    w = rand(g, e, kd, n, scale=0.5)
    return x, w, offs, m


def make_grouped_fwd_empty_expert(g):
    x, w, offs, m = grouped_inputs(g, (7, 0, 19, 0, 5, 33, 0, 1), 64, 48)
    return (x, w, offs,
            torch.empty(m, 48, device="cuda", dtype=torch.bfloat16)), {}


def make_grouped_fwd_one_expert(g):
    x, w, offs, m = grouped_inputs(g, (0, 0, 0, 65, 0, 0, 0, 0), 64, 48)
    return (x, w, offs,
            torch.empty(m, 48, device="cuda", dtype=torch.bfloat16)), {}


def make_grouped_dgrad_empty_expert(g):
    x, w, offs, m = grouped_inputs(g, (7, 0, 19, 0, 5, 33, 0, 1), 64, 48)
    dy = rand(g, m, 48, scale=0.5)
    return (dy, w, offs,
            torch.empty(m, 64, device="cuda", dtype=torch.bfloat16)), {}


def make_grouped_wgrad_empty_expert(g):
    x, w, offs, m = grouped_inputs(g, (7, 0, 19, 0, 5, 33, 0, 1), 64, 48)
    dy = rand(g, m, 48, scale=0.5)
    return (x, dy, offs,
            torch.empty(8, 64, 48, device="cuda", dtype=torch.bfloat16)), \
        {"accumulate": False}


def make_grouped_wgrad_accumulate(g):
    x, w, offs, m = grouped_inputs(g, (7, 0, 19, 0, 5, 33, 0, 1), 64, 48)
    dy = rand(g, m, 48, scale=0.5)
    dw = rand(g, 8, 64, 48, scale=0.1)
    return (x, dy, offs, dw), {"accumulate": True}


# every grouped case keeps offsets[-1] == M: that is the op CONTRACT (rows
# beyond offsets[-1] are impl-defined — probed: eager zeroes, triton leaves
# prior bytes, aten computes garbage; the module docstring pins it)
add_cases("moe_grouped_mm_fwd",
          AuditCase("empty_experts", make_grouped_fwd_empty_expert, outputs=(3,), tol=3e-2),
          AuditCase("all_rows_one_expert", make_grouped_fwd_one_expert, outputs=(3,), tol=3e-2))
add_cases("moe_grouped_mm_dgrad",
          AuditCase("empty_experts", make_grouped_dgrad_empty_expert, outputs=(3,), tol=3e-2))
add_cases("moe_grouped_mm_wgrad",
          AuditCase("empty_experts_create", make_grouped_wgrad_empty_expert, outputs=(3,), tol=3e-2),
          AuditCase("empty_experts_accumulate", make_grouped_wgrad_accumulate, inout=(3,), tol=3e-2))


# ================= case builders: muon / gdn / causal conv ===================

def make_muon_matrix(g):
    r, c = 65, 47
    w = rand(g, r * c)
    gr = rand(g, r * c)
    m = rand(g, r * c, scale=0.1)
    return (w, gr, m), dict(shape=(r, c), lr=1e-3, beta=0.95, eps=1e-7,
                            weight_decay=0.1)


def make_muon_expert_batched(g):
    e, r, c = 3, 33, 17
    w = rand(g, e * r * c)
    gr = rand(g, e * r * c)
    m = torch.zeros(e * r * c, device="cuda", dtype=torch.bfloat16)
    return (w, gr, m), dict(shape=(e, r, c), lr=1e-3, beta=0.95, eps=1e-7,
                            weight_decay=0.1)


def make_muon_zero_grad(g):
    r, c = 32, 32
    w = rand(g, r * c)
    gr = torch.zeros(r * c, device="cuda", dtype=torch.bfloat16)
    m = torch.zeros(r * c, device="cuda", dtype=torch.bfloat16)
    return (w, gr, m), dict(shape=(r, c), lr=1e-3, beta=0.95, eps=1e-7,
                            weight_decay=0.1)


add_cases("muon_step",
          AuditCase("odd_matrix", make_muon_matrix, inout=(0, 2)),
          AuditCase("expert_batched", make_muon_expert_batched, inout=(0, 2)),
          AuditCase("zero_grad", make_muon_zero_grad, inout=(0, 2)))


def make_gated_rmsnorm_fwd(g):
    rows, d = 37, 96
    return (rand(g, rows, d), rand(g, rows, d), rand(g, d),
            torch.empty(rows, d, device="cuda", dtype=torch.bfloat16),
            torch.empty(rows, device="cuda", dtype=torch.float32)), {}


def make_gated_rmsnorm_fwd_saturated_gate(g):
    rows, d = 16, 64
    return (rand(g, rows, d), rand(g, rows, d, scale=40.0), rand(g, d),
            torch.empty(rows, d, device="cuda", dtype=torch.bfloat16),
            torch.empty(rows, device="cuda", dtype=torch.float32)), {}


def make_gated_rmsnorm_bwd(g):
    rows, d = 37, 96
    o, z, w = rand(g, rows, d), rand(g, rows, d), rand(g, d)
    rstd = torch.rsqrt(o.float().pow(2).mean(-1) + 1e-5)
    return (rand(g, rows, d), o, z, w, rstd,
            torch.empty(rows, d, device="cuda", dtype=torch.bfloat16),
            torch.empty(rows, d, device="cuda", dtype=torch.bfloat16),
            torch.empty(d, device="cuda", dtype=torch.float32),
            torch.empty(rows, d, device="cuda", dtype=torch.bfloat16)), {}


add_cases("gated_rmsnorm_fwd",
          AuditCase("odd_rows", make_gated_rmsnorm_fwd, outputs=(3, 4), tol=3e-2),
          AuditCase("saturated_gate", make_gated_rmsnorm_fwd_saturated_gate, outputs=(3, 4), tol=3e-2))
add_cases("gated_rmsnorm_bwd",
          AuditCase("odd_rows", make_gated_rmsnorm_bwd, outputs=(5, 6, 7, 8), tol=6e-2))


def conv_cu(bounds):
    return torch.tensor(bounds, device="cuda", dtype=torch.int32)


def make_conv_fwd_packed(g):
    t, d, wd = 37, 24, 4
    return (rand(g, t, d), rand(g, d, wd),
            torch.empty(t, d, device="cuda", dtype=torch.bfloat16),
            conv_cu([0, 5, 6, 37])), {}


def make_conv_fwd_short_seqs(g):
    t, d, wd = 7, 24, 4          # every segment shorter than the kernel
    return (rand(g, t, d), rand(g, d, wd),
            torch.empty(t, d, device="cuda", dtype=torch.bfloat16),
            conv_cu([0, 1, 2, 4, 7])), {}


def make_conv_bwd_packed(g):
    t, d, wd = 37, 24, 4
    x, w = rand(g, t, d), rand(g, d, wd)
    return (x, rand(g, t, d), w,
            torch.empty(t, d, device="cuda", dtype=torch.bfloat16),
            torch.empty(d, wd, device="cuda", dtype=torch.bfloat16),
            conv_cu([0, 5, 6, 37])), {}


add_cases("causal_conv1d_silu_fwd",
          AuditCase("packed_with_len1", make_conv_fwd_packed, outputs=(2,), tol=3e-2),
          AuditCase("seqs_shorter_than_kernel", make_conv_fwd_short_seqs, outputs=(2,), tol=3e-2))
add_cases("causal_conv1d_silu_bwd",
          AuditCase("packed_with_len1", make_conv_bwd_packed, outputs=(3, 4), tol=6e-2))


# ======================= case builders: DSA family ===========================
# The two latent Part-1 bugs lived here: dead-row online softmax and a
# consumed-but-unwritten workspace region. Geometry deliberately ugly:
# L % 64 != 0, ragged multi-seq, len-1 sequences, far-from-diagonal
# selections (early tiles dead), k > L.

HI, DI = 4, 32     # indexer heads
H, D = 2, 64       # sparse-attn heads


def dsa_bounds_single(t):
    return ((0, t),)


def dsa_bounds_ragged(t):
    return ((0, 65), (65, 66), (66, t))


def make_index_scores(bounds_fn, t):
    def build(g):
        q = rand(g, t, HI * DI, scale=0.3)
        k = rand(g, t, DI, scale=0.3)
        wts = rand(g, t, HI, dtype=torch.float32).abs()
        return (q, k, wts,
                torch.empty(t, t, device="cuda", dtype=torch.float32)), \
            {"n_heads": HI, "head_dim": DI, "seq_bounds": bounds_fn(t)}
    return build


make_index_scores_odd = make_index_scores(dsa_bounds_single, 131)
make_index_scores_ragged = make_index_scores(dsa_bounds_ragged, 149)

add_cases("dsa_index_scores",
          AuditCase("single_seq_odd_len", make_index_scores_odd,
                    outputs=(3,), tol=2e-2, inf_ok=True),
          AuditCase("ragged_with_len1", make_index_scores_ragged,
                    outputs=(3,), tol=2e-2, inf_ok=True))


def causal_dscores(g, t, bounds):
    d = rand(g, t, t, dtype=torch.float32, scale=0.1)
    keep = torch.zeros(t, t, device="cuda", dtype=torch.bool)
    for lo, hi in bounds:
        keep[lo:hi, lo:hi] = torch.ones(hi - lo, hi - lo, device="cuda",
                                        dtype=torch.bool).tril()
    return d.masked_fill(~keep, 0.0)


def make_index_bwd(g):
    t = 131
    bounds = dsa_bounds_ragged(149) if t == 149 else dsa_bounds_single(t)
    q = rand(g, t, HI * DI, scale=0.3)
    k = rand(g, t, DI, scale=0.3)
    wts = rand(g, t, HI, dtype=torch.float32).abs()
    ds = causal_dscores(g, t, bounds)
    return (ds, q, k, wts,
            torch.empty(t, HI * DI, device="cuda", dtype=torch.bfloat16),
            torch.empty(t, DI, device="cuda", dtype=torch.bfloat16),
            torch.empty(t, HI, device="cuda", dtype=torch.float32)), \
        {"n_heads": HI, "head_dim": DI, "seq_bounds": bounds}


def make_index_bwd_ragged(g):
    t = 149
    bounds = dsa_bounds_ragged(t)
    q = rand(g, t, HI * DI, scale=0.3)
    k = rand(g, t, DI, scale=0.3)
    wts = rand(g, t, HI, dtype=torch.float32).abs()
    ds = causal_dscores(g, t, bounds)
    return (ds, q, k, wts,
            torch.empty(t, HI * DI, device="cuda", dtype=torch.bfloat16),
            torch.empty(t, DI, device="cuda", dtype=torch.bfloat16),
            torch.empty(t, HI, device="cuda", dtype=torch.float32)), \
        {"n_heads": HI, "head_dim": DI, "seq_bounds": bounds}


add_cases("dsa_index_bwd",
          AuditCase("single_seq_odd_len", make_index_bwd, outputs=(4, 5, 6), tol=6e-2),
          AuditCase("ragged_with_len1", make_index_bwd_ragged, outputs=(4, 5, 6), tol=6e-2))


def dsa_idx_self_only(t, k):
    """Every row selects ONLY ITSELF: for rows far into the sequence the
    early key tiles are all dead — the exact geometry of the online-softmax
    dead-row bug."""
    return torch.arange(t, device="cuda", dtype=torch.int32) \
        .unsqueeze(1).expand(t, k).contiguous()


def dsa_idx_prefix(g, t, k, bounds):
    """Uniform selection over each row's causal prefix (kk<k rows repeat)."""
    idx = torch.zeros(t, k, device="cuda", dtype=torch.int32)
    for lo, hi in bounds:
        for r in range(lo, hi):
            width = r - lo + 1
            picks = torch.randint(lo, lo + width, (k,), device="cuda",
                                  generator=g, dtype=torch.int32)
            idx[r] = picks
    return idx


def make_sparse_fwd_self_only(g):
    t, k = 131, 8
    bounds = dsa_bounds_single(t)
    return (rand(g, t, H * D, scale=0.3), rand(g, t, H * D, scale=0.3),
            rand(g, t, H * D, scale=0.3), dsa_idx_self_only(t, k),
            torch.empty(t, H * D, device="cuda", dtype=torch.bfloat16),
            torch.empty(H, t, device="cuda", dtype=torch.float32)), \
        {"n_heads": H, "head_dim": D, "seq_bounds": bounds}


def make_sparse_fwd_ragged(g):
    t, k = 149, 8
    bounds = dsa_bounds_ragged(t)
    return (rand(g, t, H * D, scale=0.3), rand(g, t, H * D, scale=0.3),
            rand(g, t, H * D, scale=0.3), dsa_idx_prefix(g, t, k, bounds),
            torch.empty(t, H * D, device="cuda", dtype=torch.bfloat16),
            torch.empty(H, t, device="cuda", dtype=torch.float32)), \
        {"n_heads": H, "head_dim": D, "seq_bounds": bounds}


add_cases("dsa_sparse_attn_fwd",
          AuditCase("self_only_early_tiles_dead", make_sparse_fwd_self_only,
                    outputs=(4, 5), tol=3e-2),
          AuditCase("ragged_with_len1", make_sparse_fwd_ragged,
                    outputs=(4, 5), tol=3e-2))


def make_sparse_bwd_ragged(g):
    t, k = 149, 8
    bounds = dsa_bounds_ragged(t)
    q = rand(g, t, H * D, scale=0.3)
    kf = rand(g, t, H * D, scale=0.3)
    vp = rand(g, t, H * D, scale=0.3)
    idx = dsa_idx_prefix(g, t, k, bounds)
    out = torch.empty(t, H * D, device="cuda", dtype=torch.bfloat16)
    lse = torch.empty(H, t, device="cuda", dtype=torch.float32)
    eager = available_impls("dsa_sparse_attn_fwd")["eager"]
    eager.fn(KCTX, q, kf, vp, idx, out, lse,
             n_heads=H, head_dim=D, seq_bounds=bounds)
    return (rand(g, t, H * D, scale=0.3), q, kf, vp, idx, lse,
            torch.empty(t, H * D, device="cuda", dtype=torch.bfloat16),
            torch.empty(t, H * D, device="cuda", dtype=torch.bfloat16),
            torch.empty(t, H * D, device="cuda", dtype=torch.bfloat16)), \
        {"n_heads": H, "head_dim": D, "seq_bounds": bounds, "out": out}


add_cases("dsa_sparse_attn_bwd",
          AuditCase("ragged_with_len1", make_sparse_bwd_ragged,
                    outputs=(6, 7, 8), tol=6e-2))


def make_probs_sum_ragged(g):
    t, k = 149, 8
    bounds = dsa_bounds_ragged(t)
    q = rand(g, t, H * D, scale=0.3)
    kf = rand(g, t, H * D, scale=0.3)
    vp = rand(g, t, H * D, scale=0.3)
    idx = dsa_idx_prefix(g, t, k, bounds)
    out = torch.empty(t, H * D, device="cuda", dtype=torch.bfloat16)
    lse = torch.empty(H, t, device="cuda", dtype=torch.float32)
    eager = available_impls("dsa_sparse_attn_fwd")["eager"]
    eager.fn(KCTX, q, kf, vp, idx, out, lse,
             n_heads=H, head_dim=D, seq_bounds=bounds)
    return (q, kf, idx, lse,
            torch.empty(t, t, device="cuda", dtype=torch.float32)), \
        {"n_heads": H, "head_dim": D, "seq_bounds": bounds}


add_cases("dsa_probs_sum",
          AuditCase("ragged_with_len1", make_probs_sum_ragged,
                    outputs=(4,), tol=3e-2))


def make_dsa_topk_short_seq(g):
    t, k = 66, 24
    bounds = ((0, 65), (65, 66))
    scores = rand(g, t, t, dtype=torch.float32)
    keep = torch.zeros(t, t, device="cuda", dtype=torch.bool)
    for lo, hi in bounds:
        keep[lo:hi, lo:hi] = torch.ones(hi - lo, hi - lo, device="cuda",
                                        dtype=torch.bool).tril()
    scores = scores.masked_fill(~keep, float("-inf"))
    return (scores, torch.empty(t, k, device="cuda", dtype=torch.int32)), {}


add_cases("dsa_topk",
          AuditCase("k_exceeds_short_rows", make_dsa_topk_short_seq,
                    outputs=(1,)))


def make_pack_bits_ragged(g):
    t, k = 149, 8
    bounds = dsa_bounds_ragged(t)
    idx = dsa_idx_prefix(g, t, k, bounds)
    words = (t + 63) // 64
    return (idx, torch.empty(t, words, device="cuda", dtype=torch.int64)), \
        {"seq_bounds": bounds}


add_cases("dsa_pack_bits",
          AuditCase("ragged_with_len1", make_pack_bits_ragged, outputs=(1,)))


EXEMPT["dsa_sparse_attn_fwd_absorbed"] = (
    "flashmla impl is shape-specialized to (d_qk, d_v) = (576, 512) and the "
    "library is absent on dev boxes (requires() gates it); the eager anchor "
    "is plain torch already covered by tests/dataflow_training/modules/test_mla.py absorbed "
    "gates. Audit it when a flashmla-capable box joins CI."
)


# ============================ the three legs ==================================

def audit_params():
    params = []
    for op, cases in sorted(CASES.items()):
        for impl_id, entry in sorted(available_impls(op).items()):
            for case in cases:
                params.append(pytest.param(entry, case,
                                           id=f"{op}:{impl_id}:{case.name}"))
    return params


@pytest.mark.parametrize("entry,case", audit_params())
def test_write_coverage_poison_invariance(entry, case):
    """Outputs must be a pure function of inputs — never of whatever bytes
    the output buffers or scratch held before the launch."""
    a = run_once(entry, case, seed=17, poison=POISONS[0])
    b = run_once(entry, case, seed=17, poison=POISONS[1])
    labels = [f"arg{i}" for i in tuple(case.outputs) + tuple(case.inout)]
    for name, ta, tb in zip(labels, a, b):
        if entry.deterministic:
            assert bitwise_equal(ta, tb), \
                f"{entry.op}:{entry.impl_id} {case.name} {name}: {poison_report(ta, tb)}"
        else:
            assert rel_l2(ta, tb) < 1e-3, \
                f"{entry.op}:{entry.impl_id} {case.name} {name} varies with poison"


@pytest.mark.parametrize("entry,case", audit_params())
def test_degenerate_finite(entry, case):
    if not case.finite:
        pytest.skip("op documented non-finite for this case")
    outs = run_once(entry, case, seed=23, poison=POISONS[0])
    labels = [f"arg{i}" for i in tuple(case.outputs) + tuple(case.inout)]
    for name, t in zip(labels, outs):
        if not t.dtype.is_floating_point:
            continue
        if case.inf_ok:
            bad = int(torch.isnan(t.float()).sum())
            assert bad == 0, \
                f"{entry.op}:{entry.impl_id} {case.name} {name}: {bad} NaN(s)"
        else:
            bad = int((~torch.isfinite(t.float())).sum())
            assert bad == 0, \
                f"{entry.op}:{entry.impl_id} {case.name} {name}: {bad} non-finite value(s)"


def cross_impl_params():
    params = []
    for op, cases in sorted(CASES.items()):
        impls = available_impls(op)
        if "eager" not in impls or len(impls) < 2:
            continue
        for impl_id, entry in sorted(impls.items()):
            if impl_id == "eager":
                continue
            for case in cases:
                params.append(pytest.param(entry, impls["eager"], case,
                                           id=f"{op}:{impl_id}-vs-eager:{case.name}"))
    return params


@pytest.mark.parametrize("entry,anchor,case", cross_impl_params())
def test_degenerate_cross_impl(entry, anchor, case):
    """Fused and eager must agree ON THE EDGE GEOMETRY — typical shapes are
    already gated elsewhere; the latent bugs lived in the edges."""
    fused = run_once(entry, case, seed=29, poison=POISONS[0])
    ref = run_once(anchor, case, seed=29, poison=POISONS[0])
    labels = [f"arg{i}" for i in tuple(case.outputs) + tuple(case.inout)]
    for name, tf, tr in zip(labels, fused, ref):
        if not tf.dtype.is_floating_point:
            assert torch.equal(tf, tr), \
                f"{entry.op}:{entry.impl_id} {case.name} {name}: integer mismatch vs eager"
        elif case.inf_ok:
            mf, mr = torch.isfinite(tf.float()), torch.isfinite(tr.float())
            assert torch.equal(mf, mr), \
                f"{entry.op}:{entry.impl_id} {case.name} {name}: finiteness pattern differs"
            d = rel_l2(tf.float()[mf], tr.float()[mr])
            assert d < case.tol, \
                f"{entry.op}:{entry.impl_id} {case.name} {name}: rel_l2 {d:.3e} vs eager (finite subset)"
        else:
            d = rel_l2(tf, tr)
            assert d < case.tol, \
                f"{entry.op}:{entry.impl_id} {case.name} {name}: rel_l2 {d:.3e} vs eager"


def test_every_registered_op_is_audited():
    """A new kernel op cannot land silently unaudited: it either carries
    cases here or an explicit reasoned exemption."""
    import importlib
    import pkgutil

    import dataflow_training.kernels as kernels_pkg

    for m in pkgutil.iter_modules(kernels_pkg.__path__):
        if m.name != "registry":
            importlib.import_module(f"dataflow_training.kernels.{m.name}")
    all_ops = set(reg._REGISTRY)
    covered = set(CASES) | set(EXEMPT)
    missing = sorted(all_ops - covered)
    stale = sorted((set(CASES) | set(EXEMPT)) - all_ops)
    assert not missing, f"ops with no audit cases and no exemption: {missing}"
    assert not stale, f"audit entries for ops no longer registered: {stale}"
    overlap = sorted(set(CASES) & set(EXEMPT))
    assert not overlap, f"ops both audited and exempt: {overlap}"
