"""FLOP accounting over a lowered program (docs/notes plan v3).

THE source of truth is the per-task ``metadata["cost_subops"]`` every
family's builder already stamps and the simulator already prices — this
module never re-derives model math, it READS the same numbers and sorts
them into the three reported quantities:

- EFFECTIVE flops: algorithmic fwd + bwd. Attention fwd is the causal
  (triangular) count 0.5*4*sum(s_i^2)*H*hd; attention bwd counts the
  0.5*8 algorithmic factor — the stamped seeds carry 0.5*10 (flash's
  in-kernel recompute is real executed work), so effective applies the
  8/10 correction on the CAUSAL_DENSE kinds whose seeds are known to
  use it. Planner recompute tasks are excluded (not model work).
- HARDWARE flops: everything actually executed — stamped values as-is
  (attention bwd at 0.5*10) plus recompute-task replays.
- OPTIMIZER flops: matmul work only, from the weight layouts x the
  config's OptPolicy — adamw/sgd/sgdm are elementwise (0); muon fields
  count the Newton-Schulz quintic (5 iterations of X@X^T, A@A, and the
  polynomial apply). The ALL-IN quantity = hardware + optimizer.

Variable sequence lengths: attention subops scale by the round's
quadratic mass ratio sum(l_j^2) / (t * seq_len) — exact for causal
block-diagonal packing, and the family's width constant cancels in the
ratio. Non-causal-dense kinds (DSA selected-prefix) keep their stamped
static values (their seeds own their sparsity math).

Completeness: a task with NO cost_subops metadata hard-fails unless its
compute_block_key is in EXEMPT (reasoned zero-flop plumbing). A family
whose kinds are not in CAUSAL_DENSE still reports (stamped values,
effective == hardware for attention) — absent corrections, never wrong
sums.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# compute_block_keys that legitimately carry no cost_subops / no flops
EXEMPT = {
    "prologue_round",      # round-boundary publish, no math
    "family_init",         # init program task
}

# (family kind key_prefix) whose attention subops are TRUE causal
# softmax with the pinned factors (fwd 0.5*4*sum(s^2)*H*hd, bwd
# 0.5*10): these get the effective 8/10 bwd correction and the varlen
# quadratic scaling. Kinds NOT here (DeltaNet scans, DSA selected-
# prefix) keep stamped values, effective == hardware, never scaled —
# a linear-in-t or sparsity-owned cost has no quadratic mass to scale.
CAUSAL_DENSE_PREFIXES = {"block", "tp_block", "gattn"}

ATTN_BWD_EFFECTIVE_OVER_HARDWARE = 8.0 / 10.0


def seq_sq_ratio(seq_lens, tokens: int, seq_len: int) -> float:
    """sum(l_j^2) / (t * seq) — actual over uniform quadratic mass."""
    if not seq_lens:
        return 1.0
    return sum(int(n) * int(n) for n in seq_lens) / float(tokens * seq_len)


@dataclass
class FlopReport:
    """Static per-STEP totals plus the pieces per-step scaling needs."""

    mm_effective: float = 0.0        # matmul subops, fwd+bwd (no recompute)
    mm_hardware: float = 0.0         # + recompute-task matmuls
    # CAUSAL buckets (CAUSAL_DENSE kinds): varlen-scalable, 8/10 split
    attn_fwd: float = 0.0            # stamped causal fwd (uniform static)
    attn_bwd_hw: float = 0.0         # stamped 0.5*10 bwd (uniform static)
    attn_bwd_eff: float = 0.0        # the 0.5*8 correction
    attn_recompute_hw: float = 0.0   # recompute-task causal replays
    # STATIC attention-class buckets (linear scans, DSA selection):
    # stamped as-is, effective == hardware, never varlen-scaled
    attn_static: float = 0.0         # fwd + bwd of non-causal kinds
    attn_static_rc_hw: float = 0.0   # their recompute replays (hw only)
    optimizer: float = 0.0           # NS matmuls (muon fields); adamw 0
    by_group: dict = field(default_factory=dict)

    def per_step(self, seq_lens_by_round=None, *, tokens: int = 0,
                 seq_len: int = 0):
        """(effective, hardware, all_in) flops for ONE step; doc-aware
        rounds pass {round: [lens]} for the exact quadratic mass.
        Scaling applies to the whole attention bucket — correct for
        all-causal families (the ones that run doc-aware today); mixed
        causal/DSA families should pass None (static)."""
        scale = 1.0
        if seq_lens_by_round:
            ratios = [seq_sq_ratio(lens, tokens, seq_len)
                      for lens in seq_lens_by_round.values()]
            scale = sum(ratios) / len(ratios)
        a_fwd = self.attn_fwd * scale
        a_bwd_eff = self.attn_bwd_eff * scale
        a_bwd_hw = self.attn_bwd_hw * scale
        a_rc = self.attn_recompute_hw * scale
        effective = (self.mm_effective + a_fwd + a_bwd_eff
                     + self.attn_static)
        hardware = (self.mm_hardware + a_fwd + a_bwd_hw + a_rc
                    + self.attn_static + self.attn_static_rc_hw)
        return effective, hardware, hardware + self.optimizer


def _kind_prefix(key: str) -> str:
    for suffix in ("_fwd", "_bwd", "_recompute"):
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return key


def muon_ns_flops(m: int, n: int, iters: int = 5) -> float:
    """Newton-Schulz quintic on an (m, n) matrix, m <= n after the
    orientation swap: per iteration A = X@X^T (2 m^2 n), B = A@A
    (2 m^3), (bA + cB)@X (2 m^2 n)."""
    if m > n:
        m, n = n, m
    per_iter = 2.0 * m * m * n + 2.0 * m ** 3 + 2.0 * m * m * n
    return iters * per_iter


def optimizer_flops(cfg) -> float:
    """Matmul optimizer work per step from the config's OptPolicy:
    muon fields count NS5; everything else is elementwise (0)."""
    from dataflow_training.blocks.optim import resolve_opt_policy
    from dataflow_training.model_families.families import resolve_family

    fam = resolve_family(cfg)
    dims = fam.derive_dims(cfg)
    policy = resolve_opt_policy(getattr(dims, "opt_policy", None) or "adamw")
    total = 0.0
    if fam.weight_layout is None:
        # heterogeneous families expose layouts per kind through their
        # blocks; walk layer 0..n via the family lower is overkill here —
        # uniform-layout families cover the current muon studies. A
        # heterogeneous muon config reports optimizer=0 with a note.
        return 0.0
    n_layers = cfg.n_layers
    for layer in range(n_layers):
        wl = fam.weight_layout(dims, layer=layer)
        for f in wl.fields:
            if len(f.shape) != 2:
                continue
            rule = policy.for_field(f.name, layer, f.shape)
            if rule == "muon":
                total += muon_ns_flops(int(f.shape[0]), int(f.shape[1]))
    return total


def flop_report(cfg, program) -> FlopReport:
    """Walk a LOWERED (or planned) program's cost_subops into the
    report. Hard-fails on tasks missing cost metadata (EXEMPT keys and
    zero-subop tasks pass — absent flops is honest zero; absent
    METADATA is a builder bug)."""
    rep = FlopReport()
    for task in program.tasks:
        key = task.compute_block_key
        subops = (task.metadata or {}).get("cost_subops")
        if subops is None:
            if key in EXEMPT:
                continue
            raise ValueError(
                f"flop accounting: task {task.id!r} ({key!r}) carries no "
                f"cost_subops metadata and is not exempt — its builder "
                f"must stamp roofline subops (or add a reasoned EXEMPT "
                f"entry)")
        prefix = _kind_prefix(key)
        causal = prefix in CAUSAL_DENSE_PREFIXES
        is_recompute = key.endswith("_recompute") or task.group == "recompute"
        for op in subops:
            fl = float(op.get("flops", 0) or 0)
            if fl == 0.0:
                continue
            rep.by_group[task.group] = rep.by_group.get(task.group, 0.0) + fl
            if op.get("efficiency") == "attention":
                if not causal:
                    if is_recompute:
                        rep.attn_static_rc_hw += fl
                    else:
                        rep.attn_static += fl
                elif is_recompute:
                    rep.attn_recompute_hw += fl
                elif key.endswith("_bwd"):
                    rep.attn_bwd_hw += fl
                    rep.attn_bwd_eff += fl * ATTN_BWD_EFFECTIVE_OVER_HARDWARE
                else:
                    rep.attn_fwd += fl
            else:
                rep.mm_hardware += fl
                if not is_recompute:
                    rep.mm_effective += fl
    rep.optimizer = optimizer_flops(cfg)
    return rep
