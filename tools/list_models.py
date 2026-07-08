"""Enumerate builtin model families and their presets as a markdown
table — the generator behind docs/builtin_models.md.

    python tools/list_models.py > docs/builtin_models.md

Parameter counts are computed from the lowered program's weight-object
byte sizes at the preset's dtype policy (pure layout arithmetic — no
tensors, so trillion-parameter presets enumerate in milliseconds).
Presets whose lowering is deliberately host-RAM-infeasible on one box
still lower fine (lowering allocates nothing).
"""
from __future__ import annotations

import sys

from dataflow.training import families as F

HEADER = """# Builtin model families and presets

GENERATED — regenerate with `python tools/list_models.py >
docs/builtin_models.md` after adding a family or preset. Families
register in `training/families.py`; presets are classmethods on each
family's Shaped config (external families: docs/extending_external.md).

Params are computed from the lowered weight layouts at each preset's
dtype policy (bf16 default). `tiny` presets are the correctness-ladder
scale (docs/extending.md); `mini` presets are single-GPU bench scale;
full-scale presets match the published architectures (dims verified
against the HF configs; totals match announced parameter counts).

| family | preset | layers | d_model | vocab | seq default | params |
|---|---|---|---|---|---|---|
"""


def params_of(fam, cfg) -> int:
    prog = fam.lower(cfg)
    from dataflow.tasks.interop import TORCH_DTYPE_BY_NAME  # noqa: F401

    total_bytes = 0
    n = 0
    for oid, size in prog.object_sizes().items():
        if oid.startswith("W_"):
            total_bytes += size
            n += 1
    # params ~= bytes / 2 under the default all-bf16 param policy;
    # alignment padding makes this a slight over-count (<0.1%)
    return total_bytes // 2


def fmt_params(p: int) -> str:
    if p >= 10**12:
        return f"{p / 10**12:.3f}T"
    if p >= 10**9:
        return f"{p / 10**9:.2f}B"
    if p >= 10**6:
        return f"{p / 10**6:.1f}M"
    return f"{p / 10**3:.0f}K"


def main() -> None:
    F.load_plugins()
    out = [HEADER.rstrip("\n")]
    seen_shapes: dict[tuple, str] = {}
    for name in F._FAMILIES:
        fam = F.family(name)
        cls = fam.config_type
        presets = [n for n in dir(cls)
                   if not n.startswith("_")
                   and isinstance(cls.__dict__.get(n), classmethod)]
        for preset in sorted(presets):
            try:
                cfg = getattr(cls, preset)()
            except TypeError:
                continue          # needs arguments: not a plain preset
            try:
                p = params_of(fam, cfg)
            except Exception as exc:  # pragma: no cover
                print(f"[warn] {name}.{preset}: {exc}", file=sys.stderr)
                continue
            key = (name, cfg.n_layers, cfg.d_model, p)
            alias = seen_shapes.get(key)
            note = f" (alias of `{alias}`)" if alias else ""
            if not alias:
                seen_shapes[key] = preset
            out.append(
                f"| {name} | `{preset}`{note} | {cfg.n_layers} | "
                f"{cfg.d_model} | {cfg.vocab_size} | {cfg.seq_len} | "
                f"{fmt_params(p)} |")
    out.append("""
Notes:
- Aliases share the exact architecture shape of an earlier preset
  (e.g. Kimi K2.5/2.6/2.7 are shape-identical to K2; GLM 5.1 to 5).
- `bench_train`/`bench_frontier` config names compose as
  `{preset-prefix}-s{seq}k-bs{B}ga{G}` — see docs/benchmarking.md.
- Correctness: `python tools/verify_family.py --family <name>`.
""")
    print("\n".join(out))


if __name__ == "__main__":
    main()
