"""Enumerate builtin model families and presets as grouped markdown
tables — the generator behind docs/builtin_models.md.

    python tools/gen_model_docs/list_models.py > docs/builtin_models.md

One section per family; each family's preset table carries, beyond the
common columns, that family's UNIQUE configuration axes (each family is
its own configuration space): the structural config fields not shared
by every family, with nested specs (MoESpec) expanded to dotted scalar
columns. Parameter counts come from the lowered weight layouts (pure
arithmetic — trillion-parameter presets enumerate in milliseconds).
"""
from __future__ import annotations

import dataclasses
import sys
from collections import Counter

from dataflow_training.model_families import families as F

HEADER = """# Builtin model families and presets

GENERATED — regenerate with `python tools/gen_model_docs/list_models.py >
docs/builtin_models.md` after adding a family or preset. Families
register in `dataflow_training/model_families/families.py`; presets are classmethods on each
family's Shaped config (external families: docs/extending_external.md).

Params are computed from the lowered weight layouts at each preset's
dtype policy (bf16 default). `tiny` presets are the correctness-ladder
scale (docs/extending.md); `mini` presets are single-GPU bench scale;
full-scale presets match the published architectures (dims verified
against the HF configs; totals match announced parameter counts).
One section per family; the extra columns in each table are that
family's OWN configuration axes (fields no other family shares).
Each preset name links to its generated deep reference (objects,
stages, kernels) at the standard 16×4K run shape; other run shapes:
`tools/gen_model_docs/gen_model_page.py`. Index: [models/](models/README.md).
"""

# run-shape knobs, policies, and the cross-family commons — never
# "unique configuration axes"
RUN_KNOBS = {
    "batch", "grad_accum_rounds", "num_steps", "seq_len", "max_tokens",
    "optimizer_placement", "opt_policy", "dtypes", "vocab_size",
    "n_layers", "d_model",
}


def params_of(fam, cfg) -> int:
    prog = fam.lower(cfg)
    total = 0
    for oid, size in prog.object_sizes().items():
        if oid.startswith("W_"):
            total += size
    # bytes/2 under the default all-bf16 param policy (<0.1% padding)
    return total // 2


def fmt_params(p: int) -> str:
    if p >= 10**12:
        return f"{p / 10**12:.3f}T"
    if p >= 10**9:
        return f"{p / 10**9:.2f}B"
    if p >= 10**6:
        return f"{p / 10**6:.1f}M"
    return f"{p / 10**3:.0f}K"


def _is_dc(v) -> bool:
    return dataclasses.is_dataclass(v) and not isinstance(v, type)


def fmt_val(v) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, float, str)):
        s = str(v)
        return s if len(s) <= 14 else s[:11] + "..."
    if isinstance(v, tuple):
        if len(v) <= 4:
            return str(v)
        c = Counter(v)
        return "/".join(f"{n}x{k}" for k, n in c.most_common())
    return "-"


def unique_columns(all_configs: dict) -> dict[str, list[str]]:
    names = {fam: {f.name for f in dataclasses.fields(cls)}
             for fam, cls in all_configs.items()}
    common = set.intersection(*names.values())
    return {fam: sorted((ns - common) - RUN_KNOBS)
            for fam, ns in names.items()}


def expand_field(cfg, name):
    v = getattr(cfg, name)
    if _is_dc(v):
        out = []
        for f in dataclasses.fields(v):
            sv = getattr(v, f.name)
            if isinstance(sv, (int, float, bool, str)):
                out.append((f"{name}.{f.name}", fmt_val(sv)))
        return out
    return [(name, fmt_val(v))]


def main() -> None:
    F.load_plugins()
    out = [HEADER.rstrip("\n")]
    seen_shapes: dict[tuple, str] = {}
    uniq = unique_columns({n: F.family(n).config_type for n in F._FAMILIES})
    for name in F._FAMILIES:
        fam = F.family(name)
        cls = fam.config_type
        presets = [n for n in dir(cls)
                   if not n.startswith("_")
                   and isinstance(cls.__dict__.get(n), classmethod)]
        col_labels: list[str] | None = None
        rows = []
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
            pairs = []
            for uc in uniq[name]:
                pairs.extend(expand_field(cfg, uc))
            if col_labels is None:
                col_labels = [lbl for lbl, _ in pairs]
            vals = dict(pairs)
            key = (name, cfg.n_layers, cfg.d_model, p)
            alias = seen_shapes.get(key)
            note = f" (alias of `{alias}`)" if alias else ""
            if not alias:
                seen_shapes[key] = preset
            ucells = "".join(f" {vals.get(lbl, '-')} |"
                             for lbl in (col_labels or []))
            rows.append(
                f"| [`{preset}`](models/{name}/{preset}_16x4K.md){note} | "
                f"{cfg.n_layers} | {cfg.d_model} | "
                f"{cfg.vocab_size} | {cfg.seq_len} |{ucells} "
                f"{fmt_params(p)} |")
        out.append(f"\n## {name} — `{cls.__name__}`\n")
        uh = "".join(f" `{c}` |" for c in (col_labels or []))
        out.append(f"| preset | layers | d_model | vocab | seq default |"
                   f"{uh} params |")
        out.append("|---" * (6 + len(col_labels or [])) + "|")
        out.extend(rows)
    out.append("""
Notes:
- Aliases share the exact architecture shape of an earlier preset
  (e.g. Kimi K2.5/2.6/2.7 are shape-identical to K2; GLM 5.1 to 5).
- every preset name in this table resolves as `--preset` in the tools
  (predict_step, measure_step, nsys_profile, train_solo, ...); names
  shared by several families qualify as `family:preset` (`gpt2:tiny`).
- Correctness: `python tools/verify/verify_family.py --family <name>`.
""")
    print("\n".join(out))


if __name__ == "__main__":
    main()
