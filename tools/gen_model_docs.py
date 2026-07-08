"""Generate docs/models/<family>.md — a comprehensive, per-family
reference derived entirely from the code (no hand-maintained content):

    python tools/gen_model_docs.py                 # all families, GPU record
    python tools/gen_model_docs.py --family glm52
    python tools/gen_model_docs.py --no-record     # CPU-only (skip kernel seqs)

Per family page:
- dims of the documentation preset (mini where it exists, else tiny) —
  every scalar dims field, so object shapes below are derivable;
- objects per layer kind: field-level tables (name / dtype / shape /
  bytes) for W_i, the saved context A_i, and M_i where the kind has
  one, plus embed/head weight tables and the dW/O sizing rules;
- every task kind: executable, buffer contract (a representative
  lowered task's inputs/outputs/mutates with byte sizes), the forward
  STAGE list with emitted-context fields and the derived recompute
  boundary, and the TRACED kernel-call sequence.

Generation is LIGHT by construction: everything about the
documentation preset (object tables, sizes, contracts) is pure layout
arithmetic — no allocation, no kernels, any scale in milliseconds.
Kernel sequences are dims-invariant per task kind, so they are traced
ONCE at the family's TINY preset (megabyte buffers, one launch per
signature through a recording KernelSet proxy); per-sequence op
counts scale with the microbatch and are labeled as traced.

External families generate identically (plugins load first): add a
family, rerun, get the same page.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from dataflow.training import families as F

REPO = Path(__file__).resolve().parent.parent

# documentation presets (Shein): full-scale where a single GPU can
# still profile the per-task signatures; mini for the 671B-class
# families. External families fall back to {name}_mini, then tiny.
DOC_PRESETS = {
    "llama3": "llama3_8b",
    "olmoe": "olmoe_7b",
    "qwen3": "qwen3_8b",
    "qwen3moe": "qwen3moe_30b",
    "qwen35": "qwen35_9b",
    "qwen35moe": "qwen35moe_35b",
    "dsv3": "dsv3_mini",
    "dsv32": "dsv32_mini",
    "glm52": "glm52_mini",
}
# every page documents the SAME run shape so A_*/M_* tables compare
# across families
DOC_SEQ_LEN = 4096
DOC_BATCH = 16


# ---------------------------------------------------------------- helpers

def _meta_layout(family: str, dims, layer: int):
    """Family -> per-layer M layout (None where the kind has no M).
    The one hand-maintained dispatch in this generator: metadata
    layouts are family vocabulary (docs/extending.md §2)."""
    try:
        if family in ("dsv32",):
            from dataflow.tasks.layouts import dsv32_meta_layout

            return dsv32_meta_layout(dims, dims.kind_of(layer))
        if family in ("glm52",):
            from dataflow.tasks.layouts import glm52_meta_layout

            return glm52_meta_layout(dims, dims.kind_of(layer))
        if family in ("olmoe", "qwen3moe", "qwen35moe", "dsv3"):
            from dataflow.tasks.moe.spec import moe_meta_layout

            kind_of = getattr(dims, "kind_of", lambda i: "moe")
            kind = kind_of(layer)
            if "dense" in str(kind) or str(kind) in ("lin", "full"):
                return None
            return moe_meta_layout(dims, dims.moe)
    except Exception:
        return None
    return None


def field_table(layout, title: str, note: str = "",
                per_token: int | None = None) -> list[str]:
    if layout is None or not layout.fields:
        return []
    head = f"**{title}** — {layout.total_bytes:,} bytes"
    if per_token:
        head += f" = **{layout.total_bytes / per_token:,.1f} bytes/token**"
    if note:
        head += f" ({note})"
    out = [head, "", "| field | dtype | shape | bytes |", "|---|---|---|---|"]
    for f in layout.fields:
        out.append(f"| `{f.name}` | {f.dtype} | {tuple(f.shape)} | {f.nbytes:,} |")
    out.append("")
    return out


class RecordingKernels:
    """Proxy over the resolved KernelSet: logs op names per task key."""

    def __init__(self, real):
        self._real = real
        self.log: dict[str, list[str]] = {}
        self.current: str | None = None

    def __getattr__(self, op):
        real_fn = getattr(self._real, op)
        if not callable(real_fn):
            return real_fn

        def wrapper(*a, **k):
            if self.current is not None:
                self.log.setdefault(self.current, []).append(op)
            return real_fn(*a, **k)

        return wrapper


def compress(seq: list[str]) -> str:
    out, i = [], 0
    while i < len(seq):
        j = i
        while j < len(seq) and seq[j] == seq[i]:
            j += 1
        out.append(seq[i] if j - i == 1 else f"{seq[i]}×{j - i}")
        i = j
    return " → ".join(out)


def record_kernel_seqs(fam, cfg, dims, prog) -> dict[str, str]:
    """Execute each task signature once (profiler machinery) through a
    recording kernel proxy; return {compute_key: op sequence}."""
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.tasks.kernels import resolve_kernels
    from dataflow.training.profiling import profile_program

    proxy = RecordingKernels(resolve_kernels())
    base = fam.build_resolver(dims, kernels=proxy)
    seen: set[str] = set()

    class Tag:
        def __init__(self, inner, key):
            self.inner, self.key = inner, key
            # profiling hooks (profile_fill) must remain reachable
            if hasattr(inner, "profile_fill"):
                self.profile_fill = inner.profile_fill

        def launch(self, ctx):
            first = self.key not in seen
            seen.add(self.key)
            proxy.current = self.key if first else None
            try:
                self.inner.launch(ctx)
            finally:
                proxy.current = None

    def resolver(task):
        return Tag(base(task), task.compute_block_key)

    profile_program(prog, resolver, CudaBackend(),
                    warmup=0, repeats=1, soak_seconds=0)
    return {k: compress(v) for k, v in proxy.log.items()}


# ---------------------------------------------------------------- page

def gen_family(name: str, record: bool) -> str:
    fam = F.family(name)
    cls = fam.config_type
    preset = DOC_PRESETS.get(name)
    if preset is None or not isinstance(cls.__dict__.get(preset), classmethod):
        for cand in (f"{name}_mini", "tiny"):
            if isinstance(cls.__dict__.get(cand), classmethod):
                preset = cand
                break
    cfg = dataclasses.replace(getattr(cls, preset)(),
                              seq_len=DOC_SEQ_LEN, batch=DOC_BATCH)
    dims = fam.dims_of(cfg)
    prog = fam.lower(cfg)
    resolver = fam.build_resolver(dims)
    sizes = prog.object_sizes()
    tasks = prog.task_by_id()

    kern_seqs: dict[str, str] = {}
    rec_note = ""
    if record:
        try:
            # trace at TINY dims: sequences are dims-invariant per task
            # kind; only the trace should ever touch the GPU
            tiny_cfg = cls.tiny()
            tiny_dims = fam.dims_of(tiny_cfg)
            kern_seqs = record_kernel_seqs(
                fam, tiny_cfg, tiny_dims, fam.lower(tiny_cfg))
        except Exception as exc:
            rec_note = f" (kernel tracing unavailable: {exc})"

    L = cfg.n_layers
    kind_of = getattr(dims, "kind_of", lambda i: "block")
    kinds_seq = [str(kind_of(i)) for i in range(L)]
    rep_layer = {k: kinds_seq.index(k) for k in dict.fromkeys(kinds_seq)}

    out = [f"# {name}: tasks, objects, kernels",
           "",
           f"GENERATED from `{cls.__name__}.{preset}()` at the standard "
           f"documentation run shape (seq {DOC_SEQ_LEN} × microbatch "
           f"{DOC_BATCH}) — regenerate with "
           f"`python tools/gen_model_docs.py --family {name}`. Presets: "
           f"[builtin_models.md](../builtin_models.md); task-kind fleet "
           f"index: [task_kinds.md](../task_kinds.md).",
           "",
           f"Layer kinds ({L} layers): `{' '.join(kinds_seq)}`",
           "",
           f"**Run shape of this documentation preset**: microbatch "
           f"{cfg.batch} × seq_len {cfg.seq_len} = **{dims.tokens:,} "
           f"tokens per round** (× {cfg.grad_accum_rounds} grad-accum "
           f"round(s) per step). `A_*`/`M_*` objects are sized per "
           f"round; their bytes/token figures below transfer to any "
           f"run shape.",
           "",
           "## Dims (documentation preset)",
           "",
           "| field | value |", "|---|---|"]
    for f in dataclasses.fields(dims):
        v = getattr(dims, f.name)
        if isinstance(v, (int, float, str, bool)):
            out.append(f"| `{f.name}` | {v} |")
        elif isinstance(v, tuple) and len(v) <= 8 and all(
                isinstance(x, (int, str)) for x in v):
            out.append(f"| `{f.name}` | {v} |")
    out.append("")

    # ---- objects per layer kind (collect summary rows as we go) ----
    summary: list[tuple[str, str, int]] = []   # (object, per, bytes)
    detail = ["## Objects, per layer kind", "",
              "`dW_i` mirrors `W_i`'s fields at the grad dtypes; `O_i` holds "
              "the optimizer policy's state slots per field (adamw default: "
              "`m_f`+`v_f` at the opt dtype; sgd fields contribute none — "
              "see extending.md §6). `A_i`/`M_i` exist per (step, round).", ""]
    out_main = out
    out = detail
    for kind, layer in rep_layer.items():
        fwd_key = None
        for t in tasks.values():
            if t.compute_block_key.endswith("_fwd") and \
                    t.block_params.get("layer") == layer and \
                    t.id.startswith("block_fwd"):
                fwd_key = t
                break
        if fwd_key is None:
            continue
        ex = resolver(fwd_key)
        out.append(f"### kind `{kind}` (e.g. layer {layer})")
        out.append("")
        wl = ex._weight_layout(layer) if hasattr(ex, "_weight_layout") else None
        out += field_table(wl, f"`W_{layer}` weights")
        cl = getattr(ex, "cl", None)
        out += field_table(cl, f"`A_.._{layer}` saved context",
                           "per (step, round)", per_token=dims.tokens)
        ml = _meta_layout(name, dims, layer)
        out += field_table(ml, f"`M_.._{layer}` metadata",
                           "never recomputed", per_token=dims.tokens)
        if wl is not None:
            from dataflow.tasks.layouts import grad_layout, opt_state_layout

            summary.append((f"W_i ({kind})", "layer", wl.total_bytes))
            summary.append((f"dW_i ({kind})", "layer/step",
                            grad_layout(wl, dims.dtypes, layer=layer)
                            .total_bytes))
            summary.append((f"O_i ({kind})", "layer",
                            opt_state_layout(wl, dims.dtypes, layer=layer,
                                             opt_policy=getattr(
                                                 dims, "opt_policy", None))
                            .total_bytes))
        if cl is not None:
            summary.append((f"A ({kind})", "layer × round", cl.total_bytes))
        if ml is not None and ml.fields:
            summary.append((f"M ({kind})", "layer × round", ml.total_bytes))

    # embed/head weights
    try:
        from dataflow.tasks.layouts import head_weight_layout
        hl = head_weight_layout(dims)
        out += field_table(hl, "`W_head`")
        summary.append(("W_head", "run", hl.total_bytes))
    except Exception:
        pass
    for oid in ("W_embed", "O_embed", "O_head"):
        if oid in sizes:
            summary.append((oid, "run", sizes[oid]))
    summary.append(("hidden state (y)", "boundary buffer",
                    dims.tokens * dims.d_model * 2))

    out = out_main
    out += ["## Object summary", "",
            f"At the documentation run shape ({dims.tokens:,} "
            f"tokens/round). Details per kind below.", "",
            "| object | scope | bytes | bytes/token |", "|---|---|---|---|"]
    for label, per, b in summary:
        pt = (f"{b / dims.tokens:,.1f}"
              if ("round" in per or "boundary" in per) else "—")
        out.append(f"| `{label}` | {per} | {b:,} | {pt} |")
    out.append("")
    out += detail

    # ---- tasks ----
    out += ["## Tasks", ""]
    seen_keys: set[str] = set()
    for t in tasks.values():
        ck = t.compute_block_key
        if ck in seen_keys:
            continue
        seen_keys.add(ck)
        ex = resolver(t)
        out.append(f"### `{ck}` — `{type(ex).__name__}`")
        out.append("")
        ins = ", ".join(f"`{i}` ({sizes.get(i, 0):,}B)" for i in t.inputs)
        outs = ", ".join(f"`{o.id}` ({o.size_bytes:,}B)" for o in t.outputs)
        muts = ", ".join(f"`{m}`" for m in t.mutates) or "—"
        out += [f"- example task: `{t.id}`",
                f"- inputs: {ins or '—'}",
                f"- outputs: {outs or '—'}",
                f"- mutates: {muts}"]
        if hasattr(ex, "STAGES"):
            rc_n = ex.recompute_stage_count() if hasattr(
                ex, "recompute_stage_count") else None
            out += ["- stages (name — emitted ctx fields):"]
            for si, st in enumerate(ex.STAGES):
                emits = ", ".join(st[2]) if len(st) > 2 and st[2] else "—"
                meta = " [meta: never recomputed]" if len(st) > 3 else ""
                marker = ""
                if rc_n is not None and si == rc_n - 1:
                    marker = "  ← derived recompute boundary"
                out.append(f"    {si}. `{st[0]}` — {emits}{meta}{marker}")
        if ck in kern_seqs:
            out.append(f"- kernel calls (traced once at tiny dims; "
                       f"per-sequence op counts scale with microbatch): "
                       f"{kern_seqs[ck]}")
        out.append("")

    if rec_note:
        out.append(f"_Note: {rec_note.strip()}_")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default=None)
    ap.add_argument("--record", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="execute each task once to record kernel calls "
                         "(needs a CUDA device)")
    ap.add_argument("--out-dir", default=str(REPO / "docs/models"))
    args = ap.parse_args()

    F.load_plugins()
    names = [args.family] if args.family else list(F._FAMILIES)
    outd = Path(args.out_dir)
    outd.mkdir(parents=True, exist_ok=True)
    for name in names:
        page = gen_family(name, args.record)
        (outd / f"{name}.md").write_text(page)
        print(f"wrote {outd}/{name}.md", file=sys.stderr)
    index = ["# Model family references",
             "",
             "GENERATED — `python tools/gen_model_docs.py` regenerates "
             "every page (add `--family X` for one; `--no-record` skips "
             "the measured kernel sequences on CPU-only machines). New "
             "families — builtin or plugin — get the same page "
             "automatically.",
             ""]
    for name in F._FAMILIES:
        index.append(f"- [{name}]({name}.md)")
    (outd / "README.md").write_text("\n".join(index) + "\n")
    print(f"wrote {outd}/README.md", file=sys.stderr)


if __name__ == "__main__":
    main()
