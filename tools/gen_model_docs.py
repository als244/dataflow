"""Generate docs/models/<family>/<preset>_<bs>x<seq>.md — per-preset
references derived entirely from the code (no hand-maintained content):

    python tools/gen_model_docs.py                     # every (family, preset)
    python tools/gen_model_docs.py --family glm52
    python tools/gen_model_docs.py --family glm52 --preset glm52_mini
    python tools/gen_model_docs.py --no-record         # CPU-only (skip traces)

Default pages use the STANDARD documentation run shape (microbatch 16 ×
seq 4096 → files named `<preset>_16x4K.md`) so object tables compare
across presets and families. For a page at a DIFFERENT run shape, use
`tools/gen_model_page.py --preset <p> --microbatch B --seq-len S`.

Per page: dims; per-object and aggregate size summaries (dM counts
toward dW — metadata gradients are gradients); per-layer-kind
FIELD-LEVEL tables for W/A/M; every task kind with buffer contract,
forward STAGE list (meta marks + derived recompute boundary), and the
TRACED kernel-call sequence.

Generation is LIGHT by construction: object tables, sizes, and
contracts are pure layout arithmetic — no allocation, no kernels, any
scale in milliseconds. Kernel sequences are dims-invariant per task
kind, so they are traced ONCE PER FAMILY at the tiny preset (megabyte
buffers, one launch per signature through a recording KernelSet proxy)
and shared by that family's pages; per-sequence op counts scale with
the microbatch and are labeled as traced.

External families generate identically (plugins load first).
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from dataflow.training import families as F

REPO = Path(__file__).resolve().parent.parent

DOC_SEQ_LEN = 4096
DOC_BATCH = 16


def shape_tag(batch: int, seq_len: int) -> str:
    seq = f"{seq_len // 1024}K" if seq_len % 1024 == 0 else str(seq_len)
    return f"{batch}x{seq}"


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
    """Proxy over the resolved KernelSet: logs op names per task key.
    While a registry op executes, aten tracing is SUPPRESSED so the
    fused kernel appears once under its own name instead of its
    internal tensor plumbing."""

    def __init__(self, real):
        self._real = real
        self.log: dict[str, list[str]] = {}
        self.current: str | None = None
        self.in_registry_op = False

    def __getattr__(self, op):
        real_fn = getattr(self._real, op)
        if not callable(real_fn):
            return real_fn

        def wrapper(*a, **k):
            if self.current is not None:
                self.log.setdefault(self.current, []).append(op)
            prev = self.in_registry_op
            self.in_registry_op = True
            try:
                return real_fn(*a, **k)
            finally:
                self.in_registry_op = prev

        return wrapper


# aten compute ops worth showing in a task's kernel sequence (plain
# tensor plumbing — views, copies, fills — is deliberately excluded)
_ATEN_TRACE = {
    "mm", "addmm", "bmm", "baddbmm", "matmul", "linear", "einsum",
    "convolution", "conv1d", "scaled_dot_product_attention",
    "_scaled_dot_product_flash_attention",
    "_scaled_dot_product_efficient_attention",
    "_scaled_dot_product_flash_attention_backward",
    "_scaled_dot_product_efficient_attention_backward",
    "_flash_attention_backward", "convolution_backward",
    "embedding_dense_backward",
    "sort", "argsort", "topk", "cumsum", "bincount",
    "index_select", "index_add", "index_add_", "scatter_add",
    "scatter_add_", "index_copy", "index_copy_", "logsumexp", "softmax",
    "_softmax", "embedding",
}


def _aten_trace_mode(proxy):
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    class AtenTrace(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if proxy.current is not None and not proxy.in_registry_op:
                ns = getattr(func, "namespace", "aten")
                name = func.overloadpacket.__name__
                if ns != "aten":
                    proxy.log.setdefault(proxy.current, []).append(
                        f"{ns}::{name}")
                elif name in _ATEN_TRACE:
                    proxy.log.setdefault(proxy.current, []).append(name)
            return func(*args, **kwargs)

    return AtenTrace()


# third-party fused entry points invoked DIRECTLY by blocks (raw
# triton — invisible to both the registry proxy and aten dispatch).
# The second and last hand-maintained map in this generator; the
# qwen35 module header documents these direct-call contracts.
_DIRECT_TRACE = (
    ("fla.ops.gated_delta_rule.chunk", "chunk_gated_delta_rule_fwd"),
    ("fla.ops.gated_delta_rule.chunk", "chunk_gated_delta_rule_bwd"),
    ("fla.modules.l2norm", "l2norm_fwd"),
    ("fla.modules.l2norm", "l2norm_bwd"),
)


def _patch_direct_calls(proxy):
    import contextlib
    import importlib

    @contextlib.contextmanager
    def cm():
        undo = []
        for mod_name, fn_name in _DIRECT_TRACE:
            try:
                mod = importlib.import_module(mod_name)
                real = getattr(mod, fn_name)
            except Exception:
                continue

            def make(real, label):
                def wrapper(*a, **k):
                    if proxy.current is not None and \
                            not proxy.in_registry_op:
                        proxy.log.setdefault(proxy.current, []).append(
                            label)
                    return real(*a, **k)
                return wrapper

            setattr(mod, fn_name, make(real, f"fla::{fn_name}"))
            undo.append((mod, fn_name, real))
        try:
            yield
        finally:
            for mod, fn_name, real in undo:
                setattr(mod, fn_name, real)

    return cm()


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
    """Execute each task signature once (profiler machinery, tiny dims)
    through a recording kernel proxy; {compute_key: op sequence}."""
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.tasks.kernels import resolve_kernels
    from dataflow.training.profiling import profile_program

    proxy = RecordingKernels(resolve_kernels())
    base = fam.build_resolver(dims, kernels=proxy)
    seen: set[str] = set()

    class Tag:
        def __init__(self, inner, key):
            self.inner, self.key = inner, key
            if hasattr(inner, "profile_fill"):
                self.profile_fill = inner.profile_fill

        def launch(self, ctx):
            first = self.key not in seen
            seen.add(self.key)
            proxy.current = self.key if first else None
            try:
                if proxy.current is not None:
                    with _aten_trace_mode(proxy):
                        self.inner.launch(ctx)
                else:
                    self.inner.launch(ctx)
            finally:
                proxy.current = None

    def resolver(task):
        return Tag(base(task), task.compute_block_key)

    with _patch_direct_calls(proxy):
        profile_program(prog, resolver, CudaBackend(),
                        warmup=0, repeats=1, soak_seconds=0)
    return {k: compress(v) for k, v in proxy.log.items()}


# ---------------------------------------------------------------- page

def gen_page(name: str, preset: str, record: bool,
             kern_cache: dict | None = None, *,
             batch: int = DOC_BATCH, seq_len: int = DOC_SEQ_LEN) -> str:
    fam = F.family(name)
    cls = fam.config_type
    cfg = dataclasses.replace(getattr(cls, preset)(),
                              seq_len=seq_len, batch=batch)
    dims = fam.dims_of(cfg)
    prog = fam.lower(cfg)
    resolver = fam.build_resolver(dims)
    sizes = prog.object_sizes()
    tasks = prog.task_by_id()
    tag = shape_tag(batch, seq_len)

    kern_seqs: dict[str, str] = {}
    rec_note = ""
    if record:
        if kern_cache is not None and name in kern_cache:
            kern_seqs = kern_cache[name]
        else:
            try:
                # trace at TINY dims: sequences are dims-invariant per
                # task kind; one trace per family, shared by its pages
                tiny_cfg = cls.tiny()
                tiny_dims = fam.dims_of(tiny_cfg)
                kern_seqs = record_kernel_seqs(
                    fam, tiny_cfg, tiny_dims, fam.lower(tiny_cfg))
            except Exception as exc:
                rec_note = f"kernel tracing unavailable: {exc}"
            if kern_cache is not None:
                kern_cache[name] = kern_seqs

    L = cfg.n_layers
    kind_of = getattr(dims, "kind_of", lambda i: "block")
    kinds_seq = [str(kind_of(i)) for i in range(L)]
    rep_layer = {k: kinds_seq.index(k) for k in dict.fromkeys(kinds_seq)}

    out = [f"# {name} / `{preset}` @ {tag}: tasks, objects, kernels",
           "",
           f"GENERATED from `{cls.__name__}.{preset}()` at run shape "
           f"microbatch {batch} × seq {seq_len} — regenerate with "
           f"`python tools/gen_model_page.py --preset {preset} "
           f"--microbatch {batch} --seq-len {seq_len}`. All presets: "
           f"[builtin_models.md](../../builtin_models.md); task-kind "
           f"fleet index: [task_kinds.md](../../task_kinds.md).",
           "",
           f"Layer kinds ({L} layers): `{' '.join(kinds_seq)}`",
           "",
           f"**Run shape**: microbatch {cfg.batch} × seq_len "
           f"{cfg.seq_len} = **{dims.tokens:,} tokens per round** "
           f"(× {cfg.grad_accum_rounds} grad-accum round(s) per step). "
           f"`A_*`/`M_*` objects are sized per round; bytes/token "
           f"figures transfer to any run shape.",
           ""]

    # ---- objects per layer kind (collect summary rows as we go) ----
    summary: list[tuple[str, str, int]] = []
    detail = ["## Objects, per layer kind", "",
              "`dW_i` mirrors `W_i`'s fields at the grad dtypes; `O_i` "
              "holds the optimizer policy's state slots per field (adamw "
              "default: `m_f`+`v_f` at the opt dtype; sgd fields "
              "contribute none — see extending.md §6). `A_i`/`M_i` exist "
              "per (step, round).", ""]
    from dataflow.tasks.layouts import grad_layout, opt_state_layout

    for kind, layer in rep_layer.items():
        fwd_task = None
        for t in tasks.values():
            if t.id.startswith("block_fwd") and \
                    t.block_params.get("layer") == layer:
                fwd_task = t
                break
        if fwd_task is None:
            continue
        ex = resolver(fwd_task)
        detail.append(f"### kind `{kind}` (e.g. layer {layer})")
        detail.append("")
        wl = ex._weight_layout(layer) if hasattr(ex, "_weight_layout") else None
        detail += field_table(wl, f"`W_{layer}` weights")
        cl = getattr(ex, "cl", None)
        detail += field_table(cl, f"`A_.._{layer}` saved context",
                              "per (step, round)", per_token=dims.tokens)
        ml = _meta_layout(name, dims, layer)
        detail += field_table(ml, f"`M_.._{layer}` metadata",
                              "never recomputed", per_token=dims.tokens)
        if wl is not None:
            summary.append((f"W_i ({kind})", "layer", wl.total_bytes))
            summary.append((f"dW_i ({kind})", "layer/step",
                            grad_layout(wl, dims.dtypes, layer=layer)
                            .total_bytes))
            summary.append((f"O_i ({kind})", "layer",
                            opt_state_layout(
                                wl, dims.dtypes, layer=layer,
                                opt_policy=getattr(dims, "opt_policy",
                                                   None)).total_bytes))
        if cl is not None:
            summary.append((f"A ({kind})", "layer × round", cl.total_bytes))
        if ml is not None and ml.fields:
            summary.append((f"M ({kind})", "layer × round", ml.total_bytes))

    try:
        from dataflow.tasks.layouts import head_weight_layout

        hl = head_weight_layout(dims)
        detail += field_table(hl, "`W_head`")
        summary.append(("W_head", "run", hl.total_bytes))
    except Exception:
        pass
    for oid in ("W_embed", "O_embed", "O_head"):
        if oid in sizes:
            summary.append((oid, "run", sizes[oid]))
    summary.append(("hidden state (y)", "boundary buffer",
                    dims.tokens * dims.d_model * 2))

    out += ["## Object summary", "",
            f"At this run shape ({dims.tokens:,} tokens/round). "
            f"Token-scaled objects show bytes/token in parens. Details "
            f"per kind below.", "",
            "| object | scope | bytes |", "|---|---|---|"]
    for label, per, b in summary:
        cell = f"{b:,}"
        if "round" in per or "boundary" in per:
            cell += f" ({b / dims.tokens:,.1f}/token)"
        out.append(f"| `{label}` | {per} | {cell} |")
    out.append("")

    # ---- aggregate totals by type (dM counts toward dW: gradients) ----
    agg: dict[str, tuple[int, int]] = {}

    def add(g, b):
        n, tot = agg.get(g, (0, 0))
        agg[g] = (n + 1, tot + b)

    for oid, b in sizes.items():
        if oid.startswith(("dW_", "dM_")):
            add("dW", b)
        elif oid.startswith("W_"):
            add("W", b)
        elif oid.startswith("O_"):
            add("O", b)
        elif oid.startswith("A_"):
            add("A", b)
        elif oid.startswith("M_"):
            add("M", b)
    out += ["### Aggregate totals (all layers, this run shape)", "",
            "| type | objects | total bytes |", "|---|---|---|"]
    label_of = {
        "W": "W (all weights, incl. embed/head)",
        "dW": "dW (all gradients, incl. metadata grads, per step)",
        "O": "O (all optimizer state)",
        "A": "A (all saved contexts, one round)",
        "M": "M (all metadata, one round)",
    }
    for g in ("W", "dW", "O", "A", "M"):
        if g not in agg:
            continue
        n, tot = agg[g]
        cell = f"{tot:,}"
        if g in ("A", "M"):
            cell += f" ({tot / dims.tokens:,.1f}/token)"
        out.append(f"| {label_of[g]} | {n} | {cell} |")
    out.append("")

    # ---- dims table ----
    out += ["## Dims", "", "| field | value |", "|---|---|"]
    for f in dataclasses.fields(dims):
        v = getattr(dims, f.name)
        if isinstance(v, (int, float, str, bool)):
            out.append(f"| `{f.name}` | {v} |")
        elif isinstance(v, tuple) and len(v) <= 8 and all(
                isinstance(x, (int, str)) for x in v):
            out.append(f"| `{f.name}` | {v} |")
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
        out.append(f"_Note: {rec_note}_")
    return "\n".join(out) + "\n"


def presets_of(cls) -> list[str]:
    out = []
    for n in dir(cls):
        if n.startswith("_") or not isinstance(cls.__dict__.get(n), classmethod):
            continue
        try:
            getattr(cls, n)()
        except TypeError:
            continue
        out.append(n)
    return sorted(out)


def family_of_preset(preset: str) -> str | None:
    """Resolve a preset classmethod name to its family (None if
    ambiguous, e.g. 'tiny')."""
    hits = [n for n in F._FAMILIES
            if isinstance(F.family(n).config_type.__dict__.get(preset),
                          classmethod)]
    return hits[0] if len(hits) == 1 else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default=None)
    ap.add_argument("--preset", default=None)
    ap.add_argument("--record", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="trace each task kind once at tiny dims "
                         "(needs a CUDA device)")
    ap.add_argument("--out-dir", default=str(REPO / "docs/models"))
    args = ap.parse_args()

    F.load_plugins()
    names = [args.family] if args.family else list(F._FAMILIES)
    outd = Path(args.out_dir)
    tag = shape_tag(DOC_BATCH, DOC_SEQ_LEN)
    kern_cache: dict = {}
    index = ["# Model references — one page per (family, preset)",
             "",
             f"GENERATED — `python tools/gen_model_docs.py` regenerates "
             f"everything (`--family X [--preset P]` narrows; "
             f"`--no-record` skips kernel tracing on CPU-only machines). "
             f"Default pages use the standard documentation run shape "
             f"(microbatch {DOC_BATCH} × seq {DOC_SEQ_LEN} — the "
             f"`_{tag}` filename suffix); pages at other shapes: "
             f"`tools/gen_model_page.py`. New families — builtin or "
             f"plugin — appear automatically.",
             ""]
    for name in names:
        cls = F.family(name).config_type
        presets = [args.preset] if args.preset else presets_of(cls)
        fdir = outd / name
        fdir.mkdir(parents=True, exist_ok=True)
        index.append(
            f"- **{name}**: "
            + " · ".join(f"[{p}]({name}/{p}_{tag}.md)" for p in presets))
        for preset in presets:
            page = gen_page(name, preset, args.record, kern_cache)
            (fdir / f"{preset}_{tag}.md").write_text(page)
            print(f"wrote {fdir}/{preset}_{tag}.md", file=sys.stderr)
    if not args.family:
        (outd / "README.md").write_text("\n".join(index) + "\n")
        print(f"wrote {outd}/README.md", file=sys.stderr)


if __name__ == "__main__":
    main()
