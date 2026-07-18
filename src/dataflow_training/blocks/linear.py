"""The linear seam: dispatch for every matrix-parameter application.

Blocks never call raw GEMM entry points on weight fields; they compose the
per-field triple built here —

    LinearFwd    y  = x @ W      (optionally fused residual add / out=)
    LinearDGrad  dx = dy @ W.T   (optionally out= / in-place accumulate)
    LinearWGrad  dW = x.T @ dy   (value handed to the grad writer ``acc``)

— constructed from (field name, layer, the dims' DTypePolicy). Construction
resolves the field's ``ParamDTypes`` and picks a ROUTE once:

- ``param == "bf16"`` (the system convention; activations are bf16): the
  direct torch/cuBLAS lane below, whose calls are byte-identical to the
  historical inline ``@`` / ``torch.matmul(out=)`` / ``torch.addmm(out=)`` /
  ``.addmm_`` forms — introducing the seam changes no kernel entry point,
  no argument, no numeric.
- any other param dtype: the pinned kernel registry, op names
  ``linear_fwd_{param}`` / ``linear_dgrad_{param}`` / ``linear_wgrad_{param}``.
  Nothing is registered today, so the triple binds a ``MissingLane`` that
  raises ON FIRST USE naming the op to register (construction never raises:
  a policy may type fields the block never applies as a GEMM).

This is where future dtype/quantization lanes plug in with ZERO block
changes: register implementations for those op names (kernels/registry.py
ABI, signatures below) and select the dtype per field in the config's
DTypePolicy. When a lane needs richer keys (activation or grad-storage
dtypes — grad storage today is the ``acc`` writer's cast-on-store concern),
the op-name scheme grows suffixes rather than the blocks growing branches.

Route ABI (what a registered implementation signs, kctx first per the
registry contract; ``w`` is the RAW storage view, ``transposed`` marks the
(out, in)-stored tables like the LM head):

    linear_fwd_*    fn(kctx, x, w, transposed, out, add) -> y
    linear_dgrad_*  fn(kctx, dy, w, transposed, out, into) -> dx
    linear_wgrad_*  fn(kctx, x, dy, transposed) -> dW value

Outside the seam by design: embedding lookup/scatter (not a GEMM), norm
weights (rmsnorm kernel family), and the MoE routed expert path (grouped
GEMM registry kernels with their own dispatch shape).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .layouts import DTypePolicy, ParamDTypes

LINEAR_PARAM_DEFAULT = "bf16"  # the direct torch/cuBLAS lane's param dtype


# --- default lane: byte-identical to the historical inline calls -----------------


def torch_gemm_fwd(kctx, x, w, transposed, out, add):
    """cuBLAS-via-torch forward. Branches mirror the exact pre-seam forms:
    plain ``x @ w``, write-through ``torch.matmul(out=)``, fused residual
    ``torch.addmm(add, x, w, out=)``, and the scratch-path ``add + x @ w``."""
    wt = w.T if transposed else w
    if add is not None:
        if out is not None:
            return torch.addmm(add, x, wt, out=out)
        return add + x @ wt
    if out is not None:
        return torch.matmul(x, wt, out=out)
    return x @ wt


def torch_gemm_dgrad(kctx, dy, w, transposed, out, into):
    """cuBLAS-via-torch data grad: ``dy @ w.T`` (``dy @ w`` for transposed
    storage), with the pre-seam write-through (``out=``) and in-place
    accumulate (``into.addmm_``) forms."""
    wt = w if transposed else w.T
    if into is not None:
        return into.addmm_(dy, wt)
    if out is not None:
        return torch.matmul(dy, wt, out=out)
    return dy @ wt


def torch_gemm_wgrad(kctx, x, dy, transposed):
    """cuBLAS-via-torch weight grad VALUE: ``x.T @ dy`` at the operand
    dtypes (``dy.T @ x`` for transposed storage). Storage dtype/accumulate
    stay the caller's ``acc`` writer's concern — unchanged contract."""
    if transposed:
        return dy.T @ x
    return x.T @ dy


DEFAULT_ROUTES = {
    "fwd": torch_gemm_fwd,
    "dgrad": torch_gemm_dgrad,
    "wgrad": torch_gemm_wgrad,
}


# --- non-default lanes: pinned registry kernels ----------------------------------


@dataclass(frozen=True)
class MissingLane:
    """Callable placeholder bound when a field's dtype policy leaves the
    default lane but no kernel is registered for the derived op. Raises on
    FIRST USE (not at construction: policies may type fields a block never
    applies as a GEMM) with the exact registration needed."""

    op: str
    field: str
    dtypes: ParamDTypes

    def __call__(self, kctx, *args, **kwargs):
        raise NotImplementedError(
            f"no linear lane for field {self.field!r} at param dtype "
            f"{self.dtypes.param!r}: register a kernel implementation for "
            f"op {self.op!r} (dataflow_training.kernels.register, route ABI "
            f"in blocks/linear.py) and re-resolve the kernel set"
        )


def resolve_route(kind: str, field: str, dts: ParamDTypes, kernels) -> Callable:
    """Pick the route for one triple member, ONCE, at construction: the
    default torch/cuBLAS lane for bf16 params, else the pinned registry
    kernel ``linear_{kind}_{param}`` (MissingLane when unregistered)."""
    if dts.param == LINEAR_PARAM_DEFAULT:
        return DEFAULT_ROUTES[kind]
    op = f"linear_{kind}_{dts.param}"
    fn = getattr(kernels, op, None)
    if fn is None:
        return MissingLane(op=op, field=field, dtypes=dts)
    return fn


# --- the triple ------------------------------------------------------------------


@dataclass(frozen=True)
class LinearFwd:
    """Forward application of one weight field: y = x @ W (x @ W.T for
    transposed storage). ``out=`` writes through to an engine-owned view;
    ``add=`` fuses the residual (addmm) when ``out=`` is present and keeps
    the historical two-op ``add + x @ w`` form when it is not."""

    field: str
    route: Callable
    transposed: bool = False

    def __call__(self, kctx, x, w, out=None, add=None):
        return self.route(kctx, x, w[self.field], self.transposed, out, add)


@dataclass(frozen=True)
class LinearDGrad:
    """Data gradient through one weight field: dx = dy @ W.T (dy @ W for
    transposed storage). ``into=`` accumulates in place (``addmm_``, the
    residual-stream join convention); ``out=`` writes through."""

    field: str
    route: Callable
    transposed: bool = False

    def __call__(self, kctx, dy, w, out=None, into=None):
        return self.route(kctx, dy, w[self.field], self.transposed, out, into)


@dataclass(frozen=True)
class LinearWGrad:
    """Weight-gradient VALUE for one field: x.T @ dy (dy.T @ x for
    transposed storage). Callers hand the value to the grad writer
    (``acc``), which owns create-vs-accumulate and the storage-dtype cast;
    guard expensive calls with ``acc.wanted(field)`` as before."""

    field: str
    route: Callable
    transposed: bool = False

    def __call__(self, kctx, x, dy):
        return self.route(kctx, x, dy, self.transposed)


@dataclass(frozen=True)
class LinearTriple:
    """The three routed applications of one matrix-parameter field."""

    field: str
    dtypes: ParamDTypes
    fwd: LinearFwd
    dgrad: LinearDGrad
    wgrad: LinearWGrad


def resolve_linear(
    field: str,
    layer: int | None,
    policy: DTypePolicy,
    kernels,
    transposed: bool = False,
    ns: str | None = None,
) -> LinearTriple:
    """Build one field's triple: resolve the field's ParamDTypes from the
    policy (``ns`` prefixes the POLICY LOOKUP for the loose tables —
    "head.w" — exactly as the layouts do; ``layer`` selects depth-dependent
    sub-policies) and bind each member's route."""
    name = f"{ns}.{field}" if ns else field
    dts = policy.for_field(name, layer)
    return LinearTriple(
        field=field,
        dtypes=dts,
        fwd=LinearFwd(field, resolve_route("fwd", field, dts, kernels), transposed),
        dgrad=LinearDGrad(field, resolve_route("dgrad", field, dts, kernels), transposed),
        wgrad=LinearWGrad(field, resolve_route("wgrad", field, dts, kernels), transposed),
    )


def resolve_linears(
    fields: tuple,
    layer: int | None,
    policy: DTypePolicy,
    kernels,
) -> dict:
    """Triples for a block's matrix-parameter fields, keyed by field name —
    what executables stash as ``st["lin"]`` / ``a["lin"]`` for the stages."""
    return {f: resolve_linear(f, layer, policy, kernels) for f in fields}
