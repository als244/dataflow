"""Common-op kernel collection: one signature per op, swappable backends.

Importing this package registers every builtin implementation (eager torch
fallbacks + fused Triton, one file per op family); ``resolve_kernels()``
pins one implementation per op for the session. See registry.py for the
implementation ABI (stream semantics, tensors, workspace declarations) —
any toolchain that honors it can register (CuTe DSL, TileLang, CUDA C++
bindings, ...).

    from dataflow_training import kernels
    kset = kernels.resolve_kernels()            # pinned; or overrides={...}
    kset.swiglu_bwd(kctx, ds, x1, x3, dx1, dx3)
    kset.describe()  # {"adamw_step": "triton", ...} -> profiles/reports

``DATAFLOW_KERNELS=eager`` forces the eager set process-wide (numerics
bisection / A-B baseline).
"""
from .registry import (  # noqa: F401
    KernelCtx,
    KernelEntry,
    KernelSet,
    arena,
    device_caps,
    internal,
    none,
    register,
    registered,
    resolve_kernels,
)

# registration side effects: one module per op family
# NOTE: adding an op family grows KernelSet.describe() for EVERY resolver,
# which keys the profile disk cache — registering the moe families was a
# one-time cache invalidation for all configs (documented, deliberate;
# import-order-dependent lazy registration would be worse: nondeterministic
# resolution).
from . import (  # noqa: F401,E402
    adamw,
    muon,
    causal_conv,
    cross_entropy,
    dsa,
    dsa_flashmla,
    embed,
    gated_rmsnorm,
    moe_dispatch,
    moe_grouped_gemm,
    moe_router,
    rmsnorm,
    rope,
    swiglu,
)
