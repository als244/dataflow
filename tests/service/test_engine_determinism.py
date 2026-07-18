"""The one exact-drift tripwire kept from the golden era.

Within ONE daemon process, re-initializing and re-running identical
steps must reproduce losses BITWISE — kernels, plan, and update math
are all deterministic in-process (cross-PROCESS runs are not: kernel
algorithm selection varies, measured ~5e-5; that is why this gate is
same-daemon and why the parity gate uses envelopes). A kernel or
engine change that shifts numerics at all trips this before the
envelope gates can notice.
"""
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from dataflow.core.jsonio import program_to_dict  # noqa: E402
from dataflow_training.run.presets import (  # noqa: E402
    cfg_dict,
    resolver_family,
    smoke_preset,
)
from dataflow_training.run.driver import daemon_client, init_model  # noqa: E402
from dataflow_training.data.fineweb import make_stream  # noqa: E402
from dataflow_training.run.recipe import Recipe  # noqa: E402
from dataflow_training.model_families.families import resolve_family  # noqa: E402
from dataflow_training.lowering.planning import plan_program  # noqa: E402

STEPS = 3


def run_steps(client, cfg, prog_id, stream) -> list:
    losses = []
    overflows = []
    fetch = [f"loss_0_{r}" for r in range(cfg.grad_accum_rounds)]
    for step in range(STEPS):
        valid = 0
        for r in range(cfg.grad_accum_rounds):
            tok, tgt = stream(step * cfg.grad_accum_rounds + r)
            valid += int((tgt >= 0).sum())
            client.put_object(f"tokens_0_{r}", tok.numpy().tobytes())
            client.put_object(f"targets_0_{r}", tgt.numpy().tobytes())
        out = client.run(prog_id,
                         args={"step": step, "valid_rows": valid},
                         fetch=fetch)
        assert out.get("state") == "done", (step, out)
        losses.append(sum(out["fetched"][k] for k in fetch))
        overflows.append(out.get("slab_overflows"))
    # steady state must be overflow-free: step 0 may escape to the vendor
    # allocator while pools warm; later steps reuse those buffers
    if all(n is not None for n in overflows):
        assert all(n == 0 for n in overflows[1:]), overflows
    return losses


def test_same_daemon_rerun_bitwise(tmp_path):
    cfg = smoke_preset()
    recipe = Recipe(peak_lr=3e-4, min_lr=3e-5, warmup_steps=1,
                    total_steps=STEPS)
    fam = resolve_family(cfg)
    planned = plan_program(fam.lower(cfg),
                           fast_memory_capacity=4 << 30)
    cd = cfg_dict(cfg)
    fam_name = resolver_family(cfg)
    with daemon_client(slab_gib=4.0, log=print) as client:
        init_model(client, fam_name, cd, seed=11)
        stream = make_stream(cfg.tokens)
        for r in range(cfg.grad_accum_rounds):
            tok, tgt = stream(r)
            client.put_object(f"tokens_0_{r}", tok.numpy().tobytes())
            client.put_object(f"targets_0_{r}", tgt.numpy().tobytes())
        reg = client.register_program(
            program_to_dict(planned.program),
            resolver={"kind": "model_family", "family": fam_name, "cfg": cd,
                      "hyper": recipe.hyper_spec()})
        assert not reg["bindings"]["missing_inputs"]
        first = run_steps(client, cfg, reg["prog_id"],
                          make_stream(cfg.tokens))
        # re-init: same seed, same bytes
        init_model(client, fam_name, cd, seed=11)
        second = run_steps(client, cfg, reg["prog_id"],
                           make_stream(cfg.tokens))
    assert first == second, (
        f"same-daemon rerun diverged: {first} vs {second} — an "
        f"engine/kernel change shifted numerics (in-process runs are "
        f"deterministic; this is the exact-drift tripwire)")
