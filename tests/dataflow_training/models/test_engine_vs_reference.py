"""THE correctness gate: engine (service path) vs the isolated
pure-torch reference twin, every registered family, real fineweb data.

Two input modes per family:
- uniform: the family's smoke preset trained N steps by BOTH the
  daemon engine (run_engine — the production path) and the twin
  (run_reference), identical bytes in, per-step losses compared.
- ragged: one grad-accum round of MIXED-LENGTH sequences; the engine
  consumes them packed via the per-round ``seq_lens`` wire contract;
  the twin runs each sequence independently and combines with the
  global denominator (exact — attention never crosses sequences).

Bands are measured, not guessed: run with DATAFLOW_CALIBRATE=1 to
print observed deltas on this machine instead of asserting; the
pinned bands cover both fleet architectures (sm86/sm120) with
headroom. Step-0 compares at the forward-noise floor; the trajectory
band absorbs bf16 accumulation-order drift and (for MoE) top-k
routing flips.
"""
import math
import os

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from dataflow_training.run import presets as P  # noqa: E402
from dataflow_training.run.driver import (  # noqa: E402
    daemon_client,
    init_model,
    run_engine,
    run_reference,
)
from dataflow_training.data.pipeline import legacy_block_pipeline  # noqa: E402
from dataflow_training.run.recipe import Recipe  # noqa: E402

STEPS = 6
CALIBRATE = bool(os.environ.get("DATAFLOW_CALIBRATE"))

FAMILY_PRESETS = {
    "gpt2": "gpt2_smoke_preset",
    "llama3": "smoke_preset",
    "qwen3": "qwen3_smoke_preset",
    "qwen35": "qwen35_smoke_preset",
    "qwen35moe": "qwen35moe_smoke_preset",
    "olmoe": "olmoe_smoke_preset",
    "qwen3moe": "qwen3moe_smoke_preset",
    "dsv3": "dsv3_smoke_preset",
    "dsv32": "dsv32_smoke_preset",
    "glm52": "glm52_smoke_preset",
}

# (step0_tol, trajectory_tol) per family, BOTH input modes: pinned at
# ~4-5x the measured sm86 floors (refactor findings ledger, A1
# calibration table); re-verified on sm120 at the chicago pull.
# Dense families sit at 1e-4-scale; MoE adds routing discreteness;
# the MLA trio (+DSA for dsv32) is the loosest.
BANDS = {
    "gpt2": (5e-4, 2e-3),
    "llama3": (5e-4, 2e-3),
    "qwen3": (5e-4, 2e-3),
    "qwen35": (8e-4, 2e-3),
    "olmoe": (1e-3, 3e-3),
    "qwen3moe": (1e-3, 3e-3),
    "qwen35moe": (4e-3, 1e-2),
    "dsv3": (8e-3, 2e-2),
    "dsv32": (8e-3, 4e-2),
    "glm52": (8e-3, 2e-2),
}


def quiet_log(msg: str) -> None:
    if CALIBRATE:
        return


def smoke_recipe() -> Recipe:
    return Recipe(peak_lr=3e-4, min_lr=3e-5,
                  warmup_steps=1, total_steps=STEPS)


def compare_curves(name: str, mode: str, ref, eng) -> None:
    assert len(ref) == len(eng) == STEPS, (len(ref), len(eng))
    assert all(math.isfinite(x) for x in eng), (name, mode, eng)
    step0 = abs(ref[0] - eng[0])
    traj = max(abs(a - b) for a, b in zip(ref, eng))
    if CALIBRATE:
        print(f"[calibrate] {name:10s} {mode:8s} "
              f"step0 {step0:.3e}  traj {traj:.3e}")
        return
    tol0, tolT = BANDS[name]
    assert step0 < tol0, (name, mode, "step0", step0, tol0)
    assert traj < tolT, (name, mode, "trajectory", traj, tolT)


@pytest.mark.parametrize("name", sorted(FAMILY_PRESETS))
def test_engine_matches_reference_uniform(name):
    cfg = getattr(P, FAMILY_PRESETS[name])()
    recipe = smoke_recipe()
    ref = run_reference(cfg, recipe, legacy_block_pipeline(cfg),
                        STEPS, log=quiet_log)
    with daemon_client(slab_gib=4.0, log=quiet_log) as client:
        eng = run_engine(client, cfg, recipe,
                         legacy_block_pipeline(cfg), STEPS,
                         budget_gib=4.0, log=quiet_log)
    compare_curves(name, "uniform", ref.losses, eng.losses)


RAGGED_LENGTHS = (37, 256, 128, 91)     # sums to a 512-token round


@pytest.mark.parametrize("name", sorted(FAMILY_PRESETS))
def test_engine_matches_reference_ragged(name):
    from dataclasses import replace

    from dataflow.core.jsonio import program_to_dict
    from dataflow_training.run.presets import cfg_dict, resolver_family
    from dataflow_training.model_families import bridges
    from dataflow_training.run.driver import reference_optimizer
    from dataflow_training.model_families.families import resolve_family
    from dataflow_training.lowering.planning import plan_program

    total = sum(RAGGED_LENGTHS)
    base = getattr(P, FAMILY_PRESETS[name])()
    # one round of `total` tokens per step, expressed to the engine as
    # a single packed row; the seq_lens arg re-segments it
    cfg = replace(base, seq_len=total, batch=1, grad_accum_rounds=1)
    recipe = smoke_recipe()
    from dataflow_training.data.sources.shards import ShardSource

    src = ShardSource(max_seqlen=total, vocab_size=cfg.vocab_size,
                      window=total)
    src_iter = src.sequences(None)
    wins = []
    for _ in range(STEPS):
        seq, _cur = next(src_iter)
        wins.append((torch.from_numpy(seq.tokens),
                     torch.from_numpy(seq.targets)))
    fam = resolve_family(cfg)
    dims = fam.derive_dims(cfg)

    # ---- reference: per-sequence forward, global denominator -------
    from dataflow.runtime.device.cuda import CudaBackend

    backend = CudaBackend()
    values = fam.initial_values(fam.lower(cfg), cfg, backend, seed=11)
    model = bridges.build_reference_model(cfg)
    bridges.load_reference_init(model, cfg, dims,
                                bridges.get_bytes_from_values(values))
    for buf in values.values():
        backend.free(buf)
    model.train()
    opt = reference_optimizer(model, cfg, recipe)
    ref_losses = []
    for step in range(STEPS):
        tok, tgt = wins[step]
        opt.zero_grad()
        step_valid = int((tgt >= 0).sum())
        lo = 0
        acc = 0.0
        for ln in RAGGED_LENGTHS:
            t = tok[lo:lo + ln].to("cuda").view(1, ln)
            g = tgt[lo:lo + ln].to("cuda").view(1, ln)
            valid = int((g >= 0).sum())
            ce = model.loss(t, g)
            (ce * (valid / step_valid)).backward()
            acc += float(ce.detach()) * (valid / step_valid)
            lo += ln
        opt.step(step)
        ref_losses.append(acc)

    # ---- engine: packed row + per-round seq_lens -------------------
    boundaries = [0]
    for ln in RAGGED_LENGTHS:
        boundaries.append(boundaries[-1] + ln)
    with daemon_client(slab_gib=4.0, log=quiet_log) as client:
        planned = plan_program(fam.lower(cfg),
                               fast_memory_capacity=4 << 30)
        cd = cfg_dict(cfg)
        init_model(client, resolver_family(cfg), cd, seed=11)
        tok0, tgt0 = wins[0]
        client.put_object("tokens_0_0", tok0.numpy().tobytes())
        client.put_object("targets_0_0", tgt0.numpy().tobytes())
        reg = client.register_program(
            program_to_dict(planned.program),
            resolver={"kind": "model_family",
                      "family": resolver_family(cfg), "cfg": cd,
                      "hyper": recipe.hyper_spec()})
        assert not reg["bindings"]["missing_inputs"]
        eng_losses = []
        for step in range(STEPS):
            if step > 0:
                tok, tgt = wins[step]
                client.put_object("tokens_0_0", tok.numpy().tobytes())
                client.put_object("targets_0_0", tgt.numpy().tobytes())
            else:
                tok, tgt = tok0, tgt0
            out = client.run(
                reg["prog_id"],
                args={"step": step,
                      "valid_rows": int((tgt >= 0).sum()),
                      "seq_lens": {"0": boundaries}},
                fetch=["loss_0_0"])
            assert out.get("state") == "done", (name, step, out)
            eng_losses.append(out["fetched"]["loss_0_0"])
    compare_curves(name, "ragged", ref_losses, eng_losses)
