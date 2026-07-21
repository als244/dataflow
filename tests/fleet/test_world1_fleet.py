"""World-1 fleet gate: the conductor at world 1 — zero-config local
topology, child-process daemon, NO peer group, the SOLO program (no
annotation) — must reproduce a plain run_engine training run on the
same config/data/seed within the cross-process ambient envelope.

This is the merge's keystone equivalence: solo really is the world-1
special case of fleet, isolated from daemon-placement effects (both
sides run the same engine code; the process boundary is exactly what
the ambient envelope certifies elsewhere)."""
import math

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

from dataflow_training.data.pipeline import legacy_block_pipeline  # noqa: E402
from dataflow_training.distributed.fleet import (  # noqa: E402
    local_topology,
    run_fleet_dp,
)
from dataflow_training.run.recipe import Recipe  # noqa: E402

pytestmark = pytest.mark.fleet

CORPUS = Path("datasets/fineweb10B")
needs_corpus = pytest.mark.skipif(
    not CORPUS.exists(), reason="needs the shard corpus")

STEPS = 6
SEED = 11


def quiet(*a, **k):
    pass


def tiny_cfg():
    from dataflow_training.model_families.gpt2.model import ShapedGpt2Config

    return replace(ShapedGpt2Config.tiny(), vocab_size=50304,
                   grad_accum_rounds=2, num_steps=STEPS)


@pytest.mark.gpu
@needs_corpus
def test_world1_fleet_matches_solo_engine():
    from dataflow_training.run.driver import daemon_client, run_engine

    cfg = tiny_cfg()
    recipe = Recipe(peak_lr=3e-4, min_lr=3e-5, warmup_steps=2,
                    total_steps=STEPS)

    solo = None
    with daemon_client(slab_gib=4.0, log=quiet) as client:
        solo = run_engine(client, cfg, recipe,
                          legacy_block_pipeline(cfg), STEPS,
                          budget_gib=4.0, seed=SEED, log=quiet)

    res = run_fleet_dp(cfg, recipe, legacy_block_pipeline(cfg), STEPS,
                       rank_rounds=(cfg.grad_accum_rounds,),
                       budgets=(4.0,), slabs=(4.0,),
                       topology=local_topology(budget_gib=4.0,
                                               slab_gib=4.0),
                       group="local", seed=SEED, log=quiet)

    assert len(res.losses) == STEPS
    assert all(math.isfinite(x) for x in res.losses)
    worst = max(abs(a - b) for a, b in zip(solo.losses, res.losses))
    # cross-process ambient envelope (the resume-drill class)
    assert worst < 5e-4, (worst, solo.losses, res.losses)
