"""Cross-box resume drill against manifest v2 — the third and final
S4 drill: two REAL hosts from topology.toml (nccl or auto backend),
zero1rs default, checkpoints written per responsibility on each box,
artifacts distributed across boxes at resume, and the resumed tail
must reproduce the uninterrupted run within the ambient envelope.

Skips without a topology.toml carrying a remote host (runs on the
conductor box of the pair)."""
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from dataflow_training.distributed.topology import load_topology_or_none  # noqa: E402

TOPO = load_topology_or_none()
if TOPO is None or not TOPO.remotes():
    pytest.skip("crossbox drill needs a topology.toml with a remote",
                allow_module_level=True)

from dataflow_training.data.pipeline import legacy_block_pipeline  # noqa: E402
from dataflow_training.distributed.fleet import (  # noqa: E402
    ParallelismScheme,
    run,
)
from dataflow_training.run.recipe import Recipe  # noqa: E402

pytestmark = pytest.mark.fleet

STEPS = 6
SEED = 11


def quiet(*a, **k):
    pass


def tiny_cfg():
    from dataflow_training.model_families.llama3 import ShapedLlamaConfig

    return replace(ShapedLlamaConfig.tiny(), vocab_size=50304,
                   grad_accum_rounds=2, num_steps=STEPS)


@pytest.mark.gpu
def test_crossbox_zero1rs_checkpoint_resume_drill(tmp_path):
    cfg = tiny_cfg()
    recipe = Recipe(peak_lr=3e-4, min_lr=3e-5, warmup_steps=2,
                    total_steps=STEPS)
    ck_dir = tmp_path / "ck"
    common = dict(scheme=ParallelismScheme.data_parallel((1, 1)),
                  budgets=(4.0, 4.0),
                  slabs=(6.0, 6.0), group="dp", seed=SEED, log=quiet,
                  topology=TOPO, checkpoint_dir=str(ck_dir),
                  run_name="xdrill", checkpoint_every=2)

    truth = run(cfg, recipe, legacy_block_pipeline(cfg), STEPS,
                         launch_argv=["unit", "crossbox-drill"],
                         **common)

    manifests = sorted((ck_dir / "xdrill").glob("step_*/fleet.json"))
    assert manifests, "no checkpoints written"
    m = json.loads(manifests[-1].read_text())
    assert m["format"] == 2
    assert m["world"] == 2
    assert m["launch"]["resolved"]["opt_shard"] == "zero1rs"
    ck_step = m["step"]

    resumed = run(cfg, recipe, legacy_block_pipeline(cfg),
                           STEPS, launch_argv=["unit", "crossbox-drill"],
                           resume="auto", **common)

    assert all(math.isfinite(x) for x in resumed.losses)
    tail_truth = truth.losses[ck_step:]
    tail_resumed = resumed.losses[-len(tail_truth):]
    worst = max(abs(a - b) for a, b in zip(tail_truth, tail_resumed))
    assert worst < 5e-4, (worst, tail_truth, tail_resumed)
