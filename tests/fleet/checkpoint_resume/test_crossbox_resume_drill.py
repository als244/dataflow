"""Cross-box resume drill against the v2 checkpoint record: two REAL
hosts from topology.toml (nccl or auto backend), zero1rs default,
checkpoints written per responsibility on each box, artifacts
distributed across boxes at resume, and the resumed tail must
reproduce the uninterrupted run within the ambient envelope.

Skips without a topology.toml carrying a remote host (runs on the
conductor box of the pair).

Tests:
- test_crossbox_zero1rs_resume_matches_uninterrupted_tail: across two real hosts the v2 manifest records a world-2 zero1rs save, resume replays box-distributed artifacts, rank and aggregate loader views agree on the replicated weights, and the resumed tail matches the uninterrupted tail within tolerance.
"""
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no GPU", allow_module_level=True)
pytest.importorskip("cuda.bindings")

from dataflow_training.distributed.topology import load_topology_or_none  # noqa: E402

TOPO = load_topology_or_none()
if TOPO is None or not TOPO.remotes():
    pytest.skip("crossbox drill needs a topology.toml with a remote",
                allow_module_level=True)
if "dp" not in TOPO.groups or len(TOPO.groups["dp"].members) < 2:
    pytest.skip("crossbox drill needs a [groups.dp] with >=2 members",
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


def tiny_cfg(family_name):
    from dataflow_training.model_families.llama3 import ShapedLlamaConfig
    from dataflow_training.model_families.qwen35.model import ShapedQwen35Config
    from dataflow_training.model_families.qwen3moe.model import ShapedQwen3MoeConfig

    tiny = {"llama3": ShapedLlamaConfig, "qwen35": ShapedQwen35Config,
            "qwen3moe": ShapedQwen3MoeConfig}[family_name].tiny()
    return replace(tiny, vocab_size=50304,
                   grad_accum_rounds=2, num_steps=STEPS)


@pytest.mark.gpu
@pytest.mark.corpus
@pytest.mark.parametrize("family_name", ["llama3", "qwen35", "qwen3moe"])
def test_crossbox_zero1rs_resume_matches_uninterrupted_tail(tmp_path,
                                                            family_name):
    cfg = tiny_cfg(family_name)
    recipe = Recipe(peak_lr=3e-4, min_lr=3e-5, warmup_steps=2,
                    total_steps=STEPS)
    ck_dir = tmp_path / "ck"
    common = dict(scheme=ParallelismScheme.data_parallel((1, 1)),
                  budgets=(4.0, 4.0),
                  backing=(6.0, 6.0), group="dp", seed=SEED, log=quiet,
                  topology=TOPO, checkpoint_dir=str(ck_dir),
                  run_name="xdrill", checkpoint_every=2)

    truth = run(cfg, recipe, legacy_block_pipeline(cfg), STEPS,
                         launch_argv=["unit", "crossbox-drill"],
                         **common)

    manifests = sorted((ck_dir / "xdrill").glob("step_*/checkpoint_record.json"))
    assert manifests, "no checkpoints written"
    m = json.loads(manifests[-1].read_text())
    assert m["format"] == 2
    assert m["world"] == 2
    assert m["launch"]["resolved"]["opt_shard"] == "zero1rs"
    ck_step = m["step"]

    resumed = run(cfg, recipe, legacy_block_pipeline(cfg),
                           STEPS, launch_argv=["unit", "crossbox-drill"],
                           resume="auto", **common)

    # high-level loader over the distributed step dir (artifacts were
    # pulled to the conductor box at resume): rank views agree on the
    # replicated weights; aggregate weight view matches them
    from dataflow_training.run.checkpointing import load_checkpoint

    step_dir = manifests[-1].parent
    r0rec, c0 = load_checkpoint(step_dir, rank=0, include_opt=True)
    r1rec, c1 = load_checkpoint(step_dir, rank=1, include_opt=True)
    agg_rec, ca = load_checkpoint(step_dir)
    w0 = bytes(c0.get_object("W_0"))
    assert w0 == bytes(c1.get_object("W_0")) == bytes(ca.get_object("W_0"))
    for c in (c0, c1, ca):
        c.shutdown()

    assert all(math.isfinite(x) for x in resumed.losses)
    tail_truth = truth.losses[ck_step:]
    tail_resumed = resumed.losses[-len(tail_truth):]
    worst = max(abs(a - b) for a, b in zip(tail_truth, tail_resumed))
    assert worst < 5e-4, (worst, tail_truth, tail_resumed)
