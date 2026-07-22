"""World-2 same-box resume drill against manifest v2 — under the DP
default (zero1rs engages automatically at world > 1): two local
daemons train with checkpoints, a fresh pair resumes, and the tail
must reproduce the uninterrupted run within the ambient envelope.

This certifies in one gate: the zero1rs default, the
responsibility-partitioned RANGED saves (each rank writes its param
byte range + its own O shard), and resume's ordered artifact replay
reassembling complete params on both ranks.

Tests:
- test_world2_zero1rs_checkpoint_resume_drill: for each family, two local daemons emit a v2 manifest with param bytes partitioned into contiguous per-rank ranges under zero1rs, and a fresh pair resumes to reproduce the uninterrupted tail within tolerance.
"""
import json
import math
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from dataflow_training.data.pipeline import legacy_block_pipeline  # noqa: E402
from dataflow_training.distributed.fleet import (  # noqa: E402
    ParallelismScheme,
    local_pair_topology,
    run,
)
from dataflow_training.run.recipe import Recipe  # noqa: E402

pytestmark = pytest.mark.fleet

STEPS = 6
SEED = 11


def quiet(*a, **k):
    pass


PAIR_PORTS = {"llama3": (29721, 29722), "qwen35": (29723, 29724),
              "qwen3moe": (29725, 29726)}


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
@pytest.mark.vram(gib=8)
@pytest.mark.parametrize("family_name", ["llama3", "qwen35", "qwen3moe"])
def test_world2_zero1rs_checkpoint_resume_drill(tmp_path, family_name):
    cfg = tiny_cfg(family_name)
    recipe = Recipe(peak_lr=3e-4, min_lr=3e-5, warmup_steps=2,
                    total_steps=STEPS)
    ck_dir = tmp_path / "ck"

    def pair_topo():
        return local_pair_topology(ports=PAIR_PORTS[family_name])
    common = dict(scheme=ParallelismScheme.data_parallel((1, 1)),
                  budgets=(4.0, 4.0),
                  backing=(4.0, 4.0), group="pair", seed=SEED, log=quiet,
                  checkpoint_dir=str(ck_dir), run_name="drill2",
                  checkpoint_every=2)

    truth = run(cfg, recipe, legacy_block_pipeline(cfg), STEPS,
                         topology=pair_topo(),
                         launch_argv=["unit", "world2-drill"], **common)

    manifests = sorted((ck_dir / "drill2").glob("step_*/checkpoint_record.json"))
    assert manifests, "no checkpoints written"
    m = json.loads(manifests[-1].read_text())
    assert m["format"] == 2
    assert m["world"] == 2
    # zero1rs default: param bytes PARTITIONED across the two ranks
    w0 = m["save_plan"]["W_0"]
    assert [e["rank"] for e in w0] == [0, 1]
    assert w0[0]["hi"] == w0[1]["lo"]
    assert m["launch"]["resolved"]["opt_shard"] == "zero1rs"
    assert m["launch"]["programs"] == ["programs/rank0.json",
                                       "programs/rank1.json"]
    ck_step = m["step"]

    resumed = run(cfg, recipe, legacy_block_pipeline(cfg),
                           STEPS, topology=pair_topo(),
                           launch_argv=["unit", "world2-drill"],
                           resume="auto", **common)

    assert all(math.isfinite(x) for x in resumed.losses)
    tail_truth = truth.losses[ck_step:]
    tail_resumed = resumed.losses[-len(tail_truth):]
    worst = max(abs(a - b) for a, b in zip(tail_truth, tail_resumed))
    assert worst < 5e-4, (worst, tail_truth, tail_resumed)
