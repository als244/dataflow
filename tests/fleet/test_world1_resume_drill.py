"""World-1 resume drill against the v2 checkpoint record: train with
checkpoints,
resume from a FRESH conductor + daemon (process-death equivalent), and
the resumed tail must reproduce the uninterrupted run within the
cross-process ambient envelope. Also asserts the manifest v2 surface:
format, responsibility save_plan, launch record with per-rank
programs, data cursor."""
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from dataflow_training.data.pipeline import legacy_block_pipeline  # noqa: E402
from dataflow_training.distributed.fleet import (  # noqa: E402
    local_topology,
    run,
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


def tiny_cfg(family_name):
    from dataflow_training.model_families.gpt2.model import ShapedGpt2Config
    from dataflow_training.model_families.llama3 import ShapedLlamaConfig
    from dataflow_training.model_families.qwen35.model import ShapedQwen35Config
    from dataflow_training.model_families.qwen3moe.model import ShapedQwen3MoeConfig

    tiny = {"gpt2": ShapedGpt2Config, "llama3": ShapedLlamaConfig,
            "qwen35": ShapedQwen35Config,
            "qwen3moe": ShapedQwen3MoeConfig}[family_name].tiny()
    return replace(tiny, vocab_size=50304,
                   grad_accum_rounds=2, num_steps=STEPS)


@pytest.mark.gpu
@needs_corpus
@pytest.mark.parametrize("family_name", ["gpt2", "llama3",
                                         "qwen35", "qwen3moe"])
def test_world1_checkpoint_resume_drill(tmp_path, family_name):
    cfg = tiny_cfg(family_name)
    recipe = Recipe(peak_lr=3e-4, min_lr=3e-5, warmup_steps=2,
                    total_steps=STEPS)
    ck_dir = tmp_path / "ck"
    common = dict(budgets=(4.0,), backing=(4.0,), group="local",
                  seed=SEED, log=quiet,
                  checkpoint_dir=str(ck_dir), run_name="drill")

    truth = run(cfg, recipe, legacy_block_pipeline(cfg), STEPS,
                         topology=local_topology(budget_gib=4.0,
                                                 backing_gib=4.0),
                         launch_argv=["unit", "world1-drill"],
                         checkpoint_every=2, **common)

    # manifest v2 surface at the newest checkpoint
    manifests = sorted((ck_dir / "drill").glob("step_*/checkpoint_record.json"))
    assert manifests, "no checkpoints written"
    m = json.loads(manifests[-1].read_text())
    assert m["format"] == 2
    assert m["world"] == 1
    assert m["save_plan"]["W_0"][0]["rank"] == 0
    assert m["launch"]["argv"] == ["unit", "world1-drill"]
    assert m["launch"]["programs"] == ["programs/rank0.json"]
    assert (manifests[-1].parent / "programs" / "rank0.json").is_file()
    assert m["data_cursor"] is not None
    ck_step = m["step"]
    assert ck_step < STEPS

    # fresh conductor + daemon resumes the tail
    resumed = run(cfg, recipe, legacy_block_pipeline(cfg),
                           STEPS,
                           topology=local_topology(budget_gib=4.0,
                                                   backing_gib=4.0),
                           launch_argv=["unit", "world1-drill"],
                           checkpoint_every=2, resume="auto", **common)

    assert all(math.isfinite(x) for x in resumed.losses)
    tail_truth = truth.losses[ck_step:]
    tail_resumed = resumed.losses[-len(tail_truth):]
    worst = max(abs(a - b) for a, b in zip(tail_truth, tail_resumed))
    assert worst < 5e-4, (worst, tail_truth, tail_resumed)
