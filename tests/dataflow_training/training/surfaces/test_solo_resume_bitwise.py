"""Solo checkpoint/resume through the general record path: a world-1
run_engine trains with checkpoints — the SAME policy-compiled save
the fleet uses, one snapshot plus a v1 record landing last — and a
FRESH engine resumes to reproduce the uninterrupted tail exactly.
The solo convention (bare snapshots with run state in client_meta)
is gone; there is one checkpoint format at every world size.

Tests:
- test_solo_resume_reproduces_tail_bitwise: run_engine writes a v1 record (world 1, program beside it, summary scalars, no snapshot client_meta), and a fresh engine resumed from it reproduces the uninterrupted tail bit-for-bit.
"""
import json
import math
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no GPU", allow_module_level=True)
pytest.importorskip("cuda.bindings")

from dataflow.checkpoint import RECORD_SCHEMA  # noqa: E402
from dataflow_training.data.pipeline import legacy_block_pipeline  # noqa: E402
from dataflow_training.run.driver import engine_client, run_engine  # noqa: E402
from dataflow_training.run.recipe import Recipe  # noqa: E402

STEPS = 6
SEED = 11


def quiet(*a, **k):
    pass


@pytest.mark.gpu
@pytest.mark.corpus
def test_solo_resume_reproduces_tail_bitwise(tmp_path):
    from dataflow_training.model_families.llama3 import ShapedLlamaConfig

    cfg = replace(ShapedLlamaConfig.tiny(), vocab_size=50304,
                  grad_accum_rounds=2, num_steps=STEPS)
    recipe = Recipe(peak_lr=3e-4, min_lr=3e-5, warmup_steps=2,
                    total_steps=STEPS)
    ck_dir = tmp_path / "ck"

    with engine_client(backing_gib=4.0, log=quiet) as client:
        truth = run_engine(client, cfg, recipe,
                           legacy_block_pipeline(cfg), STEPS,
                           budget_gib=4.0, seed=SEED, log=quiet,
                           checkpoint_every=4,
                           checkpoint_dir=str(ck_dir))
    assert all(math.isfinite(x) for x in truth.losses)

    records = sorted(ck_dir.glob("step_*/checkpoint_record.json"))
    assert records, "no checkpoint records written"
    record = json.loads(records[-1].read_text())
    assert record["schema"] == RECORD_SCHEMA
    assert record["scheme"]["world"] == 1
    assert record["summary"]["steps_recorded"] == record["step"]
    assert (records[-1].parent / "programs" / "rank0.json").is_file()
    ck_step = record["step"]
    assert ck_step < STEPS, "the drill needs a tail after the record"

    with engine_client(backing_gib=4.0, log=quiet) as fresh:
        resumed = run_engine(fresh, cfg, recipe,
                             legacy_block_pipeline(cfg), STEPS,
                             budget_gib=4.0, seed=SEED, log=quiet,
                             checkpoint_every=4,
                             checkpoint_dir=str(ck_dir), resume=True)

    assert all(math.isfinite(x) for x in resumed.losses)
    tail_truth = truth.losses[ck_step:]
    tail_resumed = resumed.losses[-len(tail_truth):]
    assert tail_resumed == tail_truth, (
        "the solo resumed tail must be BITWISE-equal to the "
        "uninterrupted tail", tail_truth, tail_resumed)
