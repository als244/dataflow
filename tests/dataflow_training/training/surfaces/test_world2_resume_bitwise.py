"""World-2 same-box resume through the v1 record, BITWISE: two local
daemons train the tiny dense family with mid-run checkpoints, a fresh
pair resumes from the newest complete record, and the resumed tail
must equal the uninterrupted tail exactly. Exact equality is the
claim that the record captures ALL state the tail depends on —
weights, zero1 optimizer shards, data cursor, schedule position —
and that resume under the simple policy is one self-sufficient
restore per rank, byte-identical to the live state it replaces.

Tests:
- test_world2_resume_reproduces_tail_bitwise: a world-2 local-pair run under the DP default writes a v1 record (replicated W as two whole-copy slices with one authoritative winner, zero1rs O shards mapped at element offsets), and a fresh pair resumed from it reproduces the uninterrupted tail bit-for-bit.
"""
import json
import math
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no GPU", allow_module_level=True)

from dataflow.checkpoint import RECORD_SCHEMA  # noqa: E402
from dataflow_training.data.pipeline import legacy_block_pipeline  # noqa: E402
from dataflow_training.distributed.fleet import (  # noqa: E402
    ParallelismScheme,
    local_pair_topology,
    run,
)
from dataflow_training.run.recipe import Recipe  # noqa: E402

STEPS = 6
SEED = 11
PAIR_PORTS = (29731, 29732)


def quiet(*a, **k):
    pass


def pair_topo():
    return local_pair_topology(ports=PAIR_PORTS)


@pytest.mark.gpu
@pytest.mark.corpus
@pytest.mark.vram(gib=8)
def test_world2_resume_reproduces_tail_bitwise(tmp_path):
    from dataflow_training.model_families.llama3 import ShapedLlamaConfig

    cfg = replace(ShapedLlamaConfig.tiny(), vocab_size=50304,
                  grad_accum_rounds=2, num_steps=STEPS)
    recipe = Recipe(peak_lr=3e-4, min_lr=3e-5, warmup_steps=2,
                    total_steps=STEPS)
    ck_dir = tmp_path / "ck"
    common = dict(scheme=ParallelismScheme.data_parallel((1, 1)),
                  budgets=(4.0, 4.0), backing=(4.0, 4.0),
                  group="pair", seed=SEED, log=quiet,
                  checkpoint_dir=str(ck_dir), run_name="bitdrill",
                  checkpoint_every=2)

    truth = run(cfg, recipe, legacy_block_pipeline(cfg), STEPS,
                topology=pair_topo(),
                launch_argv=["unit", "bitwise-drill"], **common)

    records = sorted((ck_dir / "bitdrill").glob(
        "step_*/checkpoint_record.json"))
    assert records, "no checkpoints written"
    record = json.loads(records[-1].read_text())
    assert record["schema"] == RECORD_SCHEMA
    assert record["scheme"]["world"] == 2
    assert record["scheme"]["source_policy"] == "simple"
    assert record["launch"]["resolved"]["opt_shard"] == "zero1rs"
    ck_step = record["step"]
    assert ck_step < STEPS, "the drill needs a tail after the record"

    # the simple policy saves the replicated W whole from BOTH
    # writers — same span, hash-certified replicas, one restore winner
    w_id = sorted(o for o in record["logical_objects"]
                  if o.startswith("W_"))[0]
    w_slices = record["slices"][w_id]
    w_bytes = record["logical_objects"][w_id]["bytes"]
    assert len(w_slices) == 2
    assert all(s["object_range"] == [0, w_bytes] for s in w_slices)
    assert [s["authoritative"] for s in w_slices] == [True, False]
    assert len({s["hash"] for s in w_slices}) == 1
    # zero1rs O shards: per-writer slices land at element offsets and
    # the spans union to the full logical object (validated on write)
    o_id = sorted(o for o in record["logical_objects"]
                  if o.startswith("O_"))[0]
    assert len(record["slices"][o_id]) == 4, \
        "world-2 zero1 O = two [m|v] slices per writer"

    resumed = run(cfg, recipe, legacy_block_pipeline(cfg), STEPS,
                  topology=pair_topo(),
                  launch_argv=["unit", "bitwise-drill"],
                  resume="auto", **common)

    assert all(math.isfinite(x) for x in resumed.losses)
    tail_truth = truth.losses[ck_step:]
    tail_resumed = resumed.losses[-len(tail_truth):]
    assert tail_resumed == tail_truth, (
        "resumed tail must be BITWISE-equal to the uninterrupted "
        "tail", tail_truth, tail_resumed)
