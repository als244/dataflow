"""World-2 same-box resume through the v1 record, BITWISE: two local
daemons train a tiny family with mid-run checkpoints, a fresh pair
resumes from the newest complete record, and the resumed tail must
equal the uninterrupted tail exactly. Exact equality is the claim
that the record captures ALL state the tail depends on — weights,
zero1 optimizer shards, cross-step accumulators, data cursor,
schedule position — and that resume under the simple policy is one
self-sufficient restore per rank, byte-identical to the live state
it replaces. The dense drill (llama3) certifies the zero1 shard
mapping; the MoE drill (qwen3moe) certifies the persistent SET —
selection is the program's marker, and expert counts ride it.

Tests:
- test_world2_resume_reproduces_tail_bitwise: a world-2 local-pair run under the DP default writes a v1 record (replicated W as two hash-certified whole-copy slices, zero1rs O shards mapped at element offsets), and a fresh pair resumed from it reproduces the uninterrupted tail bit-for-bit.
- test_world2_moe_persistent_set_round_trips: the qwen3moe world-2 record covers the FULL persistent set — per-rank expert-count Aux objects ride as writer-qualified private logicals — leaves per-step objects out, and the resumed MoE tail equals the uninterrupted tail bit-for-bit.
- test_world2_remapped_resume_bitwise: resume with the rank-to-engine mapping SWAPPED (rank 0 rides the slot that wrote rank 1's snapshots): the hosts change is allowed as placement-independent, each rank still restores its OWN writer's state, and the tail stays bitwise — the same-box stand-in for a cross-box topology change.
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
MOE_PAIR_PORTS = (29733, 29734)
REMAP_PAIR_PORTS = (29735, 29736)


def quiet(*a, **k):
    pass


def pair_topo():
    return local_pair_topology(ports=PAIR_PORTS)


def moe_pair_topo():
    return local_pair_topology(ports=MOE_PAIR_PORTS)


def remap_pair_topo():
    return local_pair_topology(ports=REMAP_PAIR_PORTS)


def swapped_pair_topo():
    """The SAME two slots as remap_pair_topo, rank mapping SWAPPED:
    group members in reverse order, so rank 0 boots on the slot that
    wrote rank 1's snapshots."""
    import os

    from dataflow_training.distributed.topology import (GroupSpec,
                                                        HostSpec,
                                                        Topology)

    hosts = {}
    for i, port in enumerate(REMAP_PAIR_PORTS):
        name = f"local{i}"
        hosts[name] = HostSpec(name=name,
                               peer_listen=f"127.0.0.1:{port}",
                               ssh=None, repo=os.getcwd(),
                               backing_gib=4.0, budget_gib=4.0,
                               device=0)
    return Topology(conductor="local0", hosts=hosts,
                    groups={"pair": GroupSpec(name="pair",
                                              members=("local1",
                                                       "local0"),
                                              backend="hostmem")},
                    source="<swapped local pair>")


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
    # writers — same span, hash-certified interchangeable replicas
    w_id = sorted(o for o in record["logical_objects"]
                  if o.startswith("W_"))[0]
    w_slices = record["slices"][w_id]
    w_bytes = record["logical_objects"][w_id]["bytes"]
    assert len(w_slices) == 2
    assert all(s["object_range"] == [0, w_bytes] for s in w_slices)
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


@pytest.mark.gpu
@pytest.mark.corpus
@pytest.mark.vram(gib=8)
def test_world2_moe_persistent_set_round_trips(tmp_path):
    from dataflow_training.model_families.qwen3moe.model import \
        ShapedQwen3MoeConfig

    cfg = replace(ShapedQwen3MoeConfig.tiny(), vocab_size=50304,
                  grad_accum_rounds=2, num_steps=STEPS)
    recipe = Recipe(peak_lr=3e-4, min_lr=3e-5, warmup_steps=2,
                    total_steps=STEPS)
    ck_dir = tmp_path / "ck"
    common = dict(scheme=ParallelismScheme.data_parallel((1, 1)),
                  budgets=(4.0, 4.0), backing=(4.0, 4.0),
                  group="pair", seed=SEED, log=quiet,
                  checkpoint_dir=str(ck_dir), run_name="moedrill",
                  checkpoint_every=2)

    truth = run(cfg, recipe, legacy_block_pipeline(cfg), STEPS,
                topology=moe_pair_topo(),
                launch_argv=["unit", "moe-drill"], **common)

    records = sorted((ck_dir / "moedrill").glob(
        "step_*/checkpoint_record.json"))
    assert records, "no checkpoints written"
    record = json.loads(records[-1].read_text())
    ck_step = record["step"]
    assert ck_step < STEPS, "the drill needs a tail after the record"

    # the record covers the FULL persistent set — including the
    # expert-count Aux objects a W_/O_ prefix convention would lose —
    # and nothing per-step
    prog = json.loads((records[-1].parent / "programs"
                       / "rank0.json").read_text())
    persistent = {o["id"] for o in prog["initial_objects"]
                  if o.get("persistent")}
    per_step = {o["id"] for o in prog["initial_objects"]
                if not o.get("persistent")}
    aux_ids = sorted(o for o in persistent if o.startswith("Aux_"))
    assert aux_ids, "the MoE program persists expert-count objects"
    on_record = set(record["logical_objects"])
    for oid in sorted(persistent):
        assert oid in on_record or f"{oid}@0" in on_record, \
            f"{oid}: every persistent object must be on the record"
    for oid in sorted(per_step):
        assert oid not in on_record and f"{oid}@0" not in on_record, \
            f"{oid}: per-step objects must NOT be on the record"

    # each rank accumulates its OWN expert counts — no step
    # synchronizes the copies — so counts land writer-PRIVATE:
    # one whole qualified logical per writer, never certified as
    # replicas
    for w in (0, 1):
        lid = f"{aux_ids[0]}@{w}"
        private = record["slices"][lid]
        aux_bytes = record["logical_objects"][lid]["bytes"]
        assert len(private) == 1
        assert private[0]["object_range"] == [0, aux_bytes]

    resumed = run(cfg, recipe, legacy_block_pipeline(cfg), STEPS,
                  topology=moe_pair_topo(),
                  launch_argv=["unit", "moe-drill"],
                  resume="auto", **common)

    assert all(math.isfinite(x) for x in resumed.losses)
    tail_truth = truth.losses[ck_step:]
    tail_resumed = resumed.losses[-len(tail_truth):]
    assert tail_resumed == tail_truth, (
        "the resumed MoE tail must be BITWISE-equal — a mismatch "
        "means state the tail depends on is missing from the record",
        tail_truth, tail_resumed)


@pytest.mark.gpu
@pytest.mark.corpus
@pytest.mark.vram(gib=8)
def test_world2_remapped_resume_bitwise(tmp_path):
    from dataflow_training.model_families.llama3 import ShapedLlamaConfig

    cfg = replace(ShapedLlamaConfig.tiny(), vocab_size=50304,
                  grad_accum_rounds=2, num_steps=STEPS)
    recipe = Recipe(peak_lr=3e-4, min_lr=3e-5, warmup_steps=2,
                    total_steps=STEPS)
    ck_dir = tmp_path / "ck"
    common = dict(scheme=ParallelismScheme.data_parallel((1, 1)),
                  budgets=(4.0, 4.0), backing=(4.0, 4.0),
                  group="pair", seed=SEED, log=quiet,
                  checkpoint_dir=str(ck_dir), run_name="remapdrill",
                  checkpoint_every=2)

    truth = run(cfg, recipe, legacy_block_pipeline(cfg), STEPS,
                topology=remap_pair_topo(),
                launch_argv=["unit", "remap-drill"], **common)

    records = sorted((ck_dir / "remapdrill").glob(
        "step_*/checkpoint_record.json"))
    assert records, "no checkpoints written"
    record = json.loads(records[-1].read_text())
    ck_step = record["step"]
    assert ck_step < STEPS
    assert record["launch"]["resolved"]["hosts"] == \
        ["local0", "local1"]

    # resume with the mapping SWAPPED: hosts may move — each rank
    # still restores its OWN writer's snapshot, capability-checked
    resumed = run(cfg, recipe, legacy_block_pipeline(cfg), STEPS,
                  topology=swapped_pair_topo(),
                  launch_argv=["unit", "remap-drill"],
                  resume="auto", **common)

    assert all(math.isfinite(x) for x in resumed.losses)
    tail_truth = truth.losses[ck_step:]
    tail_resumed = resumed.losses[-len(tail_truth):]
    assert tail_resumed == tail_truth, (
        "the remapped resume must reproduce the tail BITWISE — rank "
        "state follows the RANK, not the engine it first ran on",
        tail_truth, tail_resumed)
