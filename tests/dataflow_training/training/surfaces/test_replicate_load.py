"""A world-1 checkpoint stood back up WITHOUT the conductor:
``load_checkpoint(engines="replicate")`` launches a child daemon
shaped by the record's engine spec, restores the full rank view, and
the engine RUNS — registering the checkpoint's own saved program and
stepping once reproduces the uninterrupted run's next loss exactly.
This certifies the generic loader end-to-end: a record plus its
snapshot dirs is sufficient to resume compute, no training conductor
involved.

Tests:
- test_world1_replicate_steps_once_bitwise: replicate-boot a world-1 checkpoint, register its saved program, feed the next step's rounds from the recorded data cursor, run one step, and the loss equals the uninterrupted run's loss at that step bit-for-bit.
"""
import json
import math
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no GPU", allow_module_level=True)

from dataflow_training.data.pipeline import legacy_block_pipeline  # noqa: E402
from dataflow_training.distributed.fleet import run  # noqa: E402
from dataflow_training.distributed.ranks import (  # noqa: E402
    RankState,
    put_rank_rounds,
)
from dataflow_training.distributed.topology import local_topology  # noqa: E402
from dataflow_training.run.checkpointing import load_checkpoint  # noqa: E402
from dataflow_training.run.recipe import Recipe  # noqa: E402

STEPS = 6
SEED = 11
PORT = 29739


def quiet(*a, **k):
    pass


@pytest.mark.gpu
@pytest.mark.corpus
@pytest.mark.vram(gib=8)
def test_world1_replicate_steps_once_bitwise(tmp_path):
    from dataflow_training.model_families.llama3 import ShapedLlamaConfig
    from dataflow_training.run.presets import cfg_dict, resolver_family

    cfg = replace(ShapedLlamaConfig.tiny(), vocab_size=50304,
                  grad_accum_rounds=2, num_steps=STEPS)
    recipe = Recipe(peak_lr=3e-4, min_lr=3e-5, warmup_steps=2,
                    total_steps=STEPS)
    ck_dir = tmp_path / "ck"

    truth = run(cfg, recipe, legacy_block_pipeline(cfg), STEPS,
                warm_profiles=True,  # explicit: tiny geometry, seconds, disk-cached
                topology=local_topology(budget_gib=4.0,
                                        backing_gib=4.0,
                                        peer_port=PORT),
                launch_argv=["unit", "replicate-drill"],
                budgets=(4.0,), backing=(4.0,), group="local",
                seed=SEED, log=quiet, checkpoint_dir=str(ck_dir),
                run_name="repl", checkpoint_every=2)
    assert all(math.isfinite(x) for x in truth.losses)

    records = sorted((ck_dir / "repl").glob(
        "step_*/checkpoint_record.json"))
    assert records, "no checkpoints written"
    step_dir = records[-1].parent
    ck_step = json.loads(records[-1].read_text())["step"]
    assert ck_step < STEPS, "the drill needs a step after the record"

    record, engines = load_checkpoint(step_dir, engines="replicate")
    try:
        client = engines["0"]
        # the restored engine RUNS: the checkpoint's own program, the
        # next step's rounds from the recorded cursor, one step
        prog = json.loads((step_dir / "programs"
                           / "rank0.json").read_text())
        resolver = {"kind": "model_family",
                    "family": resolver_family(cfg),
                    "cfg": cfg_dict(replace(cfg, num_steps=1)),
                    "hyper": recipe.hyper_spec()}
        stepper = legacy_block_pipeline(cfg)(
            record["client_payload"]["data_cursor"])
        packed = stepper.next_step()
        rank = RankState(name="repl0", client=client, cfg=cfg,
                         rounds=tuple(range(cfg.grad_accum_rounds)))
        lens = put_rank_rounds(rank, packed, cfg.max_tokens)
        reg = client.register_program(prog, resolver=resolver)
        assert not reg["bindings"]["missing_inputs"], reg["bindings"]

        out = client.run(reg["prog_id"],
                         args={"step": ck_step,
                               "valid_rows": packed.valid_rows,
                               "seq_lens": lens},
                         fetch=[f"loss_0_{r}"
                                for r in range(cfg.grad_accum_rounds)])
        assert out.get("state") == "done", out
        step_loss = sum(out["fetched"].values())
        assert step_loss == truth.losses[ck_step], (
            "the replicated engine's first step must equal the "
            "uninterrupted run's loss at that step",
            step_loss, truth.losses[ck_step])
    finally:
        for c in engines.values():
            c.shutdown()
