"""Two in-process daemons, SAME program, sequentially: the second boot
must run clean. A program's id is a content hash, so re-registering the
identical program in a fresh daemon lands on the SAME prog_id — and the
bridge's session cache is module-level. Before server shutdown closed
the cached sessions (before freeing the store slab), the second daemon
inherited a BufferPool full of pointers into the FIRST daemon's freed
slab and segfaulted in its first backing copy (cudaMemcpyAsync). Found
by the first back-to-back same-family boots in the parity suite.

Tests:
- test_relaunched_daemon_same_program_reruns_clean_and_reproduces_losses: running the same program in two sequential fresh in-process daemons yields finite losses both times and per-step losses that agree within 1e-6.
"""
import math

import pytest
import torch

pytest.importorskip("cuda.bindings.runtime")

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(),
                       reason="needs CUDA"),
    pytest.mark.gpu,
    pytest.mark.corpus,
]


def quiet_log(*args, **kwargs):
    pass


def one_daemon_run(steps: int) -> list[float]:
    from dataflow_training.run.driver import engine_client, run_engine
    from dataflow_training.data.pipeline import legacy_block_pipeline
    from dataflow_training.run.presets import qwen3_smoke_preset
    from dataflow_training.run.recipe import Recipe

    cfg = qwen3_smoke_preset()
    recipe = Recipe(peak_lr=3e-4, min_lr=3e-5, warmup_steps=2,
                    total_steps=steps)
    feed = legacy_block_pipeline(cfg)
    with engine_client(backing_gib=6.0, log=quiet_log) as client:
        res = run_engine(client, cfg, recipe, feed, steps,
                         budget_gib=4.0, seed=11, log=quiet_log)
    return res.losses


def test_relaunched_daemon_same_program_reruns_clean_and_reproduces_losses():
    first = one_daemon_run(3)
    second = one_daemon_run(3)   # same prog_id in a fresh daemon
    assert all(math.isfinite(x) for x in first + second), (first, second)
    # same seed, same feed, fresh daemons: the runs are replicas
    for a, b in zip(first, second):
        assert abs(a - b) < 1e-6, (first, second)
