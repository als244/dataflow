"""Cross-box DP-vs-solo parity on REAL data, per family: the same
global step batch trained (a) solo on one engine and (b) split 1:1
across two real hosts under the zero1rs DP default must produce the
same loss curve within the cross-process bf16 envelope.

This is the end-to-end complement to the family-generic weight-parity
gate (same-box, one step, rel_l2 in weight space): here the transport
is the real fabric (nccl/auto), the data is the shard corpus, and the
curve runs long enough for a wiring error to compound visibly.
Families cover the three structural regimes: dense (llama3), hybrid
linear attention (qwen35), routed experts + per-step balancing aux
(qwen3moe).

Skips without a topology.toml carrying a remote host.

Tests:
- test_crossbox_dp_matches_solo_curve: per family (llama3, qwen35, qwen3moe), a step batch trained solo on one engine and split 1:1 across two hosts under the zero1rs DP default yields finite equal-length curves whose worst per-step loss gap stays under 2e-2 and that actually descend.
"""
import math
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no GPU", allow_module_level=True)
pytest.importorskip("cuda.bindings")

from dataflow_training.distributed.topology import (  # noqa: E402
    load_topology_or_none,
    local_topology,
)

TOPO = load_topology_or_none()
if TOPO is None or not TOPO.remotes():
    pytest.skip("crossbox parity needs a topology.toml with a remote",
                allow_module_level=True)
if "dp" not in TOPO.groups or len(TOPO.groups["dp"].members) < 2:
    pytest.skip("crossbox parity needs a [groups.dp] with >=2 members",
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
def test_crossbox_dp_matches_solo_curve(family_name):
    cfg = tiny_cfg(family_name)
    recipe = Recipe(peak_lr=3e-4, min_lr=3e-5, warmup_steps=2,
                    total_steps=STEPS)

    solo = run(cfg, recipe, legacy_block_pipeline(cfg), STEPS,
               topology=local_topology(budget_gib=4.0, backing_gib=4.0),
               group="local", seed=SEED, log=quiet,
               launch_argv=["unit", "xparity-solo"])

    fleet = run(cfg, recipe, legacy_block_pipeline(cfg), STEPS,
                scheme=ParallelismScheme.data_parallel((1, 1)),
                budgets=(4.0, 4.0), backing=(4.0, 4.0),
                topology=TOPO, group="dp", seed=SEED, log=quiet,
                launch_argv=["unit", "xparity-dp"])

    assert all(math.isfinite(x) for x in fleet.losses)
    assert len(fleet.losses) == len(solo.losses) == STEPS
    # cross-process + fabric-reduction bf16 envelope; a real wiring
    # error (wrong group, missed grad, aux miscount) blows through
    # this by orders of magnitude within six steps
    worst = max(abs(a - b) for a, b in zip(solo.losses, fleet.losses))
    assert worst < 2e-2, (worst, solo.losses, fleet.losses)
    assert fleet.losses[-1] < fleet.losses[0]     # actually learned
