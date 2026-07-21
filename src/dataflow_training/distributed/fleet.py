"""Fleet: THE stable import surface for distributed training.

The machinery lives in single-purpose components; import from here:

    conductor.py        run_fleet_dp — launch, guard, register, run
    loop.py             fleet_loop — the step loop
    ranks.py            RankState / HostRig / put_rank_rounds / versions
    topology.py         Topology + the zero-config local builders
    grouped_lowering.py lower_with_group — blind lower -> annotate ->
                        exact sizes (the three composable passes)
    checkpointing.py    manifest-v2 save / resume orchestration
    sharding.py         layout-geometry shard math (unchanged)
    responsibility.py   who steps, saves (the save-plan derivation)
    manifest.py         checkpoint manifest v2 read/write
"""
from .checkpointing import (  # noqa: F401
    checkpoint_fleet,
    distribute_artifacts,
    resolve_resume,
)
from .conductor import (  # noqa: F401
    check_fleet_versions,
    run_fleet_dp,
)
from .grouped_lowering import (  # noqa: F401
    GroupedBuildVariant,
    lower_with_group,
)
from .loop import fleet_loop  # noqa: F401
from .ranks import (  # noqa: F401
    HostRig,
    RankState,
    StepRun,
    put_rank_rounds,
    wait_client,
)
from .sharding import (  # noqa: F401
    layer_fields_by_root,
    zero1rs_block_params,
)
from .topology import (  # noqa: F401
    local_pair_topology,
    local_topology,
)
