"""Fleet: THE stable import surface for distributed training.

The machinery lives in single-purpose components; import from here:

    ../run/conductor.py run(scheme=...) — THE orchestrator, any world
    parallelism.py      ParallelismScheme — THE parallelism contract
    ../run/loop.py      fleet_loop — the step loop
    ranks.py            RankState / HostRig / put_rank_rounds / versions
    topology.py         Topology + the zero-config local builders
    hosts.py            run commands / move files on any host
    daemons.py          dataflowd lifecycle on a host (launch/kill/...)
    grouped_lowering.py lower_with_group — blind lower -> annotate ->
                        exact sizes (the three composable passes)
    ../run/checkpointing.py checkpoint save / resume orchestration
    sharding.py         layout-geometry shard math (unchanged)
    responsibility.py   who steps, saves (the save-plan derivation)
    ../run/checkpoint_record.py the per-checkpoint record (v2):
                        contents + save plan + launch provenance
"""
from ..run.checkpointing import (  # noqa: F401
    checkpoint_fleet,
    load_checkpoint,
    distribute_artifacts,
    resolve_resume,
)
from ..run.conductor import (  # noqa: F401
    check_fleet_versions,
    run,
)
from .parallelism import ParallelismScheme  # noqa: F401
from .grouped_lowering import (  # noqa: F401
    GroupedBuildVariant,
    lower_with_group,
)
from ..run.loop import fleet_loop  # noqa: F401
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
