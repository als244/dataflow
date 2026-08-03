#!/usr/bin/env python3
"""Compare PressureFit feasibility and selected makespan across two sources.

The harness builds each bare TaskChain once, serializes it, runs both planner
versions in isolated Python subprocesses, and replays both returned plans with
the candidate source's validator/simulator as the common physical oracle.
This prevents workload-builder or simulator drift from being mistaken for a
planner-quality change.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Iterator, Literal


_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA_VERSION = 1
_DEFAULT_HARDWARE = ("H100", "GB300", "RTX_5090", "SRAM Accelerator")
_DEFAULT_RATIOS = (1.0, 0.75, 0.5, 0.35, 0.25)
_DEFAULT_SYNTHETIC_LAYERS = (1, 2, 5, 10, 25, 100)
_DEFAULT_SYNTHETIC_CAPS = (
    144,
    160,
    192,
    224,
    256,
    320,
    384,
    500,
    640,
    800,
    1024,
    1600,
    2400,
)

_Status = Literal["valid", "infeasible", "invalid"]


@dataclass(frozen=True, slots=True)
class _Scenario:
    scenario_id: str
    group: str
    metadata: dict[str, Any]
    chain: dict[str, Any]
    capacity_bytes: int

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _ModelSpec:
    family: str
    scale: str
    overrides: dict[str, Any]
    seqlen: int
    optimizer: str


_MODEL_SPECS = (
    _ModelSpec(
        "llama3",
        "8B",
        dict(
            n_layers=2,
            d_model=512,
            n_heads=8,
            n_kv_heads=8,
            expert_dim=2048,
            vocab_size=32_000,
        ),
        128,
        "adamw",
    ),
    _ModelSpec(
        "qwen3",
        "32B",
        dict(
            n_layers=3,
            d_model=768,
            head_dim=64,
            n_heads=12,
            n_kv_heads=4,
            expert_dim=3072,
            vocab_size=48_000,
        ),
        192,
        "none",
    ),
    _ModelSpec(
        "qwen3_moe",
        "30B-3B",
        dict(
            n_layers=4,
            d_model=1024,
            n_heads=16,
            n_kv_heads=4,
            expert_dim=512,
            num_routed_experts=16,
            top_k=4,
            vocab_size=64_000,
        ),
        256,
        "muon",
    ),
    _ModelSpec(
        "olmoe",
        "7B-1B",
        dict(
            n_layers=3,
            d_model=512,
            head_dim=64,
            n_heads=8,
            n_kv_heads=8,
            expert_dim=256,
            num_routed_experts=8,
            top_k=2,
            vocab_size=32_000,
        ),
        192,
        "adamw",
    ),
    _ModelSpec(
        "qwen3_hybrid_dense",
        "9B",
        dict(
            n_layers=4,
            d_model=512,
            n_heads=8,
            n_kv_heads=2,
            expert_dim=1024,
            intermediate_size=1024,
            vocab_size=32_000,
            linear_num_key_heads=4,
            linear_num_value_heads=8,
        ),
        128,
        "adamw",
    ),
    _ModelSpec(
        "qwen3_hybrid_moe",
        "35B-A3B",
        dict(
            n_layers=4,
            d_model=512,
            n_heads=8,
            n_kv_heads=2,
            expert_dim=128,
            num_routed_experts=8,
            top_k=2,
            vocab_size=32_000,
            linear_num_key_heads=4,
            linear_num_value_heads=8,
        ),
        128,
        "muon",
    ),
    _ModelSpec(
        "deepseek_v3",
        "671B-37B",
        dict(
            n_layers=4,
            d_model=512,
            n_heads=8,
            n_kv_heads=8,
            intermediate_size=1024,
            expert_dim=128,
            num_routed_experts=8,
            top_k=2,
            vocab_size=32_000,
            q_lora_rank=128,
            kv_lora_rank=64,
            qk_nope_head_dim=32,
            qk_rope_head_dim=16,
            v_head_dim=32,
            head_dim=48,
        ),
        128,
        "adamw",
    ),
    _ModelSpec(
        "deepseek_v3_2",
        "671B-37B",
        dict(
            n_layers=4,
            d_model=512,
            n_heads=8,
            n_kv_heads=8,
            intermediate_size=1024,
            expert_dim=128,
            num_routed_experts=8,
            top_k=2,
            vocab_size=32_000,
            q_lora_rank=128,
            kv_lora_rank=64,
            qk_nope_head_dim=32,
            qk_rope_head_dim=16,
            v_head_dim=32,
            head_dim=48,
            index_n_heads=4,
            index_head_dim=32,
            index_topk=32,
        ),
        128,
        "adamw",
    ),
    _ModelSpec(
        "glm_5_2",
        "5.2",
        dict(
            n_layers=4,
            d_model=512,
            n_heads=8,
            n_kv_heads=8,
            intermediate_size=1024,
            expert_dim=128,
            num_routed_experts=8,
            top_k=2,
            vocab_size=32_000,
            q_lora_rank=128,
            kv_lora_rank=64,
            qk_nope_head_dim=32,
            qk_rope_head_dim=16,
            v_head_dim=32,
            head_dim=48,
            index_n_heads=4,
            index_head_dim=32,
            index_topk=32,
            index_topk_freq=2,
            index_skip_topk_offset=1,
        ),
        128,
        "adamw",
    ),
    _ModelSpec(
        "gpt_oss",
        "20B",
        dict(
            n_layers=4,
            d_model=512,
            n_heads=8,
            n_kv_heads=2,
            expert_dim=128,
            num_routed_experts=8,
            top_k=2,
            vocab_size=32_000,
            sliding_window=64,
        ),
        128,
        "adamw",
    ),
    _ModelSpec(
        "nemotron_h",
        "nano",
        dict(
            n_layers=3,
            d_model=512,
            head_dim=64,
            n_heads=8,
            n_kv_heads=2,
            expert_dim=128,
            shared_expert_dim=256,
            num_routed_experts=8,
            top_k=2,
            vocab_size=32_000,
            mamba_num_heads=8,
            mamba_head_dim=32,
            ssm_state_size=16,
            n_groups=2,
            intermediate_size=256,
            hybrid_override_pattern="M*E",
        ),
        128,
        "adamw",
    ),
)


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def _parse_csv_floats(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(not 0.0 < item <= 1.0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated ratios in (0, 1]")
    return values


def _parse_csv_strings(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare PressureFit feasibility and selected makespan for two git "
            "refs or source trees using identical serialized TaskChains."
        ),
    )
    parser.add_argument(
        "--baseline",
        default="HEAD",
        help="git ref, repository path, or WORKTREE (default: HEAD)",
    )
    parser.add_argument(
        "--candidate",
        default="WORKTREE",
        help="git ref, repository path, or WORKTREE (default: WORKTREE)",
    )
    parser.add_argument(
        "--suite",
        choices=("canaries", "synthetic", "models", "all"),
        default="all",
    )
    parser.add_argument(
        "--hardware",
        type=_parse_csv_strings,
        default=_DEFAULT_HARDWARE,
        help="comma-separated hardware presets used by the model suite",
    )
    parser.add_argument(
        "--model-ratios",
        type=_parse_csv_floats,
        default=_DEFAULT_RATIOS,
        help="comma-separated fractions of total logical object bytes",
    )
    parser.add_argument(
        "--synthetic-layers",
        type=_parse_csv_ints,
        default=_DEFAULT_SYNTHETIC_LAYERS,
    )
    parser.add_argument(
        "--synthetic-caps",
        type=_parse_csv_ints,
        default=_DEFAULT_SYNTHETIC_CAPS,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="optional directory for report.json and report.md",
    )
    parser.add_argument(
        "--abs-tolerance-us",
        type=float,
        default=0.0,
        help="allowed candidate makespan increase (default: exact non-regression)",
    )
    parser.add_argument(
        "--relative-tolerance",
        type=float,
        default=0.0,
        help="allowed relative makespan increase (default: exact non-regression)",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def _chain_sizes(chain: Any) -> dict[str, int]:
    sizes = {obj.id: obj.size for obj in chain.initial_memory}
    for task in chain.tasks:
        for output in task.outputs:
            sizes[output.id] = output.size
    return sizes


def _chain_payload(chain: Any) -> dict[str, Any]:
    """Serialize a TaskChain using the public ``from_dict`` trigger spelling."""
    payload = asdict(chain)
    for task in payload["tasks"]:
        for field in ("offload_after", "prefetch_after"):
            for trigger in task[field]:
                trigger["id"] = trigger.pop("obj_id")
    return payload


def _capacity_floors(chain: Any) -> tuple[int, int, int]:
    sizes = _chain_sizes(chain)
    total = sum(sizes.values())
    initial_fast = sum(
        obj.size for obj in chain.initial_memory if obj.location == "fast"
    )
    task0_floor = initial_fast
    if chain.tasks:
        first = chain.tasks[0]
        task0_floor += sum(
            sizes[oid]
            for oid in set(first.inputs)
            if not any(
                obj.id == oid and obj.location == "fast"
                for obj in chain.initial_memory
            )
        )
        task0_floor += sum(
            output.size for output in first.outputs if output.location == "fast"
        )
    task_floor = max(
        (
            sum(sizes[oid] for oid in set(task.inputs))
            + sum(
                output.size
                for output in task.outputs
                if output.location == "fast"
            )
            for task in chain.tasks
        ),
        default=initial_fast,
    )
    return total, task0_floor, task_floor


def _synthetic_training_chain(layers: int) -> Any:
    from dataflow_sim.core.schema import Object, OutputAlloc, Task, TaskChain

    initial = [Object(id="input", size=16, location="fast", type="activation")]
    for index in range(layers):
        initial.extend(
            (
                Object(
                    id=f"W_{index}", size=64, location="backing", type="weight"
                ),
                Object(
                    id=f"dW_{index}",
                    size=64,
                    location="backing",
                    type="gradient",
                ),
            ),
        )
    initial.extend(
        (
            Object(id="W_head", size=64, location="backing", type="weight"),
            Object(id="dW_head", size=64, location="backing", type="gradient"),
        ),
    )
    tasks = []
    for index in range(layers):
        input_id = "input" if index == 0 else f"y_{index - 1}"
        tasks.append(
            Task(
                id=f"f_{index}",
                inputs=[input_id, f"W_{index}"],
                outputs=[
                    OutputAlloc(f"A_{index}", 32, type="activation"),
                    OutputAlloc(f"y_{index}", 32, type="activation"),
                ],
                runtime=10,
            ),
        )
    tasks.append(
        Task(
            id="head",
            inputs=[f"y_{layers - 1}", "W_head", "dW_head"],
            outputs=[OutputAlloc("dy_head", 32, type="gradient")],
            runtime=2,
            mutates_inputs=["dW_head"],
        ),
    )
    for index in range(layers - 1, -1, -1):
        upstream = "dy_head" if index == layers - 1 else f"dy_{index + 1}"
        tasks.extend(
            (
                Task(
                    id=f"r_{index}",
                    inputs=[f"A_{index}", f"W_{index}"],
                    outputs=[],
                    runtime=0,
                ),
                Task(
                    id=f"b_{index}",
                    inputs=[
                        upstream,
                        f"A_{index}",
                        f"W_{index}",
                        f"dW_{index}",
                    ],
                    outputs=[OutputAlloc(f"dy_{index}", 32, type="gradient")],
                    runtime=20,
                    mutates_inputs=[f"dW_{index}"],
                ),
            ),
        )
    return TaskChain(
        initial_memory=initial,
        tasks=tasks,
        bandwidth_from_slow=8,
        bandwidth_to_slow=8,
    )


def _build_synthetic_scenarios(
    layers: tuple[int, ...],
    capacities: tuple[int, ...],
) -> list[_Scenario]:
    scenarios = []
    for layer_count in layers:
        chain = _synthetic_training_chain(layer_count)
        chain_payload = _chain_payload(chain)
        for capacity in capacities:
            scenarios.append(
                _Scenario(
                    scenario_id=f"synthetic/L{layer_count}/cap-{capacity}",
                    group="synthetic",
                    metadata={"layers": layer_count},
                    chain=chain_payload,
                    capacity_bytes=capacity,
                ),
            )
    return scenarios


def _build_canary_scenarios() -> list[_Scenario]:
    """Return byte-exact cases in which the terminal-residency fix matters."""
    from dataflow_sim.core.schema import Object, OutputAlloc, Task, TaskChain

    final_fast_gap = TaskChain(
        initial_memory=[
            Object(id="retained", size=61, location="fast"),
            Object(id="later", size=61, location="fast"),
        ],
        tasks=[
            Task(id="task0", inputs=["retained"], outputs=[], runtime=1),
            Task(
                id="task1",
                inputs=[],
                outputs=[OutputAlloc("temporary", 61)],
                runtime=1,
            ),
            Task(id="task2", inputs=["later"], outputs=[], runtime=1),
        ],
        final_locations={"retained": "fast"},
        backing_memory_capacity=1_000,
        bandwidth_from_slow=100,
        bandwidth_to_slow=100,
    )
    backing_only_terminal = TaskChain(
        initial_memory=[Object(id="state", size=10, location="backing")],
        tasks=[
            Task(id="task0", inputs=[], outputs=[], runtime=1),
            Task(id="task1", inputs=[], outputs=[], runtime=1),
        ],
        final_locations={"state": "fast"},
        backing_memory_capacity=20,
        bandwidth_from_slow=10,
        bandwidth_to_slow=10,
    )
    produced_terminal = TaskChain(
        initial_memory=[],
        tasks=[
            Task(
                id="produce",
                inputs=[],
                outputs=[OutputAlloc("result", 10)],
                runtime=1,
            ),
            Task(id="tail", inputs=[], outputs=[], runtime=1),
        ],
        final_locations={"result": "fast"},
        backing_memory_capacity=20,
        bandwidth_from_slow=10,
        bandwidth_to_slow=10,
    )
    definitions = (
        (
            "final-fast-initial-gap",
            final_fast_gap,
            (122, 121),
            "former duplicate-prefetch reproducer and its proven lower bound",
        ),
        (
            "backing-only-terminal-fast",
            backing_only_terminal,
            (10,),
            "terminal-fast object with no ordinary consumer",
        ),
        (
            "produced-unused-terminal-fast",
            produced_terminal,
            (10,),
            "produced terminal-fast object with no ordinary consumer",
        ),
    )
    scenarios = []
    for name, chain, capacities, purpose in definitions:
        chain_payload = _chain_payload(chain)
        for capacity in capacities:
            scenarios.append(
                _Scenario(
                    scenario_id=f"canaries/{name}/cap-{capacity}",
                    group="canaries",
                    metadata={"purpose": purpose},
                    chain=chain_payload,
                    capacity_bytes=capacity,
                ),
            )
    return scenarios


def _build_model_scenarios(
    hardware_names: tuple[str, ...],
    ratios: tuple[float, ...],
) -> list[_Scenario]:
    from dataflow_sim.workloads.common.hardware import HARDWARE_PRESETS
    from dataflow_sim.workloads.dataflow_builder import TrainingConfig
    from dataflow_sim.workloads.models.registry import MODEL_FAMILIES

    unknown_hardware = sorted(set(hardware_names) - set(HARDWARE_PRESETS))
    if unknown_hardware:
        raise ValueError(f"unknown hardware presets: {unknown_hardware}")

    scenarios = []
    for hardware_name in hardware_names:
        hardware = HARDWARE_PRESETS[hardware_name]
        for spec in _MODEL_SPECS:
            entry = MODEL_FAMILIES[spec.family]
            config = entry.config_cls.preset(spec.scale, **spec.overrides)
            model = entry.builder_cls(config)
            training = TrainingConfig(
                seqlen=spec.seqlen,
                num_seqs=1,
                optimizer=spec.optimizer,
            )
            chain = model.build_training_workload(training, hardware).chain
            total, task0_floor, task_floor = _capacity_floors(chain)
            capacities = {max(1, int(total * ratio)) for ratio in ratios}
            capacities.update((max(1, task0_floor - 1), task0_floor))
            chain_payload = _chain_payload(chain)
            for capacity in sorted(capacities, reverse=True):
                scenarios.append(
                    _Scenario(
                        scenario_id=(
                            f"models/{spec.family}/{hardware_name}/cap-{capacity}"
                        ),
                        group="models",
                        metadata={
                            "family": spec.family,
                            "hardware": hardware_name,
                            "seqlen": spec.seqlen,
                            "optimizer": spec.optimizer,
                            "object_bytes": total,
                            "task0_floor_bytes": task0_floor,
                            "task_io_floor_bytes": task_floor,
                        },
                        chain=chain_payload,
                        capacity_bytes=capacity,
                    ),
                )
    return scenarios


def _build_scenarios(args: argparse.Namespace) -> list[_Scenario]:
    source_path = str(_ROOT / "src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    scenarios = []
    if args.suite in ("canaries", "all"):
        scenarios.extend(_build_canary_scenarios())
    if args.suite in ("synthetic", "all"):
        scenarios.extend(
            _build_synthetic_scenarios(
                args.synthetic_layers,
                args.synthetic_caps,
            ),
        )
    if args.suite in ("models", "all"):
        scenarios.extend(_build_model_scenarios(args.hardware, args.model_ratios))
    return scenarios


def _status_for_error(error: Exception) -> _Status:
    return "infeasible" if str(error).startswith("infeasible:") else "invalid"


def _worker_plan(payload: dict[str, Any]) -> dict[str, Any]:
    from dataflow_sim.core.schema import TaskChain
    from dataflow_sim.policies.pressurefit import plan_pressurefit_policy

    results = []
    for scenario in payload["scenarios"]:
        chain = TaskChain.from_dict(scenario["chain"])
        try:
            annotated, diagnostics = plan_pressurefit_policy(
                chain,
                fast_memory_capacity=int(scenario["capacity_bytes"]),
            )
            results.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "planner_status": "valid",
                    "planner_makespan_us": diagnostics.selected_makespan_us,
                    "planning_time_s": diagnostics.planning_time_s,
                    "candidate_count": diagnostics.candidate_count,
                    "valid_candidate_count": diagnostics.valid_candidate_count,
                    "selected_candidate": diagnostics.selected_candidate,
                    "chain": _chain_payload(annotated),
                },
            )
        except Exception as error:
            results.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "planner_status": _status_for_error(error),
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
    return {"results": results}


def _worker_evaluate(payload: dict[str, Any]) -> dict[str, Any]:
    from dataflow_sim.core.schema import TaskChain
    from dataflow_sim.engine.simulator import run

    results = []
    for plan in payload["plans"]:
        try:
            chain = TaskChain.from_dict(plan["chain"])
            log = run(chain, snapshots=False)
            makespan = max(
                (interval.end for interval in log.task_intervals),
                default=0,
            )
            results.append(
                {
                    "plan_id": plan["plan_id"],
                    "oracle_status": "valid",
                    "makespan_us": makespan,
                    "peak_fast_bytes": log.peak_fast_memory_bytes,
                    "peak_backing_bytes": log.peak_backing_memory_bytes,
                    "offload_count": sum(
                        len(task.offload_after) for task in chain.tasks
                    ),
                    "prefetch_count": sum(
                        len(task.prefetch_after) for task in chain.tasks
                    ),
                },
            )
        except Exception as error:
            results.append(
                {
                    "plan_id": plan["plan_id"],
                    "oracle_status": "invalid",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
    return {"results": results}


def _worker_main() -> int:
    payload = json.load(sys.stdin)
    action = payload.get("action")
    if action == "plan":
        result = _worker_plan(payload)
    elif action == "evaluate":
        result = _worker_evaluate(payload)
    else:
        raise ValueError(f"unknown worker action {action!r}")
    json.dump(result, sys.stdout, sort_keys=True)
    return 0


def _safe_extract_tar(archive: tarfile.TarFile, destination: Path) -> None:
    destination_resolved = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if destination_resolved not in target.parents and target != destination_resolved:
            raise ValueError(f"unsafe archive member {member.name!r}")
    archive.extractall(destination, filter="data")


@contextmanager
def _materialize_source(spec: str) -> Iterator[tuple[Path, dict[str, Any]]]:
    if spec == "WORKTREE":
        yield _ROOT, _source_identity(_ROOT, spec)
        return
    candidate_path = Path(spec).expanduser()
    if candidate_path.exists():
        source = candidate_path.resolve()
        yield source, _source_identity(source, spec)
        return

    resolved = subprocess.run(
        ["git", "-C", str(_ROOT), "rev-parse", f"{spec}^{{commit}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    with tempfile.TemporaryDirectory(prefix="pressurefit-quality-") as temp:
        destination = Path(temp)
        archive_bytes = subprocess.run(
            ["git", "-C", str(_ROOT), "archive", resolved],
            check=True,
            capture_output=True,
        ).stdout
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as archive:
            _safe_extract_tar(archive, destination)
        yield destination, {
            "spec": spec,
            "kind": "git-ref",
            "resolved_commit": resolved,
        }


def _source_identity(source: Path, spec: str) -> dict[str, Any]:
    identity: dict[str, Any] = {"spec": spec, "kind": "source-tree", "path": str(source)}
    try:
        identity["resolved_commit"] = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        identity["dirty"] = bool(
            subprocess.run(
                ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        pass
    return identity


def _run_worker(source: Path, payload: dict[str, Any]) -> dict[str, Any]:
    environment = dict(os.environ)
    source_path = str(source / "src")
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_path if not existing else os.pathsep.join((source_path, existing))
    )
    process = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker"],
        cwd=source,
        env=environment,
        input=json.dumps(payload, separators=(",", ":")),
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"worker failed for {source} (exit {process.returncode}):\n"
            f"{process.stderr.strip()}"
        )
    return json.loads(process.stdout)


def _chain_digest(chain: dict[str, Any]) -> str:
    canonical = json.dumps(chain, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _compare(
    scenarios: list[_Scenario],
    baseline_source: Path,
    candidate_source: Path,
    *,
    abs_tolerance_us: float,
    relative_tolerance: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    scenario_payloads = [scenario.payload() for scenario in scenarios]
    request = {"action": "plan", "scenarios": scenario_payloads}
    baseline_plans = _run_worker(baseline_source, request)["results"]
    candidate_plans = _run_worker(candidate_source, request)["results"]

    baseline_by_id = {row["scenario_id"]: row for row in baseline_plans}
    candidate_by_id = {row["scenario_id"]: row for row in candidate_plans}
    oracle_request = []
    for label, plan_rows in (
        ("baseline", baseline_by_id),
        ("candidate", candidate_by_id),
    ):
        for scenario_id, row in plan_rows.items():
            if row["planner_status"] == "valid":
                oracle_request.append(
                    {
                        "plan_id": f"{label}:{scenario_id}",
                        "chain": row["chain"],
                    },
                )
    oracle_rows = _run_worker(
        candidate_source,
        {"action": "evaluate", "plans": oracle_request},
    )["results"]
    oracle_by_id = {row["plan_id"]: row for row in oracle_rows}

    summary = {
        "equal": 0,
        "makespan_improvement": 0,
        "feasibility_improvement": 0,
        "correctness_improvement": 0,
        "both_nonvalid": 0,
        "makespan_regression": 0,
        "feasibility_regression": 0,
        "candidate_invalid": 0,
    }
    rows = []
    for scenario in scenarios:
        scenario_id = scenario.scenario_id
        baseline_plan = dict(baseline_by_id[scenario_id])
        candidate_plan = dict(candidate_by_id[scenario_id])
        baseline_chain = baseline_plan.pop("chain", None)
        candidate_chain = candidate_plan.pop("chain", None)
        if baseline_chain is not None:
            baseline_plan["chain_digest"] = _chain_digest(baseline_chain)
            baseline_oracle = oracle_by_id[f"baseline:{scenario_id}"]
        else:
            baseline_oracle = None
        if candidate_chain is not None:
            candidate_plan["chain_digest"] = _chain_digest(candidate_chain)
            candidate_oracle = oracle_by_id[f"candidate:{scenario_id}"]
        else:
            candidate_oracle = None

        classification: str
        regression = False
        delta_us = None
        delta_pct = None
        if candidate_oracle is not None and candidate_oracle["oracle_status"] != "valid":
            classification = "candidate_invalid"
            regression = True
        elif baseline_oracle is not None and baseline_oracle["oracle_status"] == "valid":
            if candidate_oracle is None:
                classification = "feasibility_regression"
                regression = True
            else:
                old_time = float(baseline_oracle["makespan_us"])
                new_time = float(candidate_oracle["makespan_us"])
                delta_us = new_time - old_time
                delta_pct = (delta_us / old_time * 100.0) if old_time else 0.0
                allowed = max(abs_tolerance_us, old_time * relative_tolerance)
                if delta_us > allowed:
                    classification = "makespan_regression"
                    regression = True
                elif delta_us < 0:
                    classification = "makespan_improvement"
                else:
                    classification = "equal"
        elif baseline_oracle is not None:
            classification = (
                "correctness_improvement"
                if candidate_oracle is not None
                else "both_nonvalid"
            )
        elif candidate_oracle is not None:
            classification = "feasibility_improvement"
        else:
            classification = "both_nonvalid"
        summary[classification] += 1

        rows.append(
            {
                "scenario_id": scenario_id,
                "group": scenario.group,
                "metadata": scenario.metadata,
                "capacity_bytes": scenario.capacity_bytes,
                "bare_chain_digest": _chain_digest(scenario.chain),
                "baseline": {
                    "planner": baseline_plan,
                    "candidate_oracle": baseline_oracle,
                },
                "candidate": {
                    "planner": candidate_plan,
                    "candidate_oracle": candidate_oracle,
                },
                "classification": classification,
                "regression": regression,
                "makespan_delta_us": delta_us,
                "makespan_delta_pct": delta_pct,
            },
        )
    return rows, summary


def _markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    regressions = [row for row in report["rows"] if row["regression"]]
    improvements = sorted(
        (
            row
            for row in report["rows"]
            if row["classification"] == "makespan_improvement"
        ),
        key=lambda row: row["makespan_delta_pct"],
    )[:20]
    feasibility_improvements = [
        row
        for row in report["rows"]
        if row["classification"] in (
            "feasibility_improvement",
            "correctness_improvement",
        )
    ]
    lines = [
        "# PressureFit planning-quality comparison",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Baseline: `{report['baseline']['spec']}`",
        f"- Candidate: `{report['candidate']['spec']}`",
        f"- Scenarios: `{report['scenario_count']}`",
        f"- Acceptance: **{'PASS' if not regressions else 'FAIL'}**",
        "",
        "## Summary",
        "",
        "| Classification | Count |",
        "|---|---:|",
    ]
    for key, count in summary.items():
        lines.append(f"| `{key}` | {count} |")
    lines.extend(
        (
            "",
            "## Regressions",
            "",
            "| Scenario | Capacity | Classification | Old us | New us | Delta |",
            "|---|---:|---|---:|---:|---:|",
        ),
    )
    if not regressions:
        lines.append("| _none_ | | | | | |")
    for row in regressions:
        baseline = row["baseline"]["candidate_oracle"] or {}
        candidate = row["candidate"]["candidate_oracle"] or {}
        lines.append(
            "| {scenario} | {cap} | `{kind}` | {old} | {new} | {delta} |".format(
                scenario=row["scenario_id"],
                cap=row["capacity_bytes"],
                kind=row["classification"],
                old=baseline.get("makespan_us", ""),
                new=candidate.get("makespan_us", ""),
                delta=row["makespan_delta_us"] or "",
            ),
        )
    lines.extend(
        (
            "",
            "## Largest makespan improvements",
            "",
            "| Scenario | Capacity | Old us | New us | Delta us | Delta % |",
            "|---|---:|---:|---:|---:|---:|",
        ),
    )
    if not improvements:
        lines.append("| _none_ | | | | | |")
    for row in improvements:
        baseline = row["baseline"]["candidate_oracle"]
        candidate = row["candidate"]["candidate_oracle"]
        lines.append(
            "| {scenario} | {cap} | {old} | {new} | {delta} | {pct:.3f}% |".format(
                scenario=row["scenario_id"],
                cap=row["capacity_bytes"],
                old=baseline["makespan_us"],
                new=candidate["makespan_us"],
                delta=row["makespan_delta_us"],
                pct=row["makespan_delta_pct"],
            ),
        )
    lines.extend(
        (
            "",
            "## Newly valid scenarios",
            "",
            "| Scenario | Capacity | Classification | Old status | New us |",
            "|---|---:|---|---|---:|",
        ),
    )
    if not feasibility_improvements:
        lines.append("| _none_ | | | | |")
    for row in feasibility_improvements:
        old_status = row["baseline"]["planner"]["planner_status"]
        candidate = row["candidate"]["candidate_oracle"] or {}
        lines.append(
            "| {scenario} | {cap} | `{kind}` | `{old}` | {new} |".format(
                scenario=row["scenario_id"],
                cap=row["capacity_bytes"],
                kind=row["classification"],
                old=old_status,
                new=candidate.get("makespan_us", ""),
            ),
        )
    lines.extend(
        (
            "",
            "The JSON companion contains every scenario, source identity, bare/",
            "planned-chain digest, planner result, candidate-oracle replay result,",
            "and exact makespan delta.",
            "",
        ),
    )
    return "\n".join(lines)


def _write_report(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    (output_dir / "report.md").write_text(_markdown_report(report))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.worker:
        return _worker_main()
    if args.abs_tolerance_us < 0 or args.relative_tolerance < 0:
        raise ValueError("quality tolerances cannot be negative")

    scenarios = _build_scenarios(args)
    with ExitStack() as stack:
        baseline_source, baseline_identity = stack.enter_context(
            _materialize_source(args.baseline),
        )
        candidate_source, candidate_identity = stack.enter_context(
            _materialize_source(args.candidate),
        )
        rows, summary = _compare(
            scenarios,
            baseline_source,
            candidate_source,
            abs_tolerance_us=args.abs_tolerance_us,
            relative_tolerance=args.relative_tolerance,
        )

    regressions = sum(1 for row in rows if row["regression"])
    report = {
        "schema_version": _SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scenario_generator": str(Path(__file__).resolve()),
        "baseline": baseline_identity,
        "candidate": candidate_identity,
        "suite": args.suite,
        "hardware": list(args.hardware),
        "model_ratios": list(args.model_ratios),
        "synthetic_layers": list(args.synthetic_layers),
        "synthetic_caps": list(args.synthetic_caps),
        "abs_tolerance_us": args.abs_tolerance_us,
        "relative_tolerance": args.relative_tolerance,
        "scenario_count": len(rows),
        "summary": summary,
        "regression_count": regressions,
        "rows": rows,
    }
    print(_markdown_report(report))
    if args.output_dir is not None:
        _write_report(report, args.output_dir)
        print(f"Reports: {args.output_dir / 'report.json'}, {args.output_dir / 'report.md'}")
    return 1 if regressions else 0


if __name__ == "__main__":
    raise SystemExit(main())
