"""Generic contracts and scheduler for workload-level training builders.

Model files are responsible for defining architecture: which modules appear,
how many there are, and in what order. This builder only schedules an ordered
list of trainable module specs into `DataflowProgram v1`.

Current scheduler shape:
- canonical activation tensors are 2-D: `[tokens, dim]`;
- each layer-like spec has forward, backward, and recompute phase op factories;
- one head/loss spec runs after the ordered layer list;
- saved backward-context objects are named `A_<step>_<round>_<layer>`;
- optional recompute levels are keyed by those `A_*` object ids; and
- optimizer tasks run per layer after each step's grad accumulation.

Those rules are generic to a stacked training workload, not to a specific
model family. Built-in model files instantiate this with their own block/head
modules; future model files can pass different module specs.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
import math
from typing import Any, Literal, Protocol, runtime_checkable

from dataflow_sim.workloads.common.hardware import HardwareSpec
from dataflow_sim.workloads.common.recompute import RecomputeOption, RecomputeRewrite
from dataflow_sim.workloads.common.workload import Workload
from dataflow_sim.workloads.dataflow import (
    DataflowCost,
    DataflowMetrics,
    DataflowProgram,
    realize_dataflow_program,
    resolve_cost,
)
from dataflow_sim.workloads.dataflow_builder import (
    DTypePolicy,
    OpDTypePolicy,
    ParallelismConfig,
    TensorRef,
    TraceContext,
    TrainingConfig,
    dtype_nbytes,
)
from dataflow_sim.workloads.ops.forward import gather as gather_cost
from dataflow_sim.workloads.ops.forward import memory as memory_cost
from dataflow_sim.workloads.ops.optimizer import adamw_step, optimizer_state_bytes


LayerOpsFactory = Callable[[int, int, OpDTypePolicy], list[DataflowCost]]
HeadOpsFactory = Callable[[int, OpDTypePolicy], list[DataflowCost]]
OptimizerOpsFactory = Callable[[str, OpDTypePolicy], list[DataflowCost]]
OptimizerStateFactor = Callable[[str], int]
ParamBytesFactory = Callable[[DTypePolicy, ParallelismConfig], int]
GradientBytesFactory = Callable[[DTypePolicy, ParallelismConfig], int]
OptimizerStateBytesFactory = Callable[[str, DTypePolicy, ParallelismConfig], int]
SavedActivationBytesFactory = Callable[[int, int, DTypePolicy], int]


@runtime_checkable
class TrainingProgramBuilder(Protocol):
    """Anything that can emit a hardware-independent training program.

    Required inputs:
    - `training`: loop/optimizer settings.
    - `input_shape`: optional canonical activation shape expected by the
      builder. Builders should validate this when supplied.
    - `name`: optional program name override.
    - `dtype_policy`: optional role-wise dtype policy; default should be bf16.
    - `parallelism`: optional workload parallelism settings.
    - `recompute`: optional map of saved activation/object id to recompute
      level. Builders define which ids and levels are valid.
    """

    def build_training_program(
        self,
        training: TrainingConfig,
        *,
        input_shape: tuple[int, int] | None = None,
        name: str | None = None,
        dtype_policy: DTypePolicy | None = None,
        parallelism: ParallelismConfig | None = None,
        recompute: Mapping[str, int] | None = None,
    ) -> DataflowProgram:
        ...


@runtime_checkable
class TrainingWorkloadBuilder(TrainingProgramBuilder, Protocol):
    """A training builder that can also realize a program on hardware."""

    input_dim: int

    def build_training_workload(
        self,
        training: TrainingConfig,
        hw: HardwareSpec,
        *,
        input_shape: tuple[int, int] | None = None,
        name: str | None = None,
        dtype_policy: DTypePolicy | None = None,
        parallelism: ParallelismConfig | None = None,
        recompute: Mapping[str, int] | None = None,
    ) -> Workload:
        ...


def validate_training_config(training: TrainingConfig) -> None:
    """Validate loop counts shared by all training workload builders."""
    if training.num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    if training.grad_accum_rounds < 1:
        raise ValueError("grad_accum_rounds must be >= 1")
    if training.optimizer_placement not in ("interleaved", "tail"):
        raise ValueError(
            "optimizer_placement must be 'interleaved' or 'tail', "
            f"got {training.optimizer_placement!r}"
        )


def validate_input_shape(
    input_shape: tuple[int, int] | None,
    expected_shape: tuple[int, int],
) -> None:
    """Validate an optional canonical 2-D activation shape."""
    if input_shape is not None and input_shape != expected_shape:
        raise ValueError(f"input_shape must be {expected_shape}, got {input_shape}")


def selected_recompute_ids(
    valid_object_ids: Iterable[str],
    recompute: Mapping[str, int] | None = None,
) -> set[str]:
    """Return object ids selected for level-1 recompute.

    The generic contract supports two levels for now:
    - `0`: save the object from forward,
    - `1`: do not save it; recompute it before backward.
    """
    levels = dict(recompute or {})
    selected: set[str] = set()
    for object_id in valid_object_ids:
        level = levels.pop(object_id, 0)
        if level not in (0, 1):
            raise ValueError(
                f"unsupported recompute level {level} for {object_id!r}"
            )
        if level == 1:
            selected.add(object_id)
    if levels:
        raise ValueError(f"unknown recompute object ids: {sorted(levels)}")
    return selected


def default_optimizer_state_factor(optimizer: str) -> int:
    if optimizer == "adamw":
        return 2
    if optimizer == "muon":
        return 1
    if optimizer in {"none", "sgd"}:
        return 0
    raise ValueError(f"unknown optimizer mode: {optimizer!r}")


def optimizer_display_name(optimizer: str) -> str:
    labels = {
        "adamw": "AdamW",
        "muon": "Muon",
        "sgd": "SGD",
        "none": "None",
    }
    return labels.get(optimizer, optimizer.upper())


@dataclass(frozen=True)
class TrainingLayerSpec:
    """One model-authored layer/module entry in a training stack."""

    name: str
    input_dim: int
    output_dim: int
    param_count: int
    saved_activation_width: int
    forward_ops: LayerOpsFactory
    backward_ops: LayerOpsFactory
    recompute_ops: LayerOpsFactory
    optimizer_ops: OptimizerOpsFactory
    gradient_count: int | None = None
    optimizer_state_factor: OptimizerStateFactor = default_optimizer_state_factor
    parameter_bytes: ParamBytesFactory | None = None
    gradient_bytes: GradientBytesFactory | None = None
    optimizer_state_bytes: OptimizerStateBytesFactory | None = None
    saved_activation_bytes: SavedActivationBytesFactory | None = None
    block_key: str = "layer"
    block_name: str = "Layer"
    optimizer_block_key: str = "optimizer_step"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def gradient_param_count(self) -> int:
        return self.param_count if self.gradient_count is None else self.gradient_count


@dataclass(frozen=True)
class TrainingHeadSpec:
    """The post-stack head/loss module for a training workload."""

    name: str
    input_dim: int
    param_count: int
    forward_ops: HeadOpsFactory
    backward_ops: HeadOpsFactory
    block_key: str = "head"
    block_name: str = "Head"
    optimizer_ops: OptimizerOpsFactory | None = None
    optimizer_state_factor: OptimizerStateFactor = default_optimizer_state_factor
    parameter_bytes: ParamBytesFactory | None = None
    optimizer_state_bytes: OptimizerStateBytesFactory | None = None
    optimizer_block_key: str = "lm_head.optimizer_step"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingEmbeddingSpec:
    """The token-embedding module for a training workload.

    Mirrors the runtime's true chain shape: tokens stream in as int32
    ids, ``embed_fwd`` gathers rows of W_embed, ``embed_bwd`` is the
    LAST backward task of every round (scatter-accumulate into
    dW_embed), and the embedding optimizer closes the step in
    interleaved placement.

    ``tied=True`` models tied-embedding families (e.g. qwen3.5): ONE
    W_embed/O_embed/dW_embed triple serves both the embedding and the
    LM head — head_loss creates the shared gradient in round 0,
    embed_bwd accumulates into it (round 0 included), and the single
    embedding optimizer covers both; no separate W_head/O_head or head
    optimizer task exists.
    """

    name: str
    output_dim: int
    param_count: int
    forward_ops: HeadOpsFactory
    backward_ops: HeadOpsFactory
    tied: bool = False
    optimizer_ops: OptimizerOpsFactory | None = None
    optimizer_state_factor: OptimizerStateFactor = default_optimizer_state_factor
    parameter_bytes: ParamBytesFactory | None = None
    optimizer_state_bytes: OptimizerStateBytesFactory | None = None
    block_key: str = "embedding"
    block_name: str = "Embedding"
    optimizer_block_key: str = "embedding.optimizer_step"
    metadata: dict[str, Any] = field(default_factory=dict)


def auto_embedding_spec(
    head: TrainingHeadSpec,
    input_dim: int,
) -> TrainingEmbeddingSpec:
    """Derive an untied embedding from the head: an LM family's embedding
    table has the head's param count (vocab x d_model) and memory-bound
    gather/scatter costs (2.t.d fwd; 2x that for the scatter-accumulate,
    which read-modify-writes the touched dW rows)."""

    def fwd_ops(tokens: int, bpe: Any) -> list[DataflowCost]:
        return [gather_cost(
            "embedding_gather", tokens=tokens, dim=input_dim, fanin=1,
        )]

    def bwd_ops(tokens: int, bpe: Any) -> list[DataflowCost]:
        return [memory_cost(
            "embedding_scatter_accum",
            bytes_total=4 * tokens * input_dim * 2,
        )]

    return TrainingEmbeddingSpec(
        name="embedding",
        output_dim=input_dim,
        param_count=head.param_count,
        forward_ops=fwd_ops,
        backward_ops=bwd_ops,
    )


class TrainingBuilder:
    """Schedule an architecture-specified module list into a training program."""

    def __init__(
        self,
        *,
        family_name: str,
        metadata_kind: str,
        preset_name: str,
        layers: Sequence[TrainingLayerSpec],
        head: TrainingHeadSpec,
        embedding: TrainingEmbeddingSpec | str | None = "auto",
        model_metadata: dict[str, Any] | None = None,
    ) -> None:
        if not layers:
            raise ValueError("training builder requires at least one layer")
        if embedding == "auto":
            embedding = auto_embedding_spec(head, layers[0].input_dim)
        elif isinstance(embedding, str):
            raise ValueError(f"embedding must be a spec, 'auto', or None; got {embedding!r}")
        self.embedding = embedding
        if head.input_dim != layers[-1].output_dim:
            raise ValueError("head input_dim must match final layer output_dim")
        for left, right in zip(layers, layers[1:]):
            if left.output_dim != right.input_dim:
                raise ValueError(
                    f"{left.name}.output_dim={left.output_dim} does not match "
                    f"{right.name}.input_dim={right.input_dim}"
                )
        self.family_name = family_name
        self.metadata_kind = metadata_kind
        self.preset_name = preset_name
        self.layers = tuple(layers)
        self.head = head
        self.model_metadata = dict(model_metadata or {})
        self.input_dim = self.layers[0].input_dim

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @staticmethod
    def _parameter_bytes(
        spec: TrainingLayerSpec | TrainingHeadSpec | TrainingEmbeddingSpec,
        policy: DTypePolicy,
        parallelism: ParallelismConfig,
    ) -> int:
        if spec.parameter_bytes is not None:
            return spec.parameter_bytes(policy, parallelism)
        return math.ceil(spec.param_count * dtype_nbytes(policy.param))

    @staticmethod
    def _gradient_bytes(
        spec: TrainingLayerSpec | TrainingHeadSpec | TrainingEmbeddingSpec,
        policy: DTypePolicy,
        parallelism: ParallelismConfig,
    ) -> int:
        if isinstance(spec, TrainingLayerSpec) and spec.gradient_bytes is not None:
            return spec.gradient_bytes(policy, parallelism)
        count = (
            spec.gradient_param_count
            if isinstance(spec, TrainingLayerSpec)
            else spec.param_count
        )
        return math.ceil(count * dtype_nbytes(policy.gradient))

    @staticmethod
    def _optimizer_state_bytes(
        spec: TrainingLayerSpec | TrainingHeadSpec | TrainingEmbeddingSpec,
        optimizer: str,
        policy: DTypePolicy,
        parallelism: ParallelismConfig,
    ) -> int:
        if spec.optimizer_state_bytes is not None:
            return spec.optimizer_state_bytes(optimizer, policy, parallelism)
        return optimizer_state_bytes(
            math.ceil(spec.param_count * dtype_nbytes(policy.optimizer_state)),
            optimizer,
        )

    @staticmethod
    def _saved_activation_bytes(
        spec: TrainingLayerSpec,
        tokens: int,
        seqlen: int,
        policy: DTypePolicy,
    ) -> int:
        if spec.saved_activation_bytes is not None:
            return spec.saved_activation_bytes(tokens, seqlen, policy)
        return math.ceil(
            tokens * spec.saved_activation_width * dtype_nbytes(policy.activation)
        )

    def build_training_program(
        self,
        training: TrainingConfig,
        *,
        input_shape: tuple[int, int] | None = None,
        name: str | None = None,
        dtype_policy: DTypePolicy | None = None,
        parallelism: ParallelismConfig | None = None,
        recompute: Mapping[str, int] | None = None,
    ) -> DataflowProgram:
        """Build a hardware-independent training `DataflowProgram`.

        Exact inputs:
        - `training.tokens = seqlen * num_seqs` sets the first tensor dimension.
        - `input_shape`, when supplied, must be `[training.tokens, input_dim]`.
        - `dtype_policy` controls object sizes for params, activations,
          parameter gradients, and optimizer state; defaults are bf16.
        - `recompute` maps saved activation ids like `A_0_0_3` to level 0/1.
        """

        validate_training_config(training)
        validate_input_shape(input_shape, (training.tokens, self.input_dim))

        policy = dtype_policy or DTypePolicy()
        parallel = parallelism or ParallelismConfig()
        param_bytes = dtype_nbytes(policy.param)
        activation_bytes = dtype_nbytes(policy.activation)
        expert_dispatch_bytes = dtype_nbytes(policy.expert_dispatch)
        param_grad_bytes = dtype_nbytes(policy.gradient)
        opt_bytes = dtype_nbytes(policy.optimizer_state)
        if any(
            value <= 0
            for value in (param_bytes, activation_bytes, param_grad_bytes, opt_bytes)
        ) or expert_dispatch_bytes <= 0:
            raise ValueError("dtype byte sizes must be positive")

        tokens = training.tokens
        seqlen = training.seqlen

        op_policy = OpDTypePolicy.from_dtype_policy(policy, parallel)

        ctx = TraceContext(
            name=name or f"{self.family_name}-{self.preset_name}-training",
            dtype_policy=policy,
            description=f"{self.family_name} training workload built by dataflow_builder",
        )

        embedding = self.embedding
        tied = embedding is not None and embedding.tied
        for k in range(training.num_steps):
            for j in range(training.grad_accum_rounds):
                if embedding is not None:
                    ctx.initial_tensor(
                        f"tokens_{k}_{j}",
                        (tokens, 1),
                        role="activation",
                        initial_location="fast" if k == 0 and j == 0 else "backing",
                        dtype="int32",
                    )
                else:
                    ctx.initial_tensor(
                        f"input_{k}_{j}",
                        (tokens, self.input_dim),
                        role="activation",
                        initial_location="fast" if k == 0 and j == 0 else "backing",
                        dtype=policy.activation,
                    )
                # next-token targets for the fused head+loss task
                ctx.initial_tensor(
                    f"targets_{k}_{j}",
                    (tokens, 1),
                    role="activation",
                    initial_location="fast" if k == 0 and j == 0 else "backing",
                    dtype="int32",
                )
        for i, layer in enumerate(self.layers):
            layer_param_bytes = self._parameter_bytes(layer, policy, parallel)
            ctx.initial_tensor(
                f"W_{i}",
                (layer.param_count, 1),
                role="parameter",
                initial_location="backing",
                dtype=policy.param,
                size_bytes=layer_param_bytes,
            )
            state_bytes = self._optimizer_state_bytes(
                layer,
                training.optimizer,
                policy,
                parallel,
            )
            if state_bytes > 0:
                ctx.initial_tensor(
                    f"O_{i}",
                    (max(1, math.ceil(state_bytes / opt_bytes)), 1),
                    role="optimizer_state",
                    initial_location="backing",
                    dtype=policy.optimizer_state,
                    size_bytes=state_bytes,
                )
        embed_param_bytes = 0
        embed_state_bytes = 0
        if embedding is not None:
            embed_param_bytes = self._parameter_bytes(embedding, policy, parallel)
            ctx.initial_tensor(
                "W_embed",
                (embedding.param_count, 1),
                role="parameter",
                initial_location="backing",
                dtype=policy.param,
                size_bytes=embed_param_bytes,
            )
            embed_state_bytes = (
                self._optimizer_state_bytes(
                    embedding,
                    training.optimizer,
                    policy,
                    parallel,
                )
                if training.optimizer != "none"
                else 0
            )
            if embed_state_bytes > 0:
                ctx.initial_tensor(
                    "O_embed",
                    (max(1, math.ceil(embed_state_bytes / opt_bytes)), 1),
                    role="optimizer_state",
                    initial_location="backing",
                    dtype=policy.optimizer_state,
                    size_bytes=embed_state_bytes,
                )
        head_param_bytes = self._parameter_bytes(self.head, policy, parallel)
        if not tied:
            # tied embeddings: ONE W_embed serves both the embedding and
            # the LM head — no separate head weight exists
            ctx.initial_tensor(
                "W_head",
                (self.head.param_count, 1),
                role="parameter",
                initial_location="backing",
                dtype=policy.param,
                size_bytes=head_param_bytes,
            )
        head_optimizer_mode = "adamw" if training.optimizer != "none" else "none"
        # tied: the embedding optimizer covers the shared W_embed/O_embed
        head_optimizer_enabled = head_optimizer_mode != "none" and not tied
        head_state_bytes = (
            self._optimizer_state_bytes(
                self.head,
                head_optimizer_mode,
                policy,
                parallel,
            )
            if head_optimizer_enabled
            else 0
        )
        if head_state_bytes > 0:
            ctx.initial_tensor(
                "O_head",
                (max(1, math.ceil(head_state_bytes / opt_bytes)), 1),
                role="optimizer_state",
                initial_location="backing",
                dtype=policy.optimizer_state,
                size_bytes=head_state_bytes,
            )

        valid_recompute_ids = [
            layer_round_id("A", k, j, i)
            for k in range(training.num_steps)
            for j in range(training.grad_accum_rounds)
            for i in range(self.n_layers)
        ]
        recomputed = selected_recompute_ids(valid_recompute_ids, recompute)

        layer_phase_ops = [
            (
                layer.forward_ops(tokens, seqlen, op_policy),
                layer.backward_ops(tokens, seqlen, op_policy),
                layer.recompute_ops(tokens, seqlen, op_policy),
                layer.optimizer_ops(training.optimizer, op_policy),
            )
            for layer in self.layers
        ]
        head_forward_ops = self.head.forward_ops(tokens, op_policy)
        head_backward_ops = self.head.backward_ops(tokens, op_policy)
        if embedding is not None:
            embed_forward_ops = embedding.forward_ops(tokens, op_policy)
            embed_backward_ops = embedding.backward_ops(tokens, op_policy)
            if training.optimizer == "none":
                embed_optimizer_ops: list[DataflowCost] = []
            elif embedding.optimizer_ops is not None:
                embed_optimizer_ops = embedding.optimizer_ops(training.optimizer, op_policy)
            else:
                embed_optimizer_ops = [
                    adamw_step(
                        "adamw_step",
                        weight_bytes=embed_param_bytes,
                        gradient_bytes=math.ceil(
                            embedding.param_count * dtype_nbytes(policy.gradient)
                        ),
                        optimizer_state_bytes=embed_state_bytes,
                    )
                ]
        if not head_optimizer_enabled:
            head_optimizer_ops: list[DataflowCost] = []
        elif self.head.optimizer_ops is not None:
            head_optimizer_ops = self.head.optimizer_ops(head_optimizer_mode, op_policy)
        else:
            head_optimizer_ops = [
                adamw_step(
                    "adamw_step",
                    weight_bytes=head_param_bytes,
                    gradient_bytes=self._gradient_bytes(self.head, policy, parallel),
                    optimizer_state_bytes=head_state_bytes,
                )
            ]

        def t(object_id: str) -> TensorRef:
            return ctx.tensors[object_id]

        def tensor(
            object_id: str,
            shape: tuple[int, int],
            role: Literal["activation", "gradient", "optimizer_state", "parameter"],
            dtype: str,
            size_bytes: int | None = None,
        ) -> TensorRef:
            return ctx.tensor(
                object_id,
                shape,
                role=role,
                dtype=dtype,
                size_bytes=size_bytes,
            )

        def input_id(k: int, j: int) -> str:
            return f"input_{k}_{j}"

        def round_id(base: str, k: int, j: int) -> str:
            return f"{base}_{k}_{j}"

        def step_grad_id(k: int, i: int) -> str:
            return f"dW_{k}_{i}"

        def step_head_grad_id(k: int) -> str:
            return f"dW_embed_{k}" if tied else f"dW_head_{k}"

        def step_embed_grad_id(k: int) -> str:
            return f"dW_embed_{k}"

        # Optimizer placement (mirrors the runtime's M4.9 semantics):
        # "interleaved" (default) emits each optimizer task right after the
        # LAST mutation of its gradient — layer i's after its final-round
        # backward, the head's after the final round's head_loss — so
        # optimizer-state streaming overlaps the remaining backward.
        # "tail" is the legacy all-optimizers-after-all-rounds order (a
        # measured 7-8% step tax at seq-1K/8B on real hardware). Task ids
        # are identical in both modes; only emission order changes.
        interleaved = training.optimizer_placement == "interleaved"
        last_round = training.grad_accum_rounds - 1

        def emit_layer_optimizer(k: int, i: int, layer: TrainingLayerSpec) -> None:
            _, _, _, optimizer_ops = layer_phase_ops[i]
            inputs = [t(step_grad_id(k, i)), t(f"W_{i}")]
            mutates = [t(f"W_{i}")]
            if self._optimizer_state_bytes(
                layer,
                training.optimizer,
                policy,
                parallel,
            ) > 0:
                inputs.append(t(f"O_{i}"))
                mutates.append(t(f"O_{i}"))
            ctx.emit_task(
                id=f"step_{k}_{i}",
                label=f"Step {k} Layer {i} Optimizer",
                group="optimizer",
                block_key=f"{layer.optimizer_block_key}.{training.optimizer}",
                block_name=(
                    f"{optimizer_display_name(training.optimizer)} "
                    f"Optimizer Step: {layer.block_name}"
                ),
                subops=optimizer_ops,
                inputs=inputs,
                mutates=mutates,
                block_metadata=self._phase_metadata(
                    "optimizer",
                    {"optimizer": training.optimizer, **layer.metadata},
                ),
            )

        def emit_head_optimizer(k: int) -> None:
            inputs = [t(step_head_grad_id(k)), t("W_head")]
            mutates = [t("W_head")]
            if head_state_bytes > 0:
                inputs.append(t("O_head"))
                mutates.append(t("O_head"))
            ctx.emit_task(
                id=f"step_{k}_head",
                label=f"Step {k} Head Optimizer",
                group="optimizer",
                block_key=f"{self.head.optimizer_block_key}.{head_optimizer_mode}",
                block_name=(
                    f"{optimizer_display_name(head_optimizer_mode)} "
                    f"Optimizer Step: {self.head.block_name}"
                ),
                subops=head_optimizer_ops,
                inputs=inputs,
                mutates=mutates,
                block_metadata=self._phase_metadata(
                    "optimizer",
                    {
                        "optimizer": head_optimizer_mode,
                        "requested_optimizer": training.optimizer,
                        **self.head.metadata,
                    },
                ),
            )

        def emit_embed_optimizer(k: int) -> None:
            inputs = [t(step_embed_grad_id(k)), t("W_embed")]
            mutates = [t("W_embed")]
            if embed_state_bytes > 0:
                inputs.append(t("O_embed"))
                mutates.append(t("O_embed"))
            ctx.emit_task(
                id=f"step_{k}_embed",
                label=f"Step {k} Embedding Optimizer",
                group="optimizer",
                block_key=f"{embedding.optimizer_block_key}.{training.optimizer}",
                block_name=(
                    f"{optimizer_display_name(training.optimizer)} "
                    f"Optimizer Step: {embedding.block_name}"
                ),
                subops=embed_optimizer_ops,
                inputs=inputs,
                mutates=mutates,
                block_metadata=self._phase_metadata(
                    "optimizer",
                    {"optimizer": training.optimizer, **embedding.metadata},
                ),
            )

        for k in range(training.num_steps):
            for j in range(training.grad_accum_rounds):
                if embedding is not None:
                    ctx.emit_task(
                        id=round_id("embed_fwd", k, j),
                        label=f"Step {k} Round {j} Embedding Forward",
                        group="forward",
                        block_key=f"{embedding.block_key}.forward",
                        block_name=f"{embedding.block_name} Forward",
                        subops=embed_forward_ops,
                        inputs=[t(f"tokens_{k}_{j}"), t("W_embed")],
                        outputs=[
                            tensor(
                                round_id("y_embed", k, j),
                                (tokens, embedding.output_dim),
                                "activation",
                                policy.activation,
                            )
                        ],
                        block_metadata=self._phase_metadata(
                            "embed_forward", embedding.metadata,
                        ),
                    )
                entry_act = (
                    round_id("y_embed", k, j)
                    if embedding is not None
                    else input_id(k, j)
                )
                # Forward follows the exact model-authored layer order.
                for i, layer in enumerate(self.layers):
                    fwd_ops, _, _, _ = layer_phase_ops[i]
                    in_act = (
                        entry_act
                        if i == 0
                        else layer_round_id("y", k, j, i - 1)
                    )
                    outputs = [
                        tensor(
                            layer_round_id("y", k, j, i),
                            (tokens, layer.output_dim),
                            "activation",
                            policy.activation,
                        )
                    ]
                    saved_id = layer_round_id("A", k, j, i)
                    if saved_id not in recomputed:
                        outputs.insert(
                            0,
                            tensor(
                                saved_id,
                                (tokens, layer.saved_activation_width),
                                "activation",
                                policy.activation,
                                size_bytes=self._saved_activation_bytes(
                                    layer,
                                    tokens,
                                    seqlen,
                                    policy,
                                ),
                            ),
                        )
                    ctx.emit_task(
                        id=layer_round_id("f", k, j, i),
                        label=f"Step {k} Round {j} Layer {i} Forward",
                        group="forward",
                        block_key=f"{layer.block_key}.forward",
                        block_name=f"{layer.block_name} Forward",
                        subops=fwd_ops,
                        inputs=[t(in_act), t(f"W_{i}")],
                        outputs=outputs,
                        block_metadata=self._phase_metadata("forward", layer.metadata),
                    )

                # ---- fused head + loss + head backward (ONE task, the
                # runtime's true shape): the (tokens, vocab) logits/dlogits
                # never exist as objects — the executable chunks over tokens
                # internally. dW is created in round 0 and accumulated after;
                # tied families read/write the shared W_embed/dW_embed
                # (head_loss runs before embed_bwd, so IT creates the shared
                # gradient and embed_bwd accumulates). ----
                head_w = "W_embed" if tied else "W_head"
                head_grad = step_head_grad_id(k)
                head_loss_inputs = [
                    t(layer_round_id("y", k, j, self.n_layers - 1)),
                    t(f"targets_{k}_{j}"),
                    t(head_w),
                ]
                head_loss_outputs = [
                    tensor(
                        layer_round_id("dy", k, j, self.n_layers - 1),
                        (tokens, self.head.input_dim),
                        "activation",
                        policy.activation,
                    ),
                    tensor(
                        round_id("loss", k, j),
                        (1, 1),
                        "activation",
                        "fp32",
                        size_bytes=4,
                    ),
                ]
                head_loss_mutates: list[TensorRef] = []
                if j == 0:
                    head_loss_outputs.append(
                        tensor(
                            head_grad,
                            (self.head.param_count, 1),
                            "gradient",
                            policy.gradient,
                            size_bytes=self._gradient_bytes(
                                self.head,
                                policy,
                                parallel,
                            ),
                        )
                    )
                else:
                    head_loss_inputs.append(t(head_grad))
                    head_loss_mutates.append(t(head_grad))
                ctx.emit_task(
                    id=round_id("head_loss", k, j),
                    label=f"Step {k} Round {j} LM Head + Loss (fused)",
                    group="head",
                    block_key=f"{self.head.block_key}.head_loss",
                    block_name=f"{self.head.block_name} + Loss (fused)",
                    subops=[*head_forward_ops, *head_backward_ops],
                    inputs=head_loss_inputs,
                    outputs=head_loss_outputs,
                    mutates=head_loss_mutates,
                    block_metadata=self._phase_metadata(
                        "head_loss",
                        self.head.metadata,
                    ),
                )
                if interleaved and head_optimizer_enabled and j == last_round:
                    # (untied) head grad is final after the last head_loss
                    emit_head_optimizer(k)

                # Backward is the reverse of the model-authored layer order.
                for i in range(self.n_layers - 1, -1, -1):
                    layer = self.layers[i]
                    _, bwd_ops, recompute_ops, _ = layer_phase_ops[i]
                    upstream = layer_round_id("dy", k, j, i)
                    r_in_act = (
                        entry_act
                        if i == 0
                        else layer_round_id("y", k, j, i - 1)
                    )
                    saved_id = layer_round_id("A", k, j, i)
                    if saved_id in recomputed:
                        ctx.emit_task(
                            id=layer_round_id("r", k, j, i),
                            label=f"Step {k} Round {j} Layer {i} Recompute",
                            group="recompute",
                            block_key=f"{layer.block_key}.recompute",
                            block_name=f"{layer.block_name} Recompute",
                            subops=recompute_ops,
                            inputs=[t(r_in_act), t(f"W_{i}")],
                            outputs=[
                                tensor(
                                    saved_id,
                                    (tokens, layer.saved_activation_width),
                                    "activation",
                                    policy.activation,
                                    size_bytes=self._saved_activation_bytes(
                                        layer,
                                        tokens,
                                        seqlen,
                                        policy,
                                    ),
                                )
                            ],
                            block_metadata=self._phase_metadata(
                                "recompute",
                                layer.metadata,
                            ),
                        )

                    grad_id = step_grad_id(k, i)
                    # the backward reads the block's forward input too (the
                    # runtime's norm-bwd contract): x = y_{i-1} / y_embed
                    b_inputs = [t(upstream), t(saved_id), t(r_in_act), t(f"W_{i}")]
                    b_outputs = [
                        tensor(
                            round_id("dy_embed", k, j)
                            if i == 0
                            else layer_round_id("dy", k, j, i - 1),
                            (tokens, layer.input_dim),
                            "activation",
                            policy.activation,
                        )
                    ]
                    b_mutates: list[TensorRef] = []
                    if j == 0:
                        b_outputs.append(
                            tensor(
                                grad_id,
                                (layer.gradient_param_count, 1),
                                "gradient",
                                policy.gradient,
                                size_bytes=self._gradient_bytes(
                                    layer,
                                    policy,
                                    parallel,
                                ),
                            )
                        )
                    else:
                        b_inputs.append(t(grad_id))
                        b_mutates.append(t(grad_id))
                    ctx.emit_task(
                        id=layer_round_id("b", k, j, i),
                        label=f"Step {k} Round {j} Layer {i} Backward",
                        group="backward",
                        block_key=f"{layer.block_key}.backward",
                        block_name=f"{layer.block_name} Backward",
                        subops=bwd_ops,
                        inputs=b_inputs,
                        outputs=b_outputs,
                        mutates=b_mutates,
                        block_metadata=self._phase_metadata("backward", layer.metadata),
                    )
                    if (
                        interleaved
                        and training.optimizer != "none"
                        and j == last_round
                    ):
                        # dW_i is final after layer i's last-round backward
                        emit_layer_optimizer(k, i, layer)

                if embedding is not None:
                    # scatter-accumulate is the LAST backward task of every
                    # round. Untied round 0 creates dW_embed; tied rounds
                    # always accumulate (head_loss created the shared grad).
                    embed_grad = step_embed_grad_id(k)
                    embed_bwd_inputs = [
                        t(round_id("dy_embed", k, j)),
                        t(f"tokens_{k}_{j}"),
                    ]
                    embed_bwd_outputs: list[TensorRef] = []
                    embed_bwd_mutates: list[TensorRef] = []
                    if j == 0 and not tied:
                        embed_bwd_outputs.append(
                            tensor(
                                embed_grad,
                                (embedding.param_count, 1),
                                "gradient",
                                policy.gradient,
                                size_bytes=math.ceil(
                                    embedding.param_count
                                    * dtype_nbytes(policy.gradient)
                                ),
                            )
                        )
                    else:
                        embed_bwd_inputs.append(t(embed_grad))
                        embed_bwd_mutates.append(t(embed_grad))
                    ctx.emit_task(
                        id=round_id("embed_bwd", k, j),
                        label=f"Step {k} Round {j} Embedding Backward",
                        group="backward",
                        block_key=f"{embedding.block_key}.backward",
                        block_name=f"{embedding.block_name} Backward",
                        subops=embed_backward_ops,
                        inputs=embed_bwd_inputs,
                        outputs=embed_bwd_outputs,
                        mutates=embed_bwd_mutates,
                        block_metadata=self._phase_metadata(
                            "embed_backward", embedding.metadata,
                        ),
                    )
                    if (
                        interleaved
                        and training.optimizer != "none"
                        and j == last_round
                    ):
                        # embed_bwd is the round's last task; the embedding
                        # optimizer closes the step (covers the tied head too)
                        emit_embed_optimizer(k)

            if training.optimizer != "none" and not interleaved:
                # legacy tail placement: all optimizers after all rounds
                # (embedding first, then layers, then head — the runtime's
                # tail order)
                if embedding is not None:
                    emit_embed_optimizer(k)
                for i, layer in enumerate(self.layers):
                    emit_layer_optimizer(k, i, layer)
                if head_optimizer_enabled:
                    emit_head_optimizer(k)

        # (loss objects stay in fast at chain end: pinning them to backing
        # would require every policy — not just pressurefit — to schedule a
        # 4-byte writeback; the runtime persists losses, demos don't need to)
        final_locations: dict[str, Literal["fast", "backing"]] = {}
        if training.optimizer != "none" and training.final_model_state_on_backing:
            for i, layer in enumerate(self.layers):
                final_locations[f"W_{i}"] = "backing"
                if self._optimizer_state_bytes(
                    layer,
                    training.optimizer,
                    policy,
                    parallel,
                ) > 0:
                    final_locations[f"O_{i}"] = "backing"
            if embedding is not None:
                final_locations["W_embed"] = "backing"
                if embed_state_bytes > 0:
                    final_locations["O_embed"] = "backing"
            if head_optimizer_enabled:
                final_locations["W_head"] = "backing"
                if head_state_bytes > 0:
                    final_locations["O_head"] = "backing"

        return ctx.program(
            metadata={
                "kind": self.metadata_kind,
                "family": self.family_name,
                "preset": self.preset_name,
                **self.model_metadata,
                "model": dict(self.model_metadata),
                "training": asdict(training),
                "dtype_policy": asdict(policy),
                "parallelism": asdict(parallel),
            },
            metrics=DataflowMetrics(
                primary_unit="tokens",
                primary_count=(
                    training.seqlen
                    * training.num_seqs
                    * training.grad_accum_rounds
                    * training.num_steps
                ),
                metadata={
                    "seqlen": training.seqlen,
                    "num_seqs": training.num_seqs,
                    "grad_accum_rounds": training.grad_accum_rounds,
                    "num_steps": training.num_steps,
                },
            ),
            final_locations=final_locations,
        )

    def build_training_workload(
        self,
        training: TrainingConfig,
        hw: HardwareSpec,
        *,
        input_shape: tuple[int, int] | None = None,
        name: str | None = None,
        dtype_policy: DTypePolicy | None = None,
        parallelism: ParallelismConfig | None = None,
        recompute: Mapping[str, int] | None = None,
    ) -> Workload:
        """Realize the program on hardware and attach planner-facing metadata."""

        program = self.build_training_program(
            training,
            input_shape=input_shape,
            name=name,
            dtype_policy=dtype_policy,
            parallelism=parallelism,
            recompute=recompute,
        )
        workload = realize_dataflow_program(program, hw)

        policy = dtype_policy or DTypePolicy()
        parallel = parallelism or ParallelismConfig()
        op_policy = OpDTypePolicy.from_dtype_policy(policy, parallel)

        rewrites: list[RecomputeRewrite] = []
        levels = dict(recompute or {})
        for k in range(training.num_steps):
            for j in range(training.grad_accum_rounds):
                for i, layer in enumerate(self.layers):
                    obj_id = layer_round_id("A", k, j, i)
                    recompute_terms = layer.recompute_ops(
                        training.tokens,
                        training.seqlen,
                        op_policy,
                    )
                    recompute_us = 0.0
                    if recompute_terms:
                        recompute_us = resolve_cost(
                            DataflowCost(
                                kind="sum",
                                name=f"{layer.block_key}.recompute",
                                terms=recompute_terms,
                            ),
                            hw,
                            default_name=f"{layer.block_key}.recompute",
                        ).runtime_us
                    options = (
                        RecomputeOption(
                            level=0,
                            saved_bytes=self._saved_activation_bytes(
                                layer,
                                training.tokens,
                                training.seqlen,
                                policy,
                            ),
                            recompute_us=0,
                            label="save-full",
                        ),
                        RecomputeOption(
                            level=1,
                            saved_bytes=0,
                            recompute_us=recompute_us,
                            label="recompute-full",
                        ),
                    )
                    rewrites.append(
                        RecomputeRewrite(
                            object_id=obj_id,
                            f_task_id=f"f_{k}_{j}_{i}",
                            r_task_id=f"r_{k}_{j}_{i}",
                            options=options,
                            f_compute_block_key=f"{layer.block_key}.forward",
                            r_compute_block_key=f"{layer.block_key}.recompute",
                            group_key=layer.block_key,
                        )
                    )
                    level = levels.pop(obj_id, 0)
                    if level not in (0, 1):
                        raise ValueError(
                            f"unsupported recompute level {level} for {obj_id!r}"
                        )
        if levels:
            raise ValueError(f"unknown recompute object ids: {sorted(levels)}")

        generic_breakdown = workload.metadata["breakdown"]
        return Workload(
            chain=workload.chain,
            metadata={
                **workload.metadata,
                "breakdown": {
                    "compute_blocks": generic_breakdown.get("compute_blocks", []),
                    **{
                        key: generic_breakdown.get(key)
                        for key in ("fwd", "bwd", "head", "optimizer", "totals_us")
                        if key in generic_breakdown
                    },
                },
                "compute_blocks": workload.metadata["compute_blocks"],
                "recompute_rewrites": rewrites,
                "summary": {
                    "kind": self.metadata_kind,
                    "family": self.family_name,
                    "n_layers": self.n_layers,
                    "total_tokens": (
                        training.seqlen
                        * training.num_seqs
                        * training.grad_accum_rounds
                        * training.num_steps
                    ),
                    "grad_accum_rounds": training.grad_accum_rounds,
                    "num_steps": training.num_steps,
                    "metrics": workload.metadata.get("metrics"),
                },
            },
        )

    def _phase_metadata(
        self,
        phase: str,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "family": self.family_name,
            "phase": phase,
            **self.model_metadata,
            **dict(extra or {}),
        }


def layer_round_id(prefix: str, k: int, j: int, i: int) -> str:
    return f"{prefix}_{k}_{j}_{i}"
