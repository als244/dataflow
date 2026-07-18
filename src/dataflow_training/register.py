"""Workload registration: the training package's ONE hookup into the
engine service — ``register_all()`` registers the "model_family"
resolver kind. The daemon default-loads this module
(``--no-default-workloads`` opts out; ``--plugin`` modules register
additional kinds the same way).

resolver_spec wire form:

    {"kind": "model_family", "family": "<name>", "cfg": {...},
     "hyper": {...}?}

The build returns a Resolver whose dispatch covers every task the
family's programs emit PLUS the shared "family_init" task
(init-as-program: one task whose outputs are the training program's
initial W_/O_/Aux_/data objects, filled by the family's seeded init —
byte-identical to the in-process ``initial_values`` path by
construction, because it IS that path writing into the task's output
buffers).
"""
from __future__ import annotations

import json


def build_hyper(h: dict | None):
    """Wire ``hyper`` dict -> AdamWHyper (+ optional LRSchedule). Lets
    a client set lr / weight decay / a cosine schedule through the
    resolver channel; ``None`` -> the family default."""
    from dataflow_training.blocks.base_blocks import AdamWHyper

    h = dict(h or {})
    sched = h.pop("schedule", None)
    if sched is not None:
        from dataflow_training.blocks.optim import LRSchedule

        h["schedule"] = LRSchedule(**sched)
    return AdamWHyper(**h)


class FamilyInitExecutable:
    """The "family_init" task: fills its output buffers (the training
    program's initial objects, declared as BACKING outputs) with the
    family's seeded init — the same ``initial_values`` code path, so
    the bytes are identical to in-process init by construction.
    Host-side pinned writes only; nothing is enqueued on the stream."""

    def __init__(self, fam, cfg, seed: int, tp_view=None):
        self.fam = fam
        self.cfg = cfg
        self.seed = seed
        self.tp_view = tp_view

    def launch(self, ctx) -> None:
        program = self.fam.lower(self.cfg)
        into = {oid: buf for oid, buf in ctx.outputs.items()}
        # outputs may be a SUBSET of the training program's initial
        # objects (object_sizes-shrunken optimizer state still fills —
        # its init is a bulk zero bounded by the buffer size)
        import dataclasses as dc

        keep = set(into)
        specs = []
        for spec in program.initial_objects:
            if spec.id not in keep:
                continue
            buf = into[spec.id]
            if buf.size_bytes != spec.size_bytes:
                specs.append(dc.replace(spec, size_bytes=buf.size_bytes))
            else:
                specs.append(spec)
        shrunk = dc.replace(program, initial_objects=tuple(specs))
        if self.tp_view is not None:
            self.fam.initial_values(shrunk, self.cfg, None, seed=self.seed,
                                    into=into, tp_view=self.tp_view)
        else:
            self.fam.initial_values(shrunk, self.cfg, None, seed=self.seed,
                                    into=into)


class ModelFamilyResolver:
    """Resolver for one (family, cfg, hyper): dispatches every training
    task through the family's own resolver and the "family_init" task
    through FamilyInitExecutable."""

    def __init__(self, fam, cfg, dims, inner):
        self.fam = fam
        self.cfg = cfg
        self.dims = dims
        self.inner = inner

    def __call__(self, task):
        if task.compute_block_key == "family_init":
            seed = int(task.block_params.get("seed", 0))
            return FamilyInitExecutable(self.fam, self.cfg, seed,
                                        task.block_params.get("tp_view"))
        return self.inner(task)


def build_model_family_resolver(spec: dict):
    from dataflow_training.model_families.families import family

    fam = family(spec["family"])
    cfg = fam.config_type(**spec["cfg"])
    dims = fam.derive_dims(cfg)
    hyper = spec.get("hyper")
    inner = (fam.build_resolver(dims, build_hyper(hyper)) if hyper
             else fam.build_resolver(dims))
    return ModelFamilyResolver(fam, cfg, dims, inner)


def register_all() -> list[str]:
    """Register every workload resolver kind this package provides.
    Idempotent; returns the registered kind names."""
    from dataflow.service.registry import register_program_resolver

    register_program_resolver("model_family", build_model_family_resolver)
    return ["model_family"]


def canonical_spec(family: str, cfg_dict: dict, hyper: dict | None = None
                   ) -> dict:
    """The wire resolver_spec for a model-family program."""
    spec = {"kind": "model_family", "family": family, "cfg": cfg_dict}
    if hyper is not None:
        spec["hyper"] = json.loads(json.dumps(hyper))
    return spec
