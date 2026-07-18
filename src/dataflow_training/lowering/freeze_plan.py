"""FreezePlan: the derived, program-shaping view of frozen parameters.

THE SPEC vs THE PLAN. Freezing is SPECIFIED exactly one way — the
optimizer policy (``dims.opt_policy``): a field whose resolved rule is
``"frozen"`` gets no gradient storage, no optimizer state, and no
update. That single oracle already drives dW/O layout shrinkage and
optimizer-task pruning. This module DERIVES the remaining, structural
consequences — which backward tasks exist, how far the dy chain
reaches, which layers save context at all, where the loss comes from —
into a ``FreezePlan``: the one optional input
``build_shaped_program(freeze=...)`` accepts, consumed by the surgery
in ``freeze_program.py``. The common builder stays branch-free.

Terminology (per-layer regimes; see the handling_frozen design note):

- TRAIN   — at least one trainable field, dy flows normally.
- PARTIAL — same structurally as TRAIN (backward emitted, dy flows);
            the per-field wgrad skips ride the shrunken dW layout
            (a wgrad computes iff its field is in dW — guards-first).
- FROZEN / pass-through — no trainable fields, but something BELOW
            still trains: the backward is emitted for its dgrads only
            (dW/O objects don't exist; the launch tolerates dw=None).
- FROZEN / truncated — no trainable fields AND nothing below trains:
            NO backward task, NO dy, and NO saved context (A is
            dropped and its recompute rewrite with it).

The dy rule is a SUFFIX property: layer i's backward produces
``dy_{i-1}`` iff something at depth < i trains (the embedding counts);
it receives ``dy_i`` iff it produces, or itself trains. A LOCAL
objective (``objective="indexer_kl"`` — the dense warm-up) replaces
CE entirely: no head, no targets, no dy anywhere, and the loss object
becomes the objective accumulator threaded through contributor
backwards.

Examples (the composer lives in ``tasks/optim.py``):

    from dataflow_training.blocks.optim import freeze

    # freeze the bottom half of the stack + the embedding (classic
    # continued-pretraining shape): backwards exist only for the top
    # half; layers 0..k-1 save no context at all
    cfg = replace(cfg, opt_policy=freeze(layers=range(0, 16),
                                         embed=True))

    # freeze one field everywhere (dW/O shrink fleet-wide; every
    # backward still runs, skipping that wgrad)
    cfg = replace(cfg, opt_policy=freeze(fields=("wq",)))

    # targeted (layer, field) pairs
    cfg = replace(cfg, opt_policy=freeze(pairs=(("wo", 3), ("w1", 7))))

    # compose over a non-default base (muon recipe on what trains)
    cfg = replace(cfg, opt_policy=freeze(base="muon",
                                         layers=range(0, 8)))

Derivation is a pure function; a policy that freezes nothing (under a
CE objective) derives to ``None`` — the fast path that keeps every
existing program byte-identical.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from dataflow_training.blocks.optim import resolve_opt_policy

OBJECTIVES = ("ce", "indexer_kl")


@dataclass(frozen=True)
class FreezePlan:
    """Structural freeze consequences for one program build.

    ``regimes[i]`` is one of ``"train"`` / ``"partial"`` /
    ``"passthrough"`` / ``"truncated"`` (see module docstring).
    ``emit_bwd[i]`` / ``recv_dy[i]`` / ``produce_dy[i]`` spell the
    backward-task and dy-chain shape the surgery realizes; ``save_ctx``
    is False exactly where no backward will ever read A (the layer's A
    output and its recompute rewrite are dropped). ``loss_contributors``
    is non-empty only for local objectives: the layers whose backwards
    create/accumulate ``loss_{s}_{r}`` (for CE the head owns the loss,
    as always).
    """

    n_layers: int
    regimes: tuple[str, ...]
    emit_bwd: tuple[bool, ...]
    recv_dy: tuple[bool, ...]
    produce_dy: tuple[bool, ...]
    save_ctx: tuple[bool, ...]
    embed_trainable: bool
    head_trainable: bool
    objective: str = "ce"
    loss_contributors: tuple[int, ...] = ()

    def __post_init__(self):
        n = self.n_layers
        for name in ("regimes", "emit_bwd", "recv_dy", "produce_dy",
                     "save_ctx"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"FreezePlan.{name}: expected {n} entries")
        if self.objective not in OBJECTIVES:
            raise ValueError(f"unknown objective {self.objective!r} "
                             f"(one of {OBJECTIVES})")
        bad = [r for r in self.regimes
               if r not in ("train", "partial", "passthrough", "truncated")]
        if bad:
            raise ValueError(f"unknown regimes {sorted(set(bad))}")
        if self.objective == "ce" and not (
                self.embed_trainable or self.head_trainable
                or any(r in ("train", "partial") for r in self.regimes)):
            raise ValueError(
                "everything is frozen under a CE objective — nothing "
                "would train; freeze less or use a local objective")
        for i in self.loss_contributors:
            if not 0 <= i < n:
                raise ValueError(f"loss contributor {i} out of range")

    def __repr__(self) -> str:  # compact, log-friendly
        def spans(pred):
            out, start = [], None
            for i in range(self.n_layers + 1):
                hit = i < self.n_layers and pred(i)
                if hit and start is None:
                    start = i
                if not hit and start is not None:
                    out.append(f"{start}" if i - 1 == start
                               else f"{start}-{i - 1}")
                    start = None
            return ",".join(out) or "-"

        return (f"FreezePlan(obj={self.objective}, "
                f"train={spans(lambda i: self.regimes[i] in ('train', 'partial'))}, "
                f"passthrough={spans(lambda i: self.regimes[i] == 'passthrough')}, "
                f"truncated={spans(lambda i: self.regimes[i] == 'truncated')}, "
                f"embed={'T' if self.embed_trainable else 'F'}, "
                f"head={'T' if self.head_trainable else 'F'})")


def derive_freeze_plan(
    dims,
    n_layers: int,
    fields_of: Callable[[int], Sequence[str]],
    *,
    embed_fields: Sequence[str] = ("w",),
    head_fields: Sequence[str] = ("w", "final_norm_w"),
    objective: str = "ce",
    tied_embeddings: bool = False,
    loss_contributors: Sequence[int] | None = None,
) -> FreezePlan | None:
    """Derive the FreezePlan from the optimizer policy — or ``None``
    when nothing structural changes (no frozen field anywhere under a
    CE objective): the fast path that leaves existing programs
    byte-identical.

    ``fields_of(layer)`` names the layer's weight fields (the family's
    weight layout); embed/head fields resolve through their ns-prefixed
    policy keys exactly as the optimizer executes them. For local
    objectives, ``loss_contributors`` defaults to every layer (families
    with grouped metadata pass producers/non-followers explicitly —
    glm52 passes its leaders)."""
    pol = resolve_opt_policy(getattr(dims, "opt_policy", None))

    def frozen(key: str, layer, shape=None) -> bool:
        return pol.for_field(key, layer, shape) == "frozen"

    trainable = []          # per layer: any trainable field?
    fully_frozen = []
    for i in range(n_layers):
        names = list(fields_of(i))
        rules = [not frozen(nm, i) for nm in names]
        trainable.append(any(rules))
        fully_frozen.append(not any(rules))
    embed_trainable = any(not frozen(f"embed.{nm}", None)
                          for nm in embed_fields)
    head_trainable = (embed_trainable if tied_embeddings else
                      any(not frozen(f"head.{nm}", None)
                          for nm in head_fields))

    if objective == "ce" and not any(fully_frozen):
        # No structural change: partial layers need no surgery (their
        # backwards run with shrunken dW; frozen embed/head reduce to
        # dW/O objects the zero-byte scrub already prunes). None keeps
        # every such program byte-identical to an unfrozen build.
        return None

    if objective == "indexer_kl":
        # local objective: no CE, no dy anywhere; backwards exist where
        # there is work (a trainable field or a loss/dM contribution)
        contributors = tuple(loss_contributors
                             if loss_contributors is not None
                             else range(n_layers))
        regimes, emit, recv, prod, save = [], [], [], [], []
        for i in range(n_layers):
            # every layer's backward works in warm-up: leaders run the
            # KL, followers deposit their rows into dM
            regimes.append("partial" if trainable[i] else "passthrough")
            emit.append(True)
            recv.append(False)
            prod.append(False)
            save.append(True)              # KL reads ctx (P2 trims fields)
        return FreezePlan(
            n_layers=n_layers, regimes=tuple(regimes),
            emit_bwd=tuple(emit), recv_dy=tuple(recv),
            produce_dy=tuple(prod), save_ctx=tuple(save),
            embed_trainable=False, head_trainable=False,
            objective="indexer_kl", loss_contributors=contributors,
        )

    # CE objective: suffix rule. produce_dy[i] == something below trains.
    below = embed_trainable                # embedding sits below layer 0
    produce_dy, recv_dy, emit, regimes, save = [], [], [], [], []
    for i in range(n_layers):
        produce_dy.append(below)
        below = below or trainable[i]
    for i in range(n_layers):
        recv = trainable[i] or produce_dy[i]
        recv_dy.append(recv)
        emit.append(recv)
        if trainable[i]:
            all_train = all(not frozen(nm, i) for nm in fields_of(i))
            regimes.append("train" if all_train else "partial")
        else:
            regimes.append("passthrough" if recv else "truncated")
        save.append(recv)                  # no backward -> A never read
    return FreezePlan(
        n_layers=n_layers, regimes=tuple(regimes), emit_bwd=tuple(emit),
        recv_dy=tuple(recv_dy), produce_dy=tuple(produce_dy),
        save_ctx=tuple(save), embed_trainable=embed_trainable,
        head_trainable=head_trainable, objective="ce",
    )
