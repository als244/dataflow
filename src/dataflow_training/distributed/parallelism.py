"""The parallelism scheme: THE contract for how a run parallelizes.

A scheme is an ordered tuple of named AXES — the mesh form, so
parallelisms COMPOSE by construction rather than by special-casing
pairs. One axis = one aspect of the mesh; a composed scheme is just
more axes. The conductor consumes any valid scheme through one entry
point (``run(..., scheme=...)``); a new parallelism must never add a
conductor entry point.

MESH SEMANTICS (what a multi-axis scheme MEANS)
-----------------------------------------------
- ``axes`` is ordered OUTER -> INNER. The scheme's WORLD is the
  product of axis sizes, and a global rank's mesh coordinate is the
  row-major unflatten of its rank index over the axis sizes — the
  LAST axis varies fastest. Placement rule of thumb: put the most
  bandwidth-hungry axis last so its peers land on adjacent ranks
  (e.g. tp innermost on an NVLink box, dp outermost across boxes).
- COMM SUB-GROUPS derive per axis: for axis i, a sub-group is the
  set of ranks sharing every coordinate EXCEPT i (world/size_i
  groups of size_i each). The group-annotation pass attaches
  {axis.name: sub-group handle} onto the lowered program, so a task
  that says ``comm_groups={"dp": ...}`` gets ITS dp peers, not the
  whole world.
- REPLICATION vs OWNERSHIP: an axis either replicates parameters
  across its extent (data axis — every peer holds the same bytes) or
  partitions them (tensor axis — each peer holds a disjoint shard by
  construction). Responsibility (who steps + saves a slice) composes
  by refinement: partitioning axes fix WHICH bytes exist on a rank;
  the data axis's ``responsibility`` mode then splits each
  replicated slice among the replicas (zero1rs: at the optimizer's
  own flat boundaries). Replicas not responsible for a slice are its
  recorded BACKUPS (multiplicity in the save plan, fault tolerance
  later).
- ROUNDS: at most one axis (the data axis) divides the step's
  DATA, expressed in round units (a round = one forward/backward's
  worth of tokens); every other axis runs its data coordinate's
  full share. Gradient accumulation is rank-LOCAL — a rank's
  grad-accum count is simply its data share. Unequal shares =
  weighted data parallelism for heterogeneous GPUs.

HOW THE LAYERS STACK (the scheme is the top; each layer references
the artifacts of the one below, never its internals):

    ParallelismScheme      WHAT the parallel structure is (pure data)
      -> group annotation  comm handles + shard/tp params on the
         (annotation pass)  lowered program, keyed by axis purposes
      -> sharding API      the GEOMETRY under a tensor/optimizer
         (sharding.py)      axis: ShardPlan, per-rank views, flat
                            slice boundaries — computed once, carried
                            by the scheme as data
      -> layouts           the coordinate system geometry compiles
         (layout registry,  onto: named field tables; narrow_layouts
          narrow_layouts)   applies slice geometry; sizes follow
      -> responsibility    who steps/saves, derived from the same
         (responsibility.py) axes; becomes the checkpoint save plan

ADDING AN AXIS ROLE (the EP/PP recipe)
--------------------------------------
A new parallelism is a new axis role here plus its program-layer
machinery — and nothing else:

1. a comm PURPOSE in the vocabulary (``Axis.name``) that
   ``TaskSpec.comm_groups`` speaks and the annotation pass binds;
2. an annotation-pass extension (which tasks get the new handle,
   which kinds swap);
3. if the role is GEOMETRIC, a plan kind in the sharding API saying
   what lives where (carried on the axis, like ShardPlan on "tp");
4. if the role needs an aspect no current field expresses, a NEW
   ``Axis`` field — fields are orthogonal aspects and unused fields
   stay at their defaults, so adding one is the sanctioned path, not
   a smell.

Concretely: EXPERT PARALLELISM ("ep") is a partitioning axis whose
plan places experts on ranks; its comm purpose is the token
all-to-all; the dense trunk is REPLICATED along ep (so the data
axis's responsibility still governs trunk slices) while expert
params are owned by placement, like tp shards. Activation scratch
under capacity routing is the ep-phase TODO recorded in the plan
docs. PIPELINE PARALLELISM ("pp") is a partitioning axis over the
PROGRAM rather than over tensors: it needs a new field (a stage
partition of the block sequence), a "pp" p2p purpose for boundary
activations, and a schedule in run/loop.py — the conductor entry
does not change.

COMPOSITION CONTRACT
--------------------
Composed schemes are spelled by concatenating axes — the intended
future spelling is simply

    ParallelismScheme(axes=dp.axes + tp.axes)      # dp outer, tp inner

with per-axis sub-groups derived as above. ``validate`` REFUSES
multi-axis schemes today (the mesh sub-group machinery lands when a
composed configuration first matters) — loudly, rather than
half-running one axis of a mesh. Everything downstream is already
written against the per-axis views, so the refusal is the only line
that changes. Schemes are pure data — constructing one performs no
work, so any composition may be BUILT and inspected today, just not
run.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Axis:
    """One aspect of the mesh. Legal field combinations:

    - ``name`` — the axis's comm PURPOSE key ("dp", "tp"; future
      "ep", "pp"). Unique within a scheme; this exact string is the
      key ``TaskSpec.comm_groups`` uses and the annotation pass
      binds to a concrete group handle.
    - ``size`` — ranks along the axis (>= 1).
    - ``rounds`` — each rank's share of the step's DATA in round
      units, only on a DATA axis (``plan is None``): length == size,
      shares sum to the step's total (the config's
      ``grad_accum_rounds``); a rank's LOCAL grad-accum count is
      its share. Unequal shares = weighted data parallelism.
      Mutually exclusive with ``plan`` (a tensor axis runs the full
      batch).
    - ``responsibility`` — who steps + saves parameter slices along
      a REPLICATING axis (see distributed/responsibility.py):
      "zero1rs" (default: one responsible rank per slice, split at
      the optimizer's own flat boundaries, byte-equal rs/ag comm),
      "zero1" (field-snapped shards), or "co" (co-responsible
      diagnostic lane: every replica steps, bitwise cross-rank
      equality acts as a corruption tripwire). Meaningless on a
      partitioning axis — there each rank owns what it holds by
      construction.
    - ``plan`` — a sharding-API plan making this a PARTITIONING
      (tensor) axis: which parameter/activation shards live on which
      coordinate. The plan is geometry computed once and carried as
      data; group annotation and lowering consume it, never
      recompute it.

    A field left at its default is an aspect this axis does not use.
    """

    name: str
    size: int
    rounds: tuple | None = None      # data split along this axis
    responsibility: str = "zero1rs"  # who steps/saves along this axis
    plan: object | None = None       # tensor plan along this axis


@dataclass(frozen=True)
class ParallelismScheme:
    axes: tuple = ()

    @property
    def world(self) -> int:
        """Product of axis sizes; () -> 1 (the solo scheme)."""
        n = 1
        for ax in self.axes:
            n *= ax.size
        return n

    # ---- the single-axis views today's conductor consumes ----------
    @property
    def data_axis(self):
        """The (single, today) replicating axis: first with no plan."""
        for ax in self.axes:
            if ax.plan is None:
                return ax
        return None

    @property
    def tensor_axis(self):
        """The (single, today) partitioning axis: first with a plan."""
        for ax in self.axes:
            if ax.plan is not None:
                return ax
        return None

    @property
    def rank_rounds(self):
        """Data axis round split, or None (solo / pure tensor)."""
        ax = self.data_axis
        return ax.rounds if ax else None

    @property
    def responsibility(self):
        """Data axis responsibility mode, or None when the scheme has
        NO replicating axis (solo / pure tensor) — there is nothing to
        partition responsibility over, and callers must not conjure a
        mode where none applies."""
        ax = self.data_axis
        return ax.responsibility if ax else None

    @property
    def tensor_plan(self):
        """Tensor axis plan, or None."""
        ax = self.tensor_axis
        return ax.plan if ax else None

    def validate(self, *, world: int, ga_rounds: int) -> None:
        """Refuse inconsistent schemes loudly, before any daemon
        launches. Enforces: single-axis only (until mesh sub-group
        machinery lands), positive sizes, known responsibility modes,
        rounds xor plan per axis, scheme world == topology world, and
        the data-axis round split summing to the global
        grad_accum_rounds."""
        if len(self.axes) > 1:
            raise ValueError(
                "composed (multi-axis) schemes are contract-ready but "
                "the mesh group machinery is not built yet — run one "
                "axis, or build the composition when it first matters")
        for ax in self.axes:
            if ax.size < 1:
                raise ValueError(f"axis {ax.name!r}: size {ax.size}")
            if ax.responsibility not in ("zero1rs", "zero1", "co"):
                raise ValueError(
                    f"axis {ax.name!r}: unknown responsibility "
                    f"{ax.responsibility!r} (zero1rs | zero1 | co)")
            if ax.plan is not None and ax.rounds is not None:
                raise ValueError(
                    f"axis {ax.name!r}: a tensor axis runs the FULL "
                    f"batch — rounds does not apply")
        if self.axes and self.world != world:
            raise ValueError(
                f"scheme world {self.world} vs topology world {world}")
        if self.tensor_axis is not None:
            if world < 2:
                raise ValueError("tensor parallelism needs >= 2 ranks")
            return
        if world == 1:
            if self.rank_rounds not in (None, (ga_rounds,)):
                raise ValueError(
                    f"world 1 runs every round on the one rank; got "
                    f"rank_rounds={self.rank_rounds}")
            return
        rounds = self.rank_rounds
        if rounds is None:
            raise ValueError(
                f"data parallelism at world {world} needs a data axis "
                f"with a per-rank round split summing to {ga_rounds}")
        if len(rounds) != world:
            raise ValueError(f"rounds {rounds} vs world {world}")
        if sum(rounds) != ga_rounds:
            raise ValueError(
                f"rounds {rounds} must sum to the global "
                f"grad_accum_rounds {ga_rounds}")

    # ---- named constructors (the readable spellings) ----------------
    @classmethod
    def solo(cls) -> "ParallelismScheme":
        """World-1: the empty mesh (no axes), world == 1.

        The identity of composition — adding axes to solo yields
        exactly those axes. The conductor treats it as one rank
        running every round with whole-object checkpoints; no comm
        purposes are bound. ``scheme=None`` at the conductor means
        solo."""
        return cls()

    @classmethod
    def data_parallel(cls, rank_rounds,
                      responsibility: str = "zero1rs",
                      ) -> "ParallelismScheme":
        """One replicating "dp" axis: world == len(rank_rounds).

        Args:
            rank_rounds: each rank's share of the step's data, in
                round units; shares sum to the step's total (the
                config's ``grad_accum_rounds``), and a rank's LOCAL
                grad-accum count equals its share. Unequal shares =
                weighted DP for heterogeneous GPUs (e.g. (6, 2) for
                a 5090/3090 pair).
            responsibility: slice-stepping mode along the axis —
                "zero1rs" (default; partitioned optimizer + byte-equal
                reduce-scatter/all-gather), "zero1" (field-snapped),
                or "co" (replicated stepping, the diagnostic lane).

        Every rank holds full parameters; gradients meet on the "dp"
        purpose; the responsibility mode decides who steps + saves
        which slice (the checkpoint save plan follows it). Composes
        (future) with a tensor axis as dp outer, tp inner:
        ``ParallelismScheme(axes=dp.axes + tp.axes)`` — then each dp
        sub-group all-reduces only its tp-coordinate's shard grads,
        and responsibility refines WITHIN each replica set."""
        rounds = tuple(rank_rounds)
        return cls(axes=(Axis("dp", size=len(rounds), rounds=rounds,
                              responsibility=responsibility),))

    @classmethod
    def tensor_parallel(cls, plan, *, size: int | None = None,
                        ) -> "ParallelismScheme":
        """One partitioning "tp" axis: world == plan's world.

        Args:
            plan: a sharding-API plan (e.g. ``tp_mlp_shards``) fixing
                which parameter/activation shards live on which
                coordinate. Geometry is computed once here and
                carried as data.
            size: rank count; defaults to ``plan.world``.

        Every rank runs the FULL batch (rounds does not apply —
        tensor parallelism splits compute, not data), and owns the
        slices it holds by construction, so ``responsibility`` is
        not consulted: the save plan is the placement itself. Partial
        activations meet on the "tp" purpose. Composes (future) with
        a data axis as the inner axis of a dp x tp mesh — keep tp
        innermost so its high-bandwidth collectives land on adjacent
        ranks."""
        n = size if size is not None else getattr(plan, "world", 2)
        return cls(axes=(Axis("tp", size=n, plan=plan),))
