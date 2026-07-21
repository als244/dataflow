"""Per-rank state and wire: the rank's client/config/rounds record,
threaded step runs, host rigs, and the round shipper (content bounds;
--execute-padding appends the masked tail; tp requires full rounds).
Split from the conductor at phase close."""
from __future__ import annotations

import threading
import time

from dataflow.service import EngineClient

class RankState:
    def __init__(self, name: str, client, cfg, rounds: tuple,
                 prog_id: str | None = None):
        self.name = name
        self.client = client
        self.cfg = cfg
        self.rounds = rounds           # ORIGINAL-stream round indices
        self.prog_id = prog_id
        self.losses: list = []
        self.error: str | None = None
        self.persist_ids: list = []    # W_*/O_* ids from the rank's
                                       # registered program (SavePlan)


class StepRun:
    def __init__(self, rank: RankState, step: int, valid: int,
                 seq_lens: dict | None = None):
        self.rank = rank
        self.step = step
        self.valid = valid
        self.seq_lens = seq_lens
        self.fetched: dict | None = None
        self.out: dict | None = None
        self.error = None

    def __call__(self):
        try:
            fetch = [f"loss_0_{r}"
                     for r in range(self.rank.cfg.grad_accum_rounds)]
            args = {"step": self.step, "valid_rows": self.valid}
            if self.seq_lens is not None:
                args["seq_lens"] = self.seq_lens
            out = self.rank.client.run(
                self.rank.prog_id, args=args, fetch=fetch)
            if out.get("state") != "done":
                raise RuntimeError(f"{self.rank.name}: {out}")
            self.out = out
            self.fetched = out["fetched"]
        except Exception as e:
            self.error = e


def put_rank_rounds(rank: RankState, packed, tokens_per_round: int, *,
                    execute_padding: bool = False,
                    require_full: bool = False) -> dict:
    """Ship the rank ITS slice of the step's packed rounds. Returns
    the rank's wire seq_lens bounds — CONTENT bounds by default (tasks
    compute only content rows); under ``execute_padding`` an under-full
    round's tail rides as one masked segment and executes.
    ``require_full``: raise on an under-full round unless padding
    executes — the TP lane's collectives run planner-sized buffers, so
    content-width rounds cannot cross them yet."""
    lens = {}
    for local_r, orig_r in enumerate(rank.rounds):
        rnd = packed.rounds[orig_r]
        if (require_full and not execute_padding
                and rnd.content < tokens_per_round):
            raise RuntimeError(
                f"round {orig_r} is under-full ({rnd.content}/"
                f"{tokens_per_round}) and this configuration requires "
                f"full rounds (tp_mlp collectives are planner-sized) — "
                f"pass --execute-padding or use exact-fill packing "
                f"(--allow-round-split)")
        bounds = rnd.bounds()
        if execute_padding and rnd.content < tokens_per_round:
            bounds = bounds + [tokens_per_round]
        lens[str(local_r)] = bounds
        rank.client.put_object(f"tokens_0_{local_r}",
                               rnd.tokens.tobytes())
        rank.client.put_object(f"targets_0_{local_r}",
                               rnd.targets.tobytes())
    return lens


def wait_client(sock: str, *, name: str, timeout_s: float,
                fail_hint: str) -> EngineClient:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            probe = EngineClient(sock, client_name="probe")
            probe.health()
            probe.close()
            return EngineClient(sock, client_name=name)
        except Exception:
            time.sleep(1.0)
    raise RuntimeError(fail_hint)


class HostRig:
    """One group member's runtime state under the conductor."""

    def __init__(self, host, slab_gib: float, budget_gib: float):
        self.host = host
        self.slab_gib = slab_gib
        self.budget_gib = budget_gib
        self.launched = False
        self.forward = None
        self.sock: str | None = None
        self.client: EngineClient | None = None
        self.prof_out: str | None = None


# Split at phase close: grouped lowering + checkpointing live in
# their own modules; these re-exports keep every existing import
# stable (tests, tools, drills).
from .checkpointing import (  # noqa: F401
    checkpoint_fleet,
    distribute_artifacts,
    resolve_resume,
)
from .grouped_lowering import (  # noqa: F401
    GroupedBuildVariant,
    lower_with_group,
)


