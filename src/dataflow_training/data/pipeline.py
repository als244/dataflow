"""DataPipeline: the one constructor the drivers and tools share.

Binds a --data SPEC + packing flags to a run's geometry; calling it
(with an optional resume cursor) builds the DataFeed + Packer stack.
The drivers take this FACTORY rather than a live packer so resume can
construct the pipeline at the checkpointed cursor.
"""
from __future__ import annotations

from dataflow_training.data.packer import Packer
from dataflow_training.data.sources import parse_spec, resolve_data

DEFAULT_SPEC = "shards:"


class DataPipeline:
    """Callable factory: pipeline(cursor|None) -> Packer."""

    def __init__(self, spec: str, *, tokens_per_round: int, ga_rounds: int,
                 max_seqlen: int, vocab_size: int,
                 policy: str = "ffd", allow_round_split: bool = False,
                 lookahead_mult: int = 4, capture=None):
        self.spec = spec or DEFAULT_SPEC
        self.tokens_per_round = int(tokens_per_round)
        self.ga_rounds = int(ga_rounds)
        self.max_seqlen = int(max_seqlen)
        self.vocab_size = int(vocab_size)
        self.policy = policy
        self.allow_round_split = bool(allow_round_split)
        self.lookahead_mult = int(lookahead_mult)
        self.capture = capture
        # constructed eagerly so bad specs fail at setup, not step 0
        self.source = resolve_data(self.spec, max_seqlen=self.max_seqlen,
                                   vocab_size=self.vocab_size)

    def __call__(self, cursor: dict | None = None) -> Packer:
        from dataflow_training.data.feed import DataFeed

        feed = DataFeed(self.source, start_cursor=cursor,
                        capture=self.capture)
        return Packer(feed, tokens_per_round=self.tokens_per_round,
                      ga_rounds=self.ga_rounds, max_seqlen=self.max_seqlen,
                      allow_round_split=self.allow_round_split,
                      policy=self.policy,
                      lookahead_mult=self.lookahead_mult)

    def describe(self) -> dict:
        out = dict(self.source.describe())
        out.update({"spec": self.spec, "policy": self.policy,
                    "allow_round_split": self.allow_round_split,
                    "lookahead_mult": self.lookahead_mult,
                    "tokens_per_round": self.tokens_per_round,
                    "ga_rounds": self.ga_rounds})
        return out


class PrepackedPipeline:
    """Factory over already-packed steps (tests, replays): ignores the
    packing flags, seeks by the PrepackedFeed step cursor."""

    def __init__(self, steps: list):
        self.steps = steps

    def __call__(self, cursor: dict | None = None):
        from dataflow_training.data.feed import PrepackedFeed

        start = int(cursor["step"]) if cursor else 0
        return PrepackedFeed(self.steps, start=start)

    def describe(self) -> dict:
        return {"scheme": "prepacked", "steps": len(self.steps),
                "deterministic": True}


def fast_forward(stepper, n_steps: int) -> None:
    """Advance a packer/stepper by consuming steps CPU-side — the
    resume fallback when a checkpoint predates data cursors (mmap
    reads only; no GPU, no engine)."""
    for _ in range(n_steps):
        stepper.next_step()


def pipeline_from_args(cfg, data_spec: str | None, *,
                       policy: str = "ffd",
                       allow_round_split: bool = False,
                       lookahead_mult: int = 4,
                       capture=None) -> DataPipeline:
    """The tools' one-liner: run geometry from cfg + flags."""
    if data_spec in ("block", "doc"):
        raise ValueError(
            f"--data {data_spec!r} retired: use a source spec — "
            f"'shards:' (per-document) for the old doc mode, "
            f"'shards:,window={cfg.seq_len}' for the old block mode; "
            f"legacy curves additionally set --allow-round-split and "
            f"long_policy=whole")
    return DataPipeline(
        data_spec or DEFAULT_SPEC,
        tokens_per_round=cfg.max_tokens, ga_rounds=cfg.grad_accum_rounds,
        max_seqlen=cfg.seq_len, vocab_size=cfg.vocab_size,
        policy=policy, allow_round_split=allow_round_split,
        lookahead_mult=lookahead_mult, capture=capture)


def legacy_block_pipeline(cfg, *, split: str = "train",
                          root: str = "") -> DataPipeline:
    """The fixed-window legacy configuration: contiguous seq_len
    windows in corpus order, exactly-full rounds — byte-identical to
    the historical block feed (the parity/determinism gates' certified
    data)."""
    spec = f"shards:{root},window={cfg.seq_len}"
    if split != "train":
        spec += f",split={split}"
    return DataPipeline(spec, tokens_per_round=cfg.max_tokens,
                        ga_rounds=cfg.grad_accum_rounds,
                        max_seqlen=cfg.seq_len, vocab_size=cfg.vocab_size,
                        policy="greedy")


def legacy_doc_pipeline(cfg, *, split: str = "train",
                        root: str = "") -> DataPipeline:
    """The doc-aware legacy configuration: whole documents in corpus
    order, round-edge splitting with round-local max_seqlen chunking —
    byte-identical to the historical doc-aware feed (the 124M study
    curves)."""
    spec = f"shards:{root},long_policy=whole"
    if split != "train":
        spec += f",split={split}"
    return DataPipeline(spec, tokens_per_round=cfg.max_tokens,
                        ga_rounds=cfg.grad_accum_rounds,
                        max_seqlen=cfg.seq_len, vocab_size=cfg.vocab_size,
                        policy="greedy", allow_round_split=True)
