"""The optimizer-step comm variants, one module each.

``base_blocks.AdamWStep.launch`` resolves layouts, group handles, and
the shard block_param, then dispatches:

- no shard param        -> :mod:`.dp` (allreduce grads, full update;
  also the standalone/warm-up run when no group is bound)
- shard mode "rs"       -> :mod:`.rs` (byte-equal slices: one
  reduce_scatter + flat owned-slice update + one all_gather)
- any other shard param -> :mod:`.shards` (field-snapped regions:
  reduce-to-owner or replica grads, owned update, owner broadcast)

:mod:`.update` holds the per-field update core all of them share.
Every variant is certified bitwise against plain single-rank
training by the fleet gates (tests/fleet/test_zero1*_loopback.py,
test_p4a_dp_step.py).
"""
from . import dp, rs, shards, update

__all__ = ["dp", "rs", "shards", "update"]
