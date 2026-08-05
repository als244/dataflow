"""Pressure reduction for PressureFit candidate interval sets."""
from __future__ import annotations

import heapq
from bisect import bisect_left, bisect_right
from dataclasses import dataclass

from dataflow_sim.policies.pressurefit_aux.core import (
    _Facts,
    _anchors,
    _fire_task_for_interval,
    _first_use_in_interval,
    _pool_size,
    _pressure_start,
    _transfer_time,
)
from dataflow_sim.policies.pressurefit_aux.transitions import (
    _backing_state_before_spans,
    _split_transition_cost,
)
from dataflow_sim.policies.pressurefit_aux.types import (
    _CutScoreKind,
    _ResidencySpan,
)

_SplitRank = tuple[float, int, int, int, int, int]
_RankedOid = tuple[_SplitRank, str]


@dataclass(frozen=True, slots=True)
class _SplitOption:
    rank: _SplitRank
    oid: str
    interval_idx: int
    left_end: int | None
    right_start: int | None
    outbound_stream_cost: int


class _PressureReducer:
    """Greedy interval splitter for one residency strategy."""

    def __init__(
        self,
        facts: _Facts,
        intervals: dict[str, list[_ResidencySpan]],
        cap: int,
        extra_pressure: list[int],
        cut_score: _CutScoreKind,
        prefetch_headroom: bool,
        continue_headroom_cuts: bool,
    ) -> None:
        self.facts = facts
        self.intervals = intervals
        self.cap = cap
        self.extra_pressure = extra_pressure
        self.cut_score = cut_score
        self.prefetch_headroom = prefetch_headroom
        self.continue_headroom_cuts = continue_headroom_cuts
        self.anchors_by_oid = {oid: _anchors(oid, facts) for oid in intervals}
        self.backing_state_by_oid = {
            oid: _backing_state_before_spans(oid, spans, facts)
            for oid, spans in intervals.items()
        }
        self.pool = _pool_size(
            facts,
            intervals,
            prefetch_headroom=prefetch_headroom,
        )
        # Incremental relaxed-model terms. _modeled_boundary_need used to be
        # recomputed from scratch inside every worst-relaxed-boundary scan:
        # O(boundaries x objects x intervals) per RELAXED SPLIT, which was
        # 98% of near-floor planning time (profiled: 412M interval-boundary
        # checks for one 8B plan). Both terms are per-interval properties — an
        # interval departs at exactly one boundary and arrives non-blocking
        # at exactly one — so they maintain in O(1) per split, exactly like
        # `pool`. Same numbers, computed incrementally.
        self.departing = [0] * (facts.n + 1)
        self.nonblocking = [0] * (facts.n + 1)
        for oid, ivs in intervals.items():
            for iv in ivs:
                self._relaxed_terms_add(oid, iv, +1)
        self.overflow_heap = [
            (-self._strict_overflow(i), i) for i in range(len(self.pool))
        ]
        heapq.heapify(self.overflow_heap)
        self.boundary_candidates = self._build_boundary_candidates()
        self.timing_candidates = self._build_timing_candidates()
        # Capacity pressure commonly requires several consecutive cuts at the
        # same worst boundary.  A cut changes only the selected object's
        # spans, so every other object's executable split option remains
        # valid.  Retain just that boundary's options and invalidate the one
        # changed object; this avoids a large repeated scan cost without a
        # global object-by-boundary cache.
        self._split_cache_key: tuple[int, bool] | None = None
        self._split_cache: dict[str, _SplitOption | None] = {}

    def _relaxed_terms_add(self, oid: str, iv: _ResidencySpan, sign: int) -> None:
        """Add/remove one interval's departing + nonblocking contributions
        (mirrors _departing_before_next / _nonblocking_arrivals_before_next
        exactly, including their boundary-range guards)."""
        facts = self.facts
        a, b = iv
        size = sign * facts.sizes[oid]
        p = facts.producer.get(oid, -1)
        fire = _fire_task_for_interval(oid, a, b, facts)
        if (
            fire is not None
            and 0 <= fire < facts.n - 1
            and _pressure_start(
                oid,
                a,
                facts,
                prefetch_headroom=self.prefetch_headroom,
            )
            <= fire
            <= b
        ):
            self.departing[fire + 1] += size
        if a > -1 and a != p:
            boundary = _pressure_start(
                oid,
                a,
                facts,
                prefetch_headroom=self.prefetch_headroom,
            )
            if 0 <= boundary < facts.n - 1:
                first_use = _first_use_in_interval(oid, a, b, facts)
                if first_use is None or first_use > boundary + 1:
                    self.nonblocking[boundary + 1] += size

    def run(self) -> None:
        max_splits = max(1, 2 * (self.facts.n + 2) * max(1, len(self.facts.sizes)))
        for _ in range(max_splits):
            worst_idx, worst_overflow = self._worst_strict_overflow()
            if worst_overflow <= 0:
                return

            if not self.continue_headroom_cuts:
                relaxed_idx = self._worst_relaxed_boundary()
                if self._relaxed_overflow(relaxed_idx) <= 0:
                    return

            split = self._best_split_at(worst_idx, allow_timing_relief=False)
            if split is None:
                worst_idx = self._worst_relaxed_boundary()
                if self._relaxed_overflow(worst_idx) <= 0:
                    return
                split = self._best_split_at(worst_idx, allow_timing_relief=True)

            if split is None:
                self._raise_unreducible(worst_idx)

            self._apply_split(split)

        raise ValueError(
            "infeasible: pressurefit pressure reduction exceeded "
            f"{max_splits} split attempts"
        )

    def _strict_overflow(self, idx: int) -> int:
        return (
            self.pool[idx]
            + self.facts.next_reservations[idx]
            + self.extra_pressure[idx]
            - self.cap
        )

    def _relaxed_overflow(self, idx: int) -> int:
        return (
            self.pool[idx]
            - self.departing[idx]
            - self.nonblocking[idx]
            + self.facts.next_reservations[idx]
            + self.extra_pressure[idx]
            - self.cap
        )

    def _worst_strict_overflow(self) -> tuple[int, int]:
        while self.overflow_heap:
            neg_overflow, idx = self.overflow_heap[0]
            overflow = self._strict_overflow(idx)
            if -neg_overflow == overflow:
                return idx, overflow
            heapq.heappop(self.overflow_heap)
        raise RuntimeError("pressurefit internal error: empty overflow heap")

    def _worst_relaxed_boundary(self) -> int:
        return max(range(len(self.pool)), key=self._relaxed_overflow)

    def _best_split_at(
        self,
        boundary_idx: int,
        *,
        allow_timing_relief: bool,
    ) -> _SplitOption | None:
        boundary = boundary_idx - 1
        cache_key = (boundary_idx, allow_timing_relief)
        if cache_key != self._split_cache_key:
            self._split_cache_key = cache_key
            self._split_cache.clear()
        candidates = self.boundary_candidates[boundary_idx]
        if allow_timing_relief:
            # Relaxed pressure may be relieved by either an ordinary gap
            # split or the additional after-anchor timing split.  The old
            # full scan considered both; keeping only the timing index made
            # otherwise feasible tight-capacity plans appear infeasible.
            candidates = heapq.merge(
                candidates,
                self.timing_candidates[boundary_idx],
            )
        best: _SplitOption | None = None
        for rank_floor, oid in candidates:
            # All dynamic transfer costs are non-negative; the remaining rank
            # fields are fixed by this anchor gap. Once the next lower bound
            # cannot beat the best executable option, no later candidate can.
            if best is not None and rank_floor >= best.rank:
                break
            if oid in self._split_cache:
                option = self._split_cache[oid]
            else:
                option = self._split_option_for_oid(
                    oid,
                    boundary,
                    allow_timing_relief=allow_timing_relief,
                )
                self._split_cache[oid] = option
            if option is not None and (
                best is None or option.rank < best.rank
            ):
                best = option
        if best is None:
            return None
        backing_current, previous_departure = self.backing_state_by_oid[
            best.oid
        ][best.interval_idx]
        actual_cost = _split_transition_cost(
            best.oid,
            self.intervals[best.oid],
            best.interval_idx,
            best.left_end,
            best.right_start,
            self.facts,
            backing_current_before=backing_current,
            previous_departure_before=previous_departure,
        )
        if actual_cost is None:
            raise RuntimeError(
                "pressurefit internal error: indexed split has no executable "
                f"transition for {best.oid!r}"
            )
        if actual_cost != best.outbound_stream_cost:
            raise RuntimeError(
                "pressurefit internal error: cached split transfer cost "
                f"{best.outbound_stream_cost} disagrees with transition cost "
                f"{actual_cost} for {best.oid!r}"
            )
        return best

    def _split_option_for_oid(
        self,
        oid: str,
        boundary: int,
        *,
        allow_timing_relief: bool,
    ) -> _SplitOption | None:
        ivs = self.intervals.get(oid)
        if not ivs:
            return None
        # Intervals are disjoint and ordered, and coverage windows
        # [a, b] never overlap, so at most one interval can cover
        # `boundary`: the last one whose start `a` is <= boundary + 1.
        idx = bisect_right(ivs, boundary + 1, key=lambda iv: iv[0]) - 1
        if idx < 0:
            return None
        a, b = ivs[idx]
        if not (
            _pressure_start(
                oid,
                a,
                self.facts,
                prefetch_headroom=self.prefetch_headroom,
            )
            <= boundary
            <= b
        ):
            return None
        split_edges = self._split_edges_for_interval(
            oid, a, b, boundary, allow_timing_relief=allow_timing_relief,
        )
        if split_edges is None:
            return None
        left_end, right_start = split_edges
        option = self._ranked_split_option(
            oid, idx, a, b, left_end, right_start,
        )
        if option is None:
            return None
        return option

    def _split_edges_for_interval(
        self,
        oid: str,
        a: int,
        b: int,
        boundary: int,
        *,
        allow_timing_relief: bool,
    ) -> tuple[int | None, int | None] | None:
        anchors = self.anchors_by_oid.get(oid)
        if anchors is None:
            anchors = _anchors(oid, self.facts)
            self.anchors_by_oid[oid] = anchors
        lo = bisect_left(anchors, a)
        hi = bisect_right(anchors, b)
        exact_pos = bisect_left(anchors, boundary, lo, hi)
        is_anchor = exact_pos < hi and anchors[exact_pos] == boundary
        if is_anchor:
            if not allow_timing_relief:
                return None
            right_pos = bisect_left(anchors, boundary + 1, lo, hi)
            if right_pos >= hi:
                return None
            left_end = boundary
            right_start = anchors[right_pos]
            if _fire_task_for_interval(oid, a, left_end, self.facts) != boundary:
                return None
            return left_end, right_start

        left_pos = bisect_right(anchors, boundary - 1, lo, hi) - 1
        right_pos = bisect_left(anchors, boundary + 1, lo, hi)
        left_end = anchors[left_pos] if left_pos >= lo else None
        right_start = anchors[right_pos] if right_pos < hi else None
        return left_end, right_start

    def _ranked_split_option(
        self,
        oid: str,
        interval_idx: int,
        interval_start: int,
        interval_end: int,
        left_end: int | None,
        right_start: int | None,
    ) -> _SplitOption | None:
        left_b = left_end if left_end is not None else interval_start - 1
        right_a = right_start if right_start is not None else interval_end + 1
        gap_len = right_a - left_b - 1
        if gap_len <= 0:
            return None
        drops_init = left_end is None and interval_start == -1
        backing_current = self.backing_state_by_oid[oid][interval_idx][0]
        if drops_init:
            if not backing_current:
                return None
            stream_cost = 0
        elif left_end is not None:
            mutations = self.facts.mutators.get(oid, ())
            mutation_pos = bisect_left(mutations, interval_start + 1)
            left_dirty = (
                mutation_pos < len(mutations)
                and mutations[mutation_pos] <= left_end + 1
            )
            stream_cost = 0 if backing_current and not left_dirty else 1
        else:
            return None
        first_use = self.facts.uses.get(oid, (self.facts.n,))[0]
        timing_penalty = self._split_timing_penalty(
            oid,
            interval_idx,
            interval_start,
            left_end,
            right_start,
            stream_cost,
        )
        if self.cut_score == "min-stall":
            rank = (
                timing_penalty,
                stream_cost,
                0 if drops_init else 1,
                -first_use,
                -self.facts.sizes[oid],
                -gap_len,
            )
        else:
            ranking_stream_cost = stream_cost
            rank = (
                float(ranking_stream_cost),
                0,
                0 if drops_init else 1,
                -first_use,
                -self.facts.sizes[oid],
                -gap_len,
            )
        return _SplitOption(
            rank,
            oid,
            interval_idx,
            left_end,
            right_start,
            stream_cost,
        )

    def _split_timing_penalty(
        self,
        oid: str,
        interval_idx: int,
        interval_start: int,
        left_end: int | None,
        right_start: int | None,
        outbound_stream_cost: int,
    ) -> float:
        """Estimate unavoidable ideal-stream delay introduced by a split.

        A clean release can still be a poor choice when the legal re-prefetch
        window begins immediately before the next use. Ranking it below a
        fully overlappable offload avoids optimizing transfer count at the
        expense of selected makespan; simulator replay remains authoritative.
        """
        if right_start is None or not self.facts.n:
            return 0.0

        right_end = self.intervals[oid][interval_idx].end
        first_use = _first_use_in_interval(
            oid, right_start, right_end, self.facts,
        )
        if left_end is None:
            previous_departure = self.backing_state_by_oid[oid][interval_idx][1]
            earliest_trigger = (
                0 if previous_departure is None else previous_departure + 1
            )
            departure = previous_departure
        else:
            departure = _fire_task_for_interval(
                oid, interval_start, left_end, self.facts,
            )
            if departure is None:
                return float("inf")
            earliest_trigger = departure + 1
        if first_use is None:
            # Terminal-only prefetches are intentionally emitted at their
            # final entry boundary rather than treated as deadline-packed
            # jobs. Match that executable policy in the ranking estimate.
            earliest_trigger = max(earliest_trigger, max(0, right_start))
        if earliest_trigger >= self.facts.n:
            return float("inf")

        trigger_time = self.facts.task_end[earliest_trigger]
        source_ready = 0.0
        if outbound_stream_cost:
            assert departure is not None
            source_ready = (
                self.facts.task_end[departure]
                + _transfer_time(
                    self.facts.sizes[oid], self.facts.outbound_bandwidth,
                )
            )
        completion = max(trigger_time, source_ready) + _transfer_time(
            self.facts.sizes[oid], self.facts.inbound_bandwidth,
        )
        deadline = (
            self.facts.task_start[first_use]
            if first_use is not None
            else self.facts.task_end[-1]
        )
        return max(0.0, completion - deadline)

    def _apply_split(self, split: _SplitOption) -> None:
        # A split changes only this object's future options. Other cached
        # options at the same pressure boundary remain exact.
        self._split_cache.pop(split.oid, None)
        a, b = self.intervals[split.oid][split.interval_idx]
        pieces: list[_ResidencySpan] = []
        if split.left_end is not None:
            pieces.append(_ResidencySpan(a, split.left_end))
        if split.right_start is not None:
            pieces.append(_ResidencySpan(split.right_start, b))
        if pieces == [(a, b)]:
            raise ValueError(
                "infeasible: pressurefit pressure reduction selected a "
                "non-progressing split"
            )

        old_span = _ResidencySpan(a, b)
        self._relaxed_terms_add(split.oid, old_span, -1)
        for piece in pieces:
            self._relaxed_terms_add(split.oid, piece, +1)
        changed_indices = _subtract_removed_interval_pressure(
            self.facts,
            self.pool,
            split.oid,
            old_span,
            pieces,
            prefetch_headroom=self.prefetch_headroom,
        )
        for changed_idx in changed_indices:
            heapq.heappush(
                self.overflow_heap,
                (-self._strict_overflow(changed_idx), changed_idx),
            )

        self.intervals[split.oid][split.interval_idx:split.interval_idx + 1] = pieces
        if not self.intervals[split.oid]:
            del self.intervals[split.oid]
            self.backing_state_by_oid.pop(split.oid, None)
        else:
            self.backing_state_by_oid[split.oid] = _backing_state_before_spans(
                split.oid, self.intervals[split.oid], self.facts,
            )

    def _rank_floor(
        self,
        oid: str,
        interval_start: int,
        interval_end: int,
        left_end: int | None,
        right_start: int | None,
    ) -> _SplitRank:
        """Return an admissible lower bound for one split's dynamic rank."""
        left_b = left_end if left_end is not None else interval_start - 1
        right_a = right_start if right_start is not None else interval_end + 1
        gap_len = right_a - left_b - 1
        drops_init = left_end is None and interval_start == -1
        first_use = self.facts.uses.get(oid, (self.facts.n,))[0]
        return (
            0.0,  # timing penalty / stream cost is non-negative
            0,  # actual outbound transfer-stream cost is zero or one
            0 if drops_init else 1,
            -first_use,
            -self.facts.sizes[oid],
            -gap_len,
        )

    def _build_boundary_candidates(self) -> list[list[_RankedOid]]:
        """Index and lower-bound-rank strict split candidates by boundary."""
        by_boundary: list[list[_RankedOid]] = [
            [] for _ in range(self.facts.n + 1)
        ]
        for oid, ivs in self.intervals.items():
            anchors = self.anchors_by_oid.get(oid)
            if anchors is None:
                anchors = _anchors(oid, self.facts)
                self.anchors_by_oid[oid] = anchors
            for a, b in ivs:
                lo = bisect_left(anchors, a)
                hi = bisect_right(anchors, b)
                span_anchors = anchors[lo:hi]
                left_end: int | None = None
                cursor = _pressure_start(
                    oid,
                    a,
                    self.facts,
                    prefetch_headroom=self.prefetch_headroom,
                )
                for right_start in span_anchors:
                    if cursor < right_start:
                        rank_floor = self._rank_floor(
                            oid, a, b, left_end, right_start,
                        )
                        for boundary in range(cursor, right_start):
                            by_boundary[boundary + 1].append((rank_floor, oid))
                    left_end = right_start
                    cursor = right_start + 1
                if cursor <= b:
                    rank_floor = self._rank_floor(
                        oid, a, b, left_end, None,
                    )
                    for boundary in range(cursor, b + 1):
                        by_boundary[boundary + 1].append((rank_floor, oid))
        for candidates in by_boundary:
            candidates.sort()
        return by_boundary

    def _build_timing_candidates(self) -> list[list[_RankedOid]]:
        """Index objects that can split immediately after an anchor.

        Timing-relief selection previously scanned every object at every
        split, making long chains quadratic even though only objects anchored
        at the selected boundary can qualify.  Splitting never creates new
        anchors, so an index built from the seed remains a safe superset; the
        normal per-object check filters stale entries after later splits.
        """
        by_boundary: list[list[_RankedOid]] = [
            [] for _ in range(self.facts.n + 1)
        ]
        for oid, spans in self.intervals.items():
            anchors = self.anchors_by_oid[oid]
            for a, b in spans:
                lo = bisect_left(anchors, a)
                hi = bisect_right(anchors, b)
                for position in range(lo, hi - 1):
                    boundary = anchors[position]
                    right_start = anchors[position + 1]
                    if right_start - boundary <= 1:
                        continue
                    if _fire_task_for_interval(
                        oid, a, boundary, self.facts,
                    ) != boundary:
                        continue
                    rank_floor = self._rank_floor(
                        oid, a, b, boundary, right_start,
                    )
                    by_boundary[boundary + 1].append((rank_floor, oid))
        for candidates in by_boundary:
            candidates.sort()
        return by_boundary

    def _raise_unreducible(self, boundary_idx: int) -> None:
        boundary = boundary_idx - 1
        raise ValueError(
            f"infeasible: pressurefit cannot reduce boundary {boundary} "
            f"under fast_memory_capacity={self.cap}"
        )


def _reduce_to_fit(
    facts: _Facts,
    intervals: dict[str, list[_ResidencySpan]],
    cap: int | None,
    extra_pressure: list[int] | None = None,
    *,
    cut_score: _CutScoreKind = "min-stall",
    prefetch_headroom: bool = True,
    continue_headroom_cuts: bool = True,
) -> None:
    """Mutate `intervals` into a pressure-fit interval set."""
    if cap is None:
        return
    if extra_pressure is None:
        extra_pressure = [0] * (facts.n + 1)
    _PressureReducer(
        facts,
        intervals,
        cap,
        extra_pressure,
        cut_score,
        prefetch_headroom,
        continue_headroom_cuts,
    ).run()


def _subtract_removed_interval_pressure(
    facts: _Facts,
    pool: list[int],
    oid: str,
    old: _ResidencySpan,
    new_pieces: list[_ResidencySpan],
    *,
    prefetch_headroom: bool = True,
) -> list[int]:
    """Update a precomputed pool after splitting one interval."""
    changed_indices: list[int] = []
    old_a, old_b = old
    old_start = max(
        -1,
        _pressure_start(
            oid,
            old_a,
            facts,
            prefetch_headroom=prefetch_headroom,
        ),
    )
    old_end = min(facts.n - 1, old_b)
    if old_start > old_end:
        return changed_indices

    normalized_pieces: list[tuple[int, int]] = []
    for a, b in new_pieces:
        start = max(
            -1,
            _pressure_start(
                oid,
                a,
                facts,
                prefetch_headroom=prefetch_headroom,
            ),
        )
        end = min(facts.n - 1, b)
        if start <= end:
            normalized_pieces.append((start, end))
    normalized_pieces.sort()

    size = facts.sizes[oid]
    cursor = old_start
    for start, end in normalized_pieces:
        if cursor <= start - 1:
            for boundary in range(cursor, start):
                idx = boundary + 1
                pool[idx] -= size
                changed_indices.append(idx)
        cursor = max(cursor, end + 1)
    if cursor <= old_end:
        for boundary in range(cursor, old_end + 1):
            idx = boundary + 1
            pool[idx] -= size
            changed_indices.append(idx)
    return changed_indices
