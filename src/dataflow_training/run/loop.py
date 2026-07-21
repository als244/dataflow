"""The fleet step loop: rounds in, losses out — global-denominator
steps, profiler brackets, v2 checkpoints at boundaries, per-rank peak
accounting. Split from fleet.py at phase close."""
from __future__ import annotations

import threading
import time
from dataclasses import replace
from pathlib import Path

from dataflow.core.jsonio import program_to_dict
from dataflow_training.lowering.planning import plan_program

from .driver import RunResult, init_model
from .presets import cfg_dict, tokens_per_step
from .checkpointing import save_checkpoint
from ..distributed.grouped_lowering import GroupedBuildVariant, lower_with_group
from ..distributed.ranks import StepRun, put_rank_rounds
from ..distributed.sharding import ALL_RANKS, tp_view

def fleet_loop(ranks, gspec, recipe, pipeline, steps, *, budgets, seed,
               log, log_every, tokens_step, r_global,
               profile: dict | None = None,
               parallels=None,
               tp_mode: bool = False, checkpoint: dict | None = None,
               ck_record: dict | None = None,
               zero1rs_world: int | None = None,
               execute_padding: bool = False,
               tp_mlp: bool = False) -> RunResult:
    world = len(ranks)
    start_step = int(ck_record["step"]) if ck_record else 0

    cursor = ck_record.get("data_cursor") if ck_record else None
    if cursor is not None:
        stepper = pipeline(cursor)
    else:
        stepper = pipeline(None)
        if start_step:
            from dataflow_training.data.pipeline import fast_forward

            log(f"[fleet] checkpoint has no data cursor — fast-"
                f"forwarding the pipeline {start_step} steps (CPU)")
            fast_forward(stepper, start_step)
    tokens_per_round = ranks[0].cfg.max_tokens
    step_packed = stepper.next_step()      # the START step's rounds
    step_lens_by_rank: dict[int, dict] = {}
    last_cursor = step_packed.cursor_after

    # ---- per-rank register + warm-up ----------------------------------
    for i, rank in enumerate(ranks):
        par = parallels[i] if parallels else None
        rank_group = gspec.name if world > 1 else None
        # the loop runs ONE optimizer step per client.run with fresh
        # data — the program must carry exactly one step slot. Lowering
        # with the run-length num_steps builds N slots of which only
        # slot 0 is ever fed: the others execute on init-residue
        # buffers, silently training junk every iteration (THE
        # solo-vs-DP divergence root cause).
        step_cfg = replace(rank.cfg, num_steps=1)
        planned = plan_program(
            lower_with_group(step_cfg, rank_group,
                             parallel=par, zero1rs_world=zero1rs_world),
            fast_memory_capacity=int(budgets[i] * 1024 ** 3),
            recompute=True,
            build_variant=GroupedBuildVariant(step_cfg, rank_group,
                                              parallel=par,
                                              zero1rs_world=zero1rs_world))
        extra_slots = [s.id for s in planned.program.initial_objects
                       if s.id.startswith("tokens_") and
                       not s.id.startswith("tokens_0_")]
        if extra_slots:
            raise RuntimeError(
                f"{rank.name}: program carries step slots beyond 0 "
                f"({extra_slots[:3]}...) — the loop feeds one step per "
                f"run; multi-slot programs silently train junk")
        prog_dict = program_to_dict(planned.program)
        rank.prog_dict = prog_dict          # record v2: saved beside
                                            # the checkpoint artifacts
        from dataflow_training.run.presets import resolver_family

        fam_name = resolver_family(rank.cfg)
        resolver = {"kind": "model_family", "family": fam_name,
                    "cfg": cfg_dict(step_cfg),
                    "hyper": recipe.hyper_spec()}
        init_kwargs = {"seed": seed}
        if zero1rs_world is not None:
            init_kwargs["object_sizes"] = {
                s.id: s.size_bytes
                for s in planned.program.initial_objects
                if s.id.startswith("O_")}
            o_bytes = sum(init_kwargs["object_sizes"].values())
            log(f"[fleet] {rank.name}: byte-equal sharded optimizer "
                f"state {o_bytes / 1024 ** 3:.2f} GiB")
        if par is not None and par.plan is not None:
            narrowed = any(a.resident != ALL_RANKS
                           for a in par.plan.assignments)
            prefixes = ("O_", "W_") if narrowed else ("O_",)
            if narrowed:
                view = tp_view(par.plan, par.rank)
                init_kwargs["tp_view"] = {
                    root: {f: list(sl) for f, sl in per.items()}
                    for root, per in view.items()}
            # the daemon must allocate THIS RANK's shrunken objects —
            # send the registered program's own sizes
            init_kwargs["object_sizes"] = {
                s.id: s.size_bytes
                for s in planned.program.initial_objects
                if s.id.startswith(prefixes)}
            o_bytes = sum(b for oid, b in init_kwargs["object_sizes"].items()
                          if oid.startswith("O_"))
            log(f"[fleet] {rank.name}: sharded optimizer state "
                f"{o_bytes / 1024 ** 3:.2f} GiB"
                + (" (tp: sharded weights too)" if narrowed else ""))
        init_model(rank.client, fam_name, cfg_dict(rank.cfg), **init_kwargs)
        step_lens_by_rank[i] = put_rank_rounds(rank, step_packed,
                                               tokens_per_round,
                                               execute_padding=execute_padding,
                                               require_full=tp_mlp)
        reg = rank.client.register_program(prog_dict, resolver=resolver)
        missing = reg["bindings"]["missing_inputs"]
        if missing:
            raise RuntimeError(f"{rank.name}: unbound {missing}")
        rank.prog_id = reg["prog_id"]
        rank.persist_ids = sorted(
            s.id for s in planned.program.initial_objects
            if s.id.startswith(("W_", "O_")))
        log(f"[fleet] {rank.name}: registered {rank.prog_id} "
            f"(rounds {rank.rounds}, budget {budgets[i]} GiB)")
        # WARM-UP (group absent => comm skips): compiles + loads every
        # kernel; a first launch during a parked collective wedges the
        # device. Then RE-SEED and RE-PUT (init refills token buffers).
        warm = rank.client.run(rank.prog_id,
                               args={"step": 0, "valid_rows": tokens_step})
        if warm.get("state") != "done":
            raise RuntimeError(f"{rank.name} warm-up: {warm}")
        if ck_record is not None:
            # RESUME: restore over the warm-up's mutated state (this
            # ordering makes the kernel warm-up harmless), then feed
            # the START step's rounds. Restores the shared artifact
            # (rank-0-deduped state, distributed to this host by the
            # conductor) plus this rank's own artifact, if any.
            from .checkpoint_record import artifacts_for_restore

            restored_step = None
            step_dir = Path(ck_record["_step_dir"])
            for art in artifacts_for_restore(ck_record, i):
                res = rank.client.restore_snapshot(
                    str(step_dir / art), overwrite=True)
                restored_step = res["client_meta"]["step"]
            if restored_step != start_step:
                raise RuntimeError(
                    f"{rank.name}: restored step {restored_step} != "
                    f"resume step {start_step}")
            step_lens_by_rank[i] = put_rank_rounds(rank, step_packed,
                                                   tokens_per_round,
                                                   execute_padding=execute_padding,
                                                   require_full=tp_mlp)
            log(f"[fleet] {rank.name}: restored checkpoint @ step "
                f"{start_step}")
        else:
            init_model(rank.client, fam_name, cfg_dict(rank.cfg),
                       **init_kwargs)
            step_lens_by_rank[i] = put_rank_rounds(rank, step_packed,
                                                   tokens_per_round,
                                                   execute_padding=execute_padding,
                                                   require_full=tp_mlp)
            log(f"[fleet] {rank.name}: warm-up done, re-seeded")

    if world > 1:
        ranks[0].client._call("create_peer_group",
                              {"name": gspec.name,
                               "members": list(gspec.members),
                               "backend": gspec.backend})
        log(f"[fleet] {gspec.name} group up ({gspec.backend}, "
            f"world {world})")
    else:
        log("[fleet] world 1 — no peer group; solo program")

    res = RunResult(backend="fleet-tp" if tp_mode else "fleet-dp",
                    budget_gib=budgets[0],
                    meta={"seed": seed, "world": world,
                          "hosts": [r.name for r in ranks],
                          "rank_rounds": [len(r.rounds) for r in ranks],
                          "prog_ids": [r.prog_id for r in ranks],
                          "budgets_gib": list(budgets),
                          "tokens_per_step": tokens_step})

    if ck_record is not None:
        # continuous curve across the resume: pre-checkpoint losses
        # ride the checkpoint record
        res.losses.extend(ck_record.get("losses", []))
        res.meta["resumed_from_step"] = start_step

    # ---- lockstep step loop -------------------------------------------
    prof_start = profile.get("start") if profile else None
    prof_stop = profile.get("stop") if profile else None
    peaks = {r.name: {} for r in ranks}
    for step in range(start_step, steps):
        if prof_start is not None and step == prof_start:
            for rank in ranks:
                rank.client.profiler_control("start")
            log(f"[fleet] profiler capture STARTED before step {step}")
        t0 = time.perf_counter()
        if step > start_step:
            step_packed = stepper.next_step()
            last_cursor = step_packed.cursor_after
            for k, rank in enumerate(ranks):
                step_lens_by_rank[k] = put_rank_rounds(
                    rank, step_packed, tokens_per_round,
                    execute_padding=execute_padding,
                    require_full=tp_mlp)
        # GLOBAL-DENOMINATOR: every rank normalizes by the step's total
        # (tp replicas share ONE batch — the packed step IS that batch)
        valid = step_packed.valid_rows
        jobs = [StepRun(rank, step, valid, step_lens_by_rank.get(k))
                for k, rank in enumerate(ranks)]
        threads = [threading.Thread(target=j) for j in jobs]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for j in jobs:
            if j.error is not None:
                raise RuntimeError(f"step {step} {j.rank.name}: {j.error}")
        if tp_mode:
            # replicas compute the SAME loss (allreduced activations);
            # take rank 0's and demand the others agree — divergence
            # here means the tp group plane broke. NaN checked
            # explicitly: abs(nan - nan) > tol is False, so a NaN run
            # sails through a plain tolerance test (incident lesson)
            per_rank = [sum(j.fetched.values()) for j in jobs]
            step_loss = per_rank[0]
            for k, other in enumerate(per_rank):
                if other != other:
                    raise RuntimeError(
                        f"step {step}: rank {k} loss is NaN")
                if abs(other - step_loss) > 1e-3:
                    raise RuntimeError(
                        f"step {step}: tp replicas disagree — rank 0 "
                        f"{step_loss} vs rank {k} {other}")
        else:
            step_loss = sum(sum(j.fetched.values()) for j in jobs)
        for j in jobs:
            for key in ("peak_fast_bytes", "torch_reserved_peak",
                        "torch_allocated_peak", "placement_extent_bytes"):
                val = j.out.get(key) if j.out else None
                if val is not None:
                    prev = peaks[j.rank.name].get(key, 0)
                    peaks[j.rank.name][key] = max(prev, int(val))
        dt = time.perf_counter() - t0
        res.losses.append(step_loss)
        res.step_wall_s.append(dt)
        res.tok_per_s.append(tokens_step / dt)
        if step % log_every == 0 or step == steps - 1:
            log(f"[fleet] step {step:4d}/{steps}  loss {step_loss:.4f}  "
                f"lr {recipe.lr(step):.2e}  {tokens_step / dt:.0f} tok/s"
                f"  ({dt:.2f}s)")
        if prof_stop is not None and step == prof_stop:
            for rank in ranks:
                rank.client.profiler_control("stop")
            log(f"[fleet] profiler capture STOPPED after step {step} "
                f"(training continues)")
        if (checkpoint is not None
                and (step + 1) % checkpoint["every"] == 0
                and step + 1 < steps):
            save_checkpoint(
                ranks, checkpoint, step + 1,
                {"run": checkpoint["run"], "world": world,
                 "hosts": [r.name for r in ranks],
                 "rank_rounds": [len(r.rounds) for r in ranks],
                 "backend": gspec.backend, "seed": seed,
                 "data_cursor": last_cursor},
                res.losses, log)
    # peak DEVICE + HOST accounting, recorded BY DEFAULT per rank
    for rank in ranks:
        try:
            status = rank.client.engine_status()
            pools = status.get("pools_backing") or status.get("store")
            if pools:
                peaks[rank.name]["host_backing"] = pools
        except Exception:
            pass
    res.meta["peaks"] = peaks
    gib = 1024 ** 3
    for name, row in peaks.items():
        fast = row.get("peak_fast_bytes", 0) / gib
        torch_peak = row.get("torch_reserved_peak", 0) / gib
        log(f"[fleet] {name}: peak fast {fast:.2f} GiB, torch reserved "
            f"{torch_peak:.2f} GiB, host {row.get('host_backing')}")
    return res
