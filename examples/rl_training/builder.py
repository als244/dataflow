"""Family-GENERIC RL program builder: order-preserving surgery on any
family's standard lowering (docs/extending_programs.md).

Walk the standard chain in order and:
- DROP embed_fwd / block_fwd tasks (the inference engine ran them);
- INSERT a block_recompute before each block_bwd, derived from the
  dropped forward task itself: same inputs (checkpoint + weights + any
  M the forward consumed, e.g. a glm52 leader's shared selection), plus
  the layer's own M if the forward emitted one; output = the A object;
  compute key = the forward's with ``_fwd -> _recompute``;
- REPLACE head_loss (CE) with rl_head_loss in place: inputs become
  (last-block output, actions, old_logprobs, advantages, <same weight
  object the CE head read — W_head, or W_embed for tied families>);
  outputs unchanged (dy, loss, dW);
- keep EVERYTHING else (backwards, optimizers, embed_bwd) untouched.

Afterwards, any object still referenced but no longer produced (the
per-layer checkpoints y_*, the M payloads) becomes an INITIAL object —
supplied by the inference engine — and unreferenced leftovers
(targets_*) are dropped. Works for any family because it never names a
kind, an optimizer id, or a metadata rule: it reads them off the
standard program.
"""
from __future__ import annotations

from dataclasses import replace

from dataflow.core import program_from_dict, program_to_dict
from dataflow.core.validate import validate_program

ROLLOUT_INPUTS = ("actions_0_0", "old_logprobs_0_0", "advantages_0_0")


def build_rl_program(fam, cfg, *, steps: int = 1):
    cfg = replace(cfg, num_steps=steps)
    n_layers = cfg.n_layers
    levels = {f"block_fwd_{s}_0_{i}": 1
              for s in range(steps) for i in range(n_layers)}
    base = fam.lower(cfg, recompute_levels=levels)
    d = program_to_dict(base)

    out_spec = {}
    for t in d["tasks"]:
        for o in t.get("outputs", []):
            out_spec[o["id"]] = o

    chain: list[dict] = []
    dropped_fwd: dict[str, dict] = {}       # by task id
    for t in d["tasks"]:
        tid = t["id"]
        if tid.startswith("embed_fwd_") or tid.startswith("block_fwd_"):
            dropped_fwd[tid] = t
            continue
        if tid.startswith("head_loss_"):
            s = tid.split("_")[2]
            y_last = t["inputs"][0]
            w_src = t["inputs"][-1]          # W_head, or W_embed when tied
            chain.append(dict(
                t, id=f"rl_head_loss_{s}_0",
                compute_block_key="rl_head_loss",
                inputs=[y_last, *ROLLOUT_INPUTS, w_src],
            ))
            continue
        if tid.startswith("block_bwd_"):
            _, _, s, r, i = tid.split("_")
            fwd = dropped_fwd[f"block_fwd_{s}_{r}_{i}"]
            own_m = [o["id"] for o in fwd.get("outputs", [])
                     if o["id"].startswith("M_")]
            a_out = next(o for o in fwd["outputs"]
                         if o["id"].startswith("A_"))
            chain.append({
                "id": f"block_recompute_{s}_{r}_{i}",
                "inputs": list(fwd["inputs"]) + own_m,
                "outputs": [dict(a_out, location="fast")],
                "runtime_us": fwd["runtime_us"],
                "compute_block_key": fwd["compute_block_key"].replace(
                    "_fwd", "_recompute"),
                "block_params": fwd.get("block_params", {}),
                "group": "recompute",
            })
        chain.append(t)

    # ---- objects: initial-ize the inference-supplied ones ----
    produced = {o["id"] for t in chain for o in t.get("outputs", [])}
    referenced = {oid for t in chain
                  for oid in list(t.get("inputs", []))
                  + list(t.get("mutates", []))}
    have_initial = {o["id"] for o in d["initial_objects"]}
    t_rows = cfg.tokens

    new_initial = [dict(out_spec[oid], location="backing")
                   for oid in sorted(referenced - produced - have_initial
                                     - set(ROLLOUT_INPUTS))]
    new_initial += [
        {"id": oid, "size_bytes": 4 * t_rows, "location": "backing",
         "role": "input"} for oid in ROLLOUT_INPUTS
    ]
    kept = [o for o in d["initial_objects"] if o["id"] in referenced]

    d["initial_objects"] = kept + new_initial
    d["tasks"] = chain
    d["name"] = f"rl-{d['name']}-s{steps}"
    prog = program_from_dict(d)
    validate_program(prog)
    return prog
