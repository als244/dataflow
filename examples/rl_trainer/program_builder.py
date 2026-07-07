"""Build the RL post-training Program by SURGERY on the standard glm52
lowering (docs/extending_programs.md: reuse the builtin task vocabulary,
change the weave).

Standard chain (one step):  embed_fwd -> block_fwd x L -> head_loss(CE)
                            -> block_bwd x L -> embed_bwd -> optimizers
RL chain (this builder):    rl_head_loss -> [block_recompute -> block_bwd
                            -> optimizer] x L (reverse) -> embed_bwd ->
                            optimizer_embed

Deltas, all mechanical:
- forward tasks DELETED — the inference engine already ran them;
- their per-layer outputs become INITIAL objects: the block inputs
  (y_embed_0_0, y_0_0_i — the "activation checkpoints") and the M
  objects (routing packs + leader index selections), supplied from the
  saved rollout;
- explicit block_recompute tasks repopulate each A from its checkpoint
  (the planner normally materializes these from rewrites; here recompute
  is unconditional — there IS no saved A);
- the CE head_loss is swapped for rl_head_loss (custom executable, same
  buffer conventions);
- everything else (bwd tasks, optimizer tasks + interleaving, embed_bwd,
  dW wiring, final_locations) is carried over UNCHANGED from the
  standard lowering — conventions by construction, not by imitation.
"""
from __future__ import annotations

from dataclasses import replace

from dataflow.core import program_from_dict, program_to_dict
from dataflow.core.validate import validate_program
from dataflow.training.glm52 import ShapedGlm52Config, lower_glm52


def build_rl_program(cfg: ShapedGlm52Config, *, steps: int = 1):
    assert cfg.train_indexer is False, (
        "the RL example consumes saved selections verbatim: build the "
        "config with train_indexer=False")
    cfg = replace(cfg, num_steps=steps)
    L = cfg.n_layers
    levels = {f"block_fwd_{s}_0_{i}": 1
              for s in range(steps) for i in range(L)}
    base = lower_glm52(cfg, recompute_levels=levels)
    d = program_to_dict(base)

    tasks = {t["id"]: t for t in d["tasks"]}
    out_spec = {}
    for t in d["tasks"]:
        for o in t.get("outputs", []):
            out_spec[o["id"]] = o

    def initial(oid, location="backing"):
        spec = dict(out_spec[oid])
        spec["location"] = location
        return spec

    from dataflow.training.families import resolve_family

    dims = resolve_family(cfg).dims_of(cfg)
    t_rows = cfg.tokens

    keep_initial = [o for o in d["initial_objects"]
                    if not o["id"].startswith("targets_")]
    new_initial = [
        {"id": "actions_0_0", "size_bytes": 4 * t_rows,
         "location": "backing", "role": "input"},
        {"id": "old_logprobs_0_0", "size_bytes": 4 * t_rows,
         "location": "backing", "role": "input"},
        {"id": "advantages_0_0", "size_bytes": 4 * t_rows,
         "location": "backing", "role": "input"},
    ]
    chain: list[dict] = []
    for s in range(steps):
        ckpt_ids = [f"y_embed_{s}_0"] + [f"y_{s}_0_{i}" for i in range(L)]
        new_initial += [initial(oid) for oid in ckpt_ids]
        new_initial += [initial(f"M_{s}_0_{i}") for i in range(L)]
        y_last = f"y_{s}_0_{L - 1}"

        head = tasks[f"head_loss_{s}_0"]
        chain.append(dict(
            head, id=f"rl_head_loss_{s}_0",
            compute_block_key="rl_head_loss",
            inputs=[y_last, "actions_0_0", "old_logprobs_0_0",
                    "advantages_0_0", "W_head"],
        ))
        chain.append(tasks[f"optimizer_head_{s}"])

        def rc_task(i: int) -> dict:
            kind = dims.kind_of(i)
            lead = dims.leader_of(i)
            extra = [f"M_{s}_0_{lead}"] if lead != i else []
            x_in = f"y_embed_{s}_0" if i == 0 else f"y_{s}_0_{i - 1}"
            return {
                "id": f"block_recompute_{s}_0_{i}",
                "inputs": [x_in, f"W_{i}", f"M_{s}_0_{i}"] + extra,
                "outputs": [dict(out_spec[f"A_{s}_0_{i}"], location="fast")],
                "runtime_us": tasks[f"block_fwd_{s}_0_{i}"]["runtime_us"],
                "compute_block_key": f"{kind}_recompute",
                "block_params": {"layer": i},
                "group": "recompute",
            }

        for i in range(L - 1, -1, -1):
            chain.append(rc_task(i))
            chain.append(tasks[f"block_bwd_{s}_0_{i}"])
            chain.append(tasks[f"optimizer_{s}_{i}"])
        chain.append(tasks[f"embed_bwd_{s}_0"])
        chain.append(tasks[f"optimizer_embed_{s}"])

    d["initial_objects"] = keep_initial + new_initial
    d["tasks"] = chain
    d["name"] = f"rl-glm52-{L}L-t{t_rows}-s{steps}"
    prog = program_from_dict(d)
    validate_program(prog)
    return prog
