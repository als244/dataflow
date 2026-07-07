"""Stand-in for the inference engine: run one rollout forward and save
everything the RL trainer consumes.

Saved artifact (single torch.save dict):
- per-layer activation checkpoints (the block INPUTS, incl. the embed
  output) and the last block's output (the head input);
- per-layer M payloads laid out EXACTLY as the runtime's M objects
  (glm52_meta_layout): leader dsa_idx selections + routing packs
  (route_w / route_ids / route_order / route_offsets, the moe_sort
  convention);
- the rollout: input tokens, sampled actions, behavior-policy logprobs
  (simulated stale policy: true logprobs + noise, so PPO ratios engage
  both clip branches), per-token advantages (per-sequence reward,
  normalized);
- the trainer starting state: packed W bytes for every weight object
  (optimizer state starts at zero on both sides).
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from dataflow.tasks import ops
from dataflow.tasks.interop import torch_view
from dataflow.tasks.layouts import glm52_meta_layout
from dataflow.training.families import resolve_family
from dataflow.training.glm52 import ShapedGlm52Config
from dataflow.training.planning import plan_program

from pinned_golden import RlGlm52


def _meta_bytes(dims, i: int, captured) -> torch.Tensor:
    """Serialize layer i's M payload in the runtime layout."""
    kind = dims.kind_of(i)
    layout = glm52_meta_layout(dims, kind)
    buf = torch.zeros(layout.total_bytes, dtype=torch.uint8)

    def field(name):
        f = next(f for f in layout.fields if f.name == name)
        n = 1
        for s in f.shape:
            n *= s
        from dataflow.tasks.interop import TORCH_DTYPE_BY_NAME

        dt = TORCH_DTYPE_BY_NAME[f.dtype]
        nbytes = n * dt.itemsize
        return buf[f.offset_bytes:f.offset_bytes + nbytes].view(dt).view(*f.shape)

    if kind in ("gdl", "gml"):
        field("dsa_idx").copy_(captured["sel"][i])
    if kind in ("gml", "gmf"):
        ids = captured["route_ids"][i]
        field("route_w").copy_(captured["route_w"][i])
        field("route_ids").copy_(ids)
        flat = ids.reshape(-1).long()
        field("route_order").copy_(
            torch.argsort(flat, stable=True).to(torch.int32))
        counts = torch.bincount(flat, minlength=dims.moe.n_experts)
        offs = torch.zeros(dims.moe.n_experts + 1, dtype=torch.int64)
        offs[1:] = torch.cumsum(counts, 0)
        field("route_offsets").copy_(offs.to(torch.int32))
    return buf


def run(out_path: str, *, seed: int = 11, reward_seed: int = 100) -> dict:
    cfg = replace(ShapedGlm52Config.tiny(), train_indexer=False)
    fam = resolve_family(cfg)
    dims = fam.dims_of(cfg)

    # trainer starting weights come from the family initializer (a real
    # deployment loads a checkpoint here); the ENGINE side reuses the
    # same bytes, so both trainers start identical
    from dataflow.runtime.device.cuda import CudaBackend

    prog = plan_program(fam.lower(cfg), fast_memory_capacity=1 << 30).program
    values = fam.initial_values(prog, cfg, CudaBackend(), seed=seed)

    def pinned(name):
        buf = values[name]
        return torch_view(buf, (buf.size_bytes,), torch.uint8).clone().cpu()

    w_bytes = {"W_embed": pinned("W_embed"), "W_head": pinned("W_head")}
    for i in range(cfg.n_layers):
        w_bytes[f"W_{i}"] = pinned(f"W_{i}")

    golden = RlGlm52.from_packed_bytes(
        dims, cfg.n_layers,
        w_bytes["W_embed"].cuda(),
        [w_bytes[f"W_{i}"].cuda() for i in range(cfg.n_layers)],
        w_bytes["W_head"].cuda(),
    )
    golden.capture = True
    golden.reset_capture()

    tokens = torch_view(values["tokens_0_0"], (dims.tokens,), torch.int32)
    tokens = tokens.long().cuda()

    with torch.no_grad():
        golden._layer_ptr = 0
        golden._group_scores = golden._group_mask = None
        x = golden.w_embed["w"][tokens]
        for w in golden.w_blocks:
            x, _ = golden.block_forward(x, w)
        y_last = x.detach().clone()
        logits = (ops.rmsnorm_reference(y_last, golden.w_head["final_norm_w"])
                  @ golden.w_head["w"].T).float()
        lse = torch.logsumexp(logits, dim=-1)

    g = torch.Generator(device="cuda").manual_seed(reward_seed)
    actions = torch.randint(0, dims.vocab_size, (dims.tokens,),
                            generator=g, device="cuda", dtype=torch.int64)
    true_lp = logits.gather(1, actions.unsqueeze(1)).squeeze(1) - lse
    old_lp = true_lp + 0.1 * torch.randn(dims.tokens, generator=g,
                                         device="cuda")
    # per-token advantages, as a GAE/GRPO pipeline would hand over
    # (per-sequence rewards degenerate at tiny scale: 1 sequence)
    adv = torch.randn(dims.tokens, generator=g, device="cuda")

    cap = golden.captured
    art = {
        "cfg_overrides": {"train_indexer": False},
        "tokens": tokens.to(torch.int32).cpu(),
        "actions": actions.to(torch.int32).cpu(),
        "old_logprobs": old_lp.float().cpu(),
        "advantages": adv.float().cpu(),
        "x_ckpt": [t.to(torch.bfloat16).cpu() for t in cap["x"]],
        "y_last": y_last.to(torch.bfloat16).cpu(),
        "m_bytes": {i: _meta_bytes(dims, i, cap)
                    for i in range(cfg.n_layers)},
        "w_bytes": w_bytes,
        "sel": cap["sel"], "route_ids": cap["route_ids"],
    }
    torch.save(art, out_path)
    print(f"rollout artifacts -> {out_path} "
          f"({sum(v.numel() for v in art['x_ckpt'])} ckpt elems, "
          f"{len(art['m_bytes'])} M payloads)")
    return art


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "rollout.pt")
