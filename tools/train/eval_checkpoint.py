#!/usr/bin/env python
"""Fineweb-VAL loss of a solo-run checkpoint — the nanogpt-comparable axis.

llm.c / modded-nanogpt speedrun curves report held-out loss on the
fineweb val shard; our training logs record train CE. This loads W_*
straight from a snapshot payload into the family's pure-torch
reference twin and evaluates val CE, so an in-flight or finished run
can be placed on the published axis at any checkpoint.

    python tools/eval_checkpoint.py l3_1b_engine_t512k_adamw --preset l3_1b
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import torch

from dataflow_training.model_families import bridges
from dataflow_training.run import presets as P
from dataflow_training.data.fineweb import make_feed

CKPTS = _ROOT / "results" / "pretrain" / "checkpoints"


class CheckpointPayload:
    """get_bytes over a snapshot: manifest offsets into payload.bin."""

    def __init__(self, ckpt_dir: Path):
        manifest = json.loads((ckpt_dir / "manifest.json").read_text())
        self.step = int(manifest["client_meta"].get("step", -1))
        self.index = {o["id"]: o["payload"] for o in manifest["objects"]}
        self.payload = open(ckpt_dir / "payload.bin", "rb")

    def __call__(self, oid: str) -> "torch.Tensor":
        seg = self.index[oid]
        self.payload.seek(int(seg["offset"]))
        raw = bytearray(self.payload.read(int(seg["size"])))
        return torch.frombuffer(raw, dtype=torch.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run name under results/pretrain/checkpoints/")
    ap.add_argument("--preset", required=True)
    ap.add_argument("--step", type=int, default=None,
                    help="checkpoint step (default: newest complete)")
    ap.add_argument("--val-tokens", type=int, default=10_485_760)
    ap.add_argument("--batch-tokens", type=int, default=8192)
    args = ap.parse_args()

    run_dir = CKPTS / args.run
    if args.step is not None:
        ck = run_dir / f"step_{args.step:06d}"
        if not (ck / "manifest.json").is_file():
            print(f"no complete checkpoint at {ck}", file=sys.stderr)
            return 1
    else:
        manifests = sorted(run_dir.glob("step_*/manifest.json"))
        if not manifests:
            print(f"no complete checkpoints under {run_dir}", file=sys.stderr)
            return 1
        ck = manifests[-1].parent

    cfg = P.resolve_preset(args.preset)
    payload = CheckpointPayload(ck)
    from dataflow_training.model_families.families import resolve_family

    fam = resolve_family(cfg)
    dims = fam.derive_dims(cfg)
    model = bridges.build_reference_model(cfg)
    bridges.load_reference_init(model, cfg, dims, payload)
    model.eval()

    B = args.batch_tokens // cfg.seq_len
    T = cfg.seq_len
    feed = make_feed(args.batch_tokens, split="val")
    rounds = max(1, args.val_tokens // args.batch_tokens)
    total_nll = 0.0
    total_valid = 0
    with torch.no_grad():
        for r in range(rounds):
            tok, tgt = feed(r)
            valid = int((tgt >= 0).sum())
            loss = model.loss(tok.cuda().view(B, T), tgt.cuda().view(B, T))
            total_nll += float(loss) * valid
            total_valid += valid
    val = total_nll / total_valid
    print(f"{args.run} @ step {payload.step}: fineweb-val loss "
          f"{val:.4f}  (ppl {torch.tensor(val).exp():.2f}; "
          f"{total_valid} tokens)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
