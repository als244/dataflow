#!/usr/bin/env python
"""Held-out validation loss of a solo-run checkpoint.

Published pretraining curves report loss on a held-out val split;
our training logs record train CE. This loads W_*
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
from dataflow_training.data.pipeline import DataPipeline

CKPTS = _ROOT / "results" / "pretrain" / "checkpoints"


class CheckpointPayload:
    """get_bytes over a checkpoint STEP DIRECTORY: restore every
    artifact the record lists, in record order, into a scratch fake
    engine — ranged saves reassemble exactly as resume does — and
    read objects from it."""

    def __init__(self, ckpt_dir: Path):
        import tempfile
        import threading
        import time

        from dataflow.service import EngineClient, EngineConfig, Server
        from dataflow_training.run.checkpoint_record import (
            artifacts_for_restore,
            read_record,
        )

        rec = read_record(ckpt_dir)
        self.step = int(rec["step"])
        sock = str(Path(tempfile.mkdtemp()) / "eval.sock")
        server = Server(EngineConfig(socket_path=sock, fake=True,
                                     slab_backing_gib=1.0))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        for _ in range(600):
            try:
                EngineClient(sock, client_name="probe").close()
                break
            except OSError:
                time.sleep(0.01)
        self.client = EngineClient(sock, client_name="eval")
        for rank in range(rec["world"]):
            for art in artifacts_for_restore(rec, rank):
                self.client.restore_snapshot(str(ckpt_dir / art),
                                             overwrite=True)

    def __call__(self, oid: str) -> "torch.Tensor":
        raw = bytearray(bytes(self.client.get_object(oid)))
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
        if not ((ck / "checkpoint_record.json").is_file()
                or (ck / "manifest.json").is_file()):
            print(f"no complete checkpoint at {ck}", file=sys.stderr)
            return 1
    else:
        marks = sorted(run_dir.glob("step_*/checkpoint_record.json"))             or sorted(run_dir.glob("step_*/manifest.json"))
        if not marks:
            print(f"no complete checkpoints under {run_dir}", file=sys.stderr)
            return 1
        ck = marks[-1].parent

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
    pipeline = DataPipeline(
        f"shards:,window={T},split=val",
        tokens_per_round=args.batch_tokens, ga_rounds=1,
        max_seqlen=T, vocab_size=cfg.vocab_size, policy="greedy")
    stepper = pipeline(None)
    rounds = max(1, args.val_tokens // args.batch_tokens)
    total_nll = 0.0
    total_valid = 0
    with torch.no_grad():
        for r in range(rounds):
            rnd = stepper.next_step().rounds[0]
            tok = torch.from_numpy(rnd.tokens)
            tgt = torch.from_numpy(rnd.targets)
            valid = int((tgt >= 0).sum())
            loss = model.loss(tok.cuda().view(B, T), tgt.cuda().view(B, T))
            total_nll += float(loss) * valid
            total_valid += valid
    val = total_nll / total_valid
    print(f"{args.run} @ step {payload.step}: val loss "
          f"{val:.4f}  (ppl {torch.tensor(val).exp():.2f}; "
          f"{total_valid} tokens)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
