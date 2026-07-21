#!/usr/bin/env python
"""Materialize a HuggingFace dataset to the local filesystem so the
ordinary local sources serve it (hub datasets are NOT a special
runtime source — they become persistent local files, naturally
replayable).

    python tools/train/fetch_dataset.py openai/gsm8k \
        --config main --split train
    python tools/train/fetch_dataset.py HuggingFaceFW/fineweb \
        --config sample-10BT --field text

Writes datasets/<name>/<split>.jsonl (one JSON object per record).
Records always carry a "text" field — for prompt/response-shaped
datasets (question/answer, instruction/output, chat messages) the
text is the joined form AND the normalized prompt/response pair is
kept alongside (fields "instruction"/"output") for future
prompt-masked training. Plain-text datasets pass their text column
through.

Idempotent: an existing target is left alone unless --force. The
resulting file trains via:

    --data jsonl:datasets/<name>/<split>.jsonl,tokenizer=gpt2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

PAIR_CANDIDATES = (
    ("instruction", "output", "input"),
    ("prompt", "completion", "input"),
    ("prompt", "response", "input"),
    ("question", "answer", "context"),
    ("query", "response", "context"),
)


def normalize_record(rec, text_field: str | None):
    """One HF record -> a JSONL row with a guaranteed "text" field."""
    if text_field:
        text = str(rec.get(text_field, "") or "").strip()
        return {"text": text} if text else None
    for prompt_key, response_key, input_key in PAIR_CANDIDATES:
        prompt = str(rec.get(prompt_key, "") or "").strip()
        response = str(rec.get(response_key, "") or "").strip()
        if prompt and response:
            extra = str(rec.get(input_key, "") or "").strip()
            joined = prompt + ("\n" + extra if extra else "") \
                + "\n\n" + response
            return {"text": joined, "instruction": prompt,
                    "output": response}
    if isinstance(rec.get("text"), str) and rec["text"].strip():
        return {"text": rec["text"].strip()}
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="hub id, e.g. openai/gsm8k")
    ap.add_argument("--config", default=None,
                    help="dataset config name (gsm8k: main)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--field", default=None,
                    help="pass this text column through verbatim "
                         "(default: normalize prompt/response shapes)")
    ap.add_argument("--target", default=None,
                    help="output path (default: datasets/<name>/<split>.jsonl)")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N records")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    name = args.dataset.rsplit("/", 1)[-1]
    target = Path(args.target) if args.target \
        else _ROOT / "datasets" / name / f"{args.split}.jsonl"
    if target.exists() and not args.force:
        print(f"[fetch] {target} exists — skipping (--force to redo)")
        return 0
    try:
        from datasets import load_dataset
    except ImportError:
        print("error: the `datasets` package is missing — install the "
              "data extra: pip install -e '.[data]'", file=sys.stderr)
        return 2

    ds = load_dataset(args.dataset, args.config, split=args.split,
                      revision=args.revision)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    kept = 0
    dropped = 0
    with open(tmp, "w") as fh:
        for i, rec in enumerate(ds):
            if args.limit is not None and kept >= args.limit:
                break
            row = normalize_record(rec, args.field)
            if row is None:
                dropped += 1
                continue
            fh.write(json.dumps(row) + "\n")
            kept += 1
    tmp.replace(target)
    print(f"[fetch] {args.dataset} ({args.config or 'default'}/"
          f"{args.split}) -> {target}  records {kept} "
          f"(dropped {dropped})")
    print(f"[fetch] train with: --data jsonl:{target.relative_to(_ROOT)}"
          f",tokenizer=gpt2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
