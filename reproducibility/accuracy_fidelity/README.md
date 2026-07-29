# accuracy_fidelity — loss-curve invariance across device budgets

Does training the SAME model, recipe, and data stream through the engine
at different device budgets produce the SAME loss curve — and does that
curve match the family's independent pure-torch twin (`reference_models`)?
The throughput experiment (`../throughput_fidelity/`) certifies that plans
*cost* what the simulator says; this one certifies that plan choice —
save-everything, recompute-boundary, deep-offload — never changes the
*math*.

## Quickstart

```bash
python reproducibility/accuracy_fidelity/run_accuracy.py \
    --preset gpt2_124m --batch 64 --ga-rounds 8 \
    --steps 10000 --peak-lr 6e-4 --auto-budgets
```

Interrupted? Re-run the SAME command with `--resume` appended. Finished
legs are kept (a curve with all its losses is skipped), the interrupted
leg restarts from its newest checkpoint (a little redundancy back to that
checkpoint is expected), and metrics/logs APPEND rather than restart.
Everything the campaign produces lives under
`reproducibility/accuracy_fidelity/results_accuracy/` (move it with
`--results`).

## Stages

| stage | what | cost |
|---|---|---|
| `plan` | SCALE-FREE regime selection on PROFILED task costs (roofline only as the labeled GPU-less fallback). One unbounded plan learns the model's save-everything peak; a log-spaced ladder of SHARES of that peak (12%…105%) is probed, each rung sweeping `t_round` candidates and keeping the frontier (best predicted tok/s). `--auto-budgets` then picks three regimes — deep (smallest feasible: heaviest recompute+offload), boundary (largest rung still recomputing), comfortable (smallest zero-recompute) — recorded with their `t_round` in `plan.json`; a resumed run reuses the recorded choice. The same rule sizes a 124M and an 8B — no per-family constants | seconds-to-a-minute |
| `engine` | one training leg per (budget, t_round) pair, sequential, FIRST — the budget-invariance evidence banks early; keeps every checkpoint | GPU |
| `reference` | the pure-torch twin at the identical recipe — the anchor curve (keeps every checkpoint) | GPU, the long pole (~2-3x an engine leg) |
| `report` | drift verdicts (reference-vs-engine + engine-vs-engine pairwise: step0 / max / final / EMA deltas, strict-parity AND EMA-band verdicts) plus a HELD-OUT VAL LADDER — val loss at every retained checkpoint of every leg (~30-60s each, resumable via `metrics/val_*.jsonl`) → `REPORT.md`. Re-runnable standalone any time: `--stages report --resume` | minutes |

Every leg is a `tools/train/train.py` subprocess — byte-identical to a
hand-launched run of the shipped tool.

## Invariance laws (enforced, because each was a real hand-run footgun)

- **One schedule horizon**: warmup and the cosine tail derive from
  `--steps` (steps/10, peak/10). Comparing a 1k-step run against a
  10k-horizon baseline requires *launching at 10k* — the driver always
  passes the full horizon to every leg.
- **One data geometry**: `tokens/step = batch x seq x ga` is part of the
  math. `--batch`/`--ga-rounds` go to every leg explicitly, the reference
  included; preset defaults are not trusted (the driver says so loudly if
  you leave them unset).
- **One seed, one data spec**: byte-identical init, one deterministic
  doc-aware fineweb stream by default.

## Layout — everything a report needs, saved as it is produced

```
results_accuracy/
  manifest.json                THE structured index: campaign config,
                               chosen (budget, t_round) pairs, per-leg
                               status + wall time + peak memory + file
                               pointers; rewritten atomically at every
                               stage boundary
  plan.json                    full ladder table + the chosen regimes
  curves/<preset>_<opt>_reference.json      train.py run-curve JSON —
  curves/<preset>_<opt>_b<budget>.json      the AUTHORITATIVE losses
                               (full precision; *_partial.json refreshed
                               at every checkpoint mid-run)
  metrics/<leg>.jsonl          streamed per step line while the leg runs
                               (appends on resume); schema.json beside it
  metrics/val_<leg>.jsonl      the val ladder as it is evaluated
  logs/<leg>.log               each leg's full stdout
  REPORT.md                    verdict + val tables (regenerate any time:
                               --stages report --resume)
```

The driver narrates progress every ~2 min per leg (`step N/M, s/step,
peak mem, eta`) with the ETA refined from the measured step rate.

## Checkpoints

Legs checkpoint sparsely (`--ckpt-every`, default 1000; 0 disables) so
`--resume` can restart them. They land under the train tool's own
`results/pretrain/checkpoints/<leg>/`; on disk-tight hosts symlink that
directory to cold storage, e.g.

```bash
ln -s /path/to/cold/model_ckpts results/pretrain/checkpoints
```

## Campaign order

1. `gpt2_124m` comprehensively to 10k steps — adamw, seq 2048,
   t_round 16K, t_step 64K (batch 8 x ga 4), varlen default packing;
   reference + three sim-chosen budgets,
2. `llama3` (scope decided from the gpt2 results),
3. `qwen3moe`.

Geometry stays FIXED across a campaign's legs — round capacity shapes
how the packer partitions the stream, so varying it across budgets
would reintroduce the confound under test. A budget that cannot fit
the campaign geometry is excluded loudly, never silently re-shaped.
