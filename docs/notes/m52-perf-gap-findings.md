# M5.2 findings — the qwen3.5-9B "perf gap" is plan shape, not kernels

Investigation of the handoff question (docs/notes/m52-perf-gap-handoff.md):
why does our qwen3.5-9B s1k run 14-19% under flextrain? Answer, in one
line: **it doesn't — the honest gap is ~9%, and it is PCIe-boundedness of
the save-all plan, not kernel cost.** Kernels are at roofline; the varlen
invocation is free; the fix is enabling the recompute planner, which the
s1k sweep never turned on.

All measurements 2026-07-03, RTX 5090, torch 2.12.1+cu130, fla 0.5.1.

## 1. The target was mis-recalled: flextrain = 2,981 tok/s, not ~3,160

Reproduced flextrain on the EXACT workload we sweep (seq 1024, synthetic,
65,536 tok/step, full bf16, 12 steps, auto memory = 24.7 GiB peak):

    overall tok/s: 2,981   (steady steps: 2,967-3,013)
    eff TFLOPS 143.5, hw TFLOPS 156.8, recompute frac 0.318
    log: artifacts/m5/flextrain-qwen35-repro/train_sl1024.log

The recalled "~3,160" (and verified_runs.md's 3,219) is a DIFFERENT
workload: --max-seq-len 2048 on ragged SFT data. Comparisons:

| | ours @16 | ours @20 | flextrain (24.7 GiB) |
|---|---:|---:|---:|
| wall tok/s | 2,572 | 2,717 | 2,981 |
| gap | -13.7% | **-8.8%** | — |

(ours = artifacts/m5/qwen35-untied-s1k, LEDGER-quoted budgets; the @20
row's placed extent is 23.9 GiB, so with torch scratch + CUDA context
its device usage runs ~27 GiB — a bit ABOVE flextrain's 24.7. The
strictly device-matched rows are in §7; -8.8% is the right order of
magnitude but ledger rows flatter us slightly.)

Context: flextrain's own hybrid tax on this machine is 3,410-3,435 (llama)
→ 2,981 (qwen3.5) = -13%, not the -7% the handoff recalled (that too was
the sl2048 workload).

## 2. Kernel hypotheses eliminated (tools/bench_qwen35_kernels.py)

fla chunk_gated_delta_rule + causal_conv1d fwd+bwd at exact 9B shapes
(HK16/HV32/K=V=128, conv 8192×4, mirrored invocations incl. contiguity):

| launch | fla_fwd | fla_bwd | conv_fwd | conv_bwd | total ns/tok |
|---|---:|---:|---:|---:|---:|
| dense (8,1024) | 0.770ms | 2.011ms | 0.210ms | 0.578ms | 435.7 |
| varlen (1,8192)+cu [ours] | 0.768ms | 1.999ms | 0.213ms | 0.580ms | 434.6 |
| dense (32,1024) | 3.121ms | 8.425ms | 0.735ms | 2.187ms | 441.6 |
| varlen (1,32768)+cu [flextrain] | 3.057ms | 8.728ms | 0.738ms | 2.182ms | 448.8 |
| varlen (1,65536)+cu | 6.885ms | 18.589ms | 1.438ms | 4.340ms | 476.9 |

- **varlen vs dense: tie** (0.2%). The handoff's prime hypothesis is dead —
  and flextrain's linear_attn.py in fact uses the SAME varlen invocation
  (unsqueeze(0) + cu_seqlens), packing 32,768-token chunks.
- **larger launches are mildly WORSE per token** (x0.97 at 32k, x0.91 at
  64k) — flextrain's big chunks are not a kernel advantage either.
- Scale: the whole DeltaNet+conv stack is 434.6 ns/tok × 24 layers ≈
  0.68 s of a ~24 s step (~3%). It never could have carried a 9% gap.

## 3. Where the step actually goes: the plan is h2d-bound

Per-family totals from the @20 annotated plan (measured per-task costs,
artifacts/m5/qwen35-untied-s1k/...-20gib.annotated.json):

| family | n | total | mean |
|---|---:|---:|---:|
| linattn_bwd | 192 | 8.07 s | 42.0 ms |
| linattn_fwd | 192 | 3.78 s | 19.7 ms |
| gattn_bwd | 64 | 2.56 s | 40.0 ms |
| head_bwd | 8 | 1.19 s | 148.9 ms |
| gattn_fwd | 64 | 1.18 s | 18.5 ms |
| head_fwd | 8 | 0.59 s | 73.3 ms |
| loss/opt/embed | — | 0.15 s | — |
| **total compute** | | **17.52 s** | |

Sim step = 24.72 s, real = 24.06 s. **Compute is 17.5 s; the other ~7 s
is exposed transfer time.** The plan's own transfer ledger (sum of
prefetch/offload directives × object sizes):

| stream | bytes/step | at measured bw | |
|---|---:|---:|---|
| h2d | **456.5 GiB** | **17.9 s @ 25.5 GB/s** | params 191.4 + activations 181.4 + grads 50.4 + opt 33.4 |
| d2h | 281.9 GiB | 10.9 s @ 25.8 GB/s | activations 181.4 + grads 50.4 + opt 33.4 + params 16.7 |

h2d work (17.9 s) ≥ compute (17.5 s): the step is **h2d-critical**. The
save-all plan ships every activation to pinned host and back (181 GiB
each way), re-streams the full weight set every one of the 8 rounds
(191 GiB), and round-trips dW across ga rounds (50 GiB). flextrain's
plan shape needs roughly ~95 GiB h2d (2 rounds × 18 GiB weights +
24.6 GiB activations saved with 32% recomputed + ~36 GiB bf16 opt state
— the same bf16-moments convention as ours, and their adamw also has a
host-side path) — comfortably hidden under its 22 s of compute. Their compute is SLOWER
than ours (22 s incl. recompute vs our 17.5 s); their plan just never
waits on the bus.

This also explains the odd replay_fidelity_gap_pct (9-13% vs llama's
0.5-2%): per-task real durations are measured amid a saturated bidi PCIe
bus, while profiled costs are uncontended (M4.3 taxonomy: ±4-5%,
amplified here because the bus is busy nearly the whole step). Aggregate
real still ≈ sim because in h2d-critical windows, slower compute doesn't
move the makespan.

## 4. Why did our planner allow this? It was never asked

`plan_with_recompute` is simulator-verified and seeds with recompute-ALL
(it would have seen the ~4-5 s makespan win) — but `tools/m4_train.py`
gates it behind `--recompute`, and the s1k sweep commands (handoff §Next
steps 4, artifacts summary `recompute_chosen: 0` on both budgets) never
passed the flag. The per-rewrite exchange rate is lopsided: a lin layer
round saves 775 MB (≈30 ms d2h + 30 ms h2d at measured bw) for 24.3 ms
of recompute (13.1 ms truncated) — recompute wins even before counting
bus contention.

## 5. Experiment: same config, --recompute on — planner engages, reality doesn't move

artifacts/m5/qwen35-s1k-rc, @20 GiB ledger:

| | save-all (baseline) | --recompute (128/256 chosen) |
|---|---:|---:|
| sim ms/step | 24,718 | **21,282** (sim: +13.9%) |
| real ms/step | 24,062 | **24,206** (real: ±0) |
| wall tok/s | 2,717 | 2,701 |
| real vs sim | +2.7% | **-12.1%** |
| replay_fidelity_gap_pct | 9.1% | 3.3% |
| pinned host | 88.6 GiB | 74.2 GiB |

Decomposing the -12.1% with the replay machinery (replay_gap_pct re-sims
the plan with MEASURED task durations): resim(measured) ≈ 23.4 s vs
sim(profiled) 21.3 s vs real 24.2 s. So ~2.2 s of the optimism is
**per-task duration inflation** — compute tasks run ~8-12% slower in-run
than their uncontended profiles while the bus is busy — and only ~0.8 s
is scheduling/transfer mismatch. The recompute plan didn't remove the
contention tax; it converted exposed-transfer time into
contended-compute time. Net zero. (M4.3 measured this contention at
±4-5% for llama-8B's traffic volume; qwen35's ~740 GiB/step total
traffic roughly doubles the exposure. The fidelity gap improving 9→3%
while the makespan stayed flat is the same story from the other side:
less bus saturation during the compute-heavy stretches.)

Conclusion so far: recompute alone rebalances WHICH resource is
critical, but the win the sim promises is eaten by contention it does
not price. To actually reach flextrain's number the plan must cut TOTAL
traffic volume — and the dominant term is the 191 GiB/step of parameter
re-streaming, which scales with round count (8 rounds × full weight
set). flextrain runs the same tokens in 2 rounds.

## 5b. Experiment: fewer/larger rounds + recompute — +7% real, config-only

bs16ga4 (4 rounds of 16,384 tokens; halves the weight re-stream) at the
same 20 GiB ledger, artifacts/m5/qwen35-s1k-rounds:

| bs16ga4 @20 | sim tok/s | real wall | fidelity gap |
|---|---:|---:|---:|
| --recompute, planner picks 64/128 | 3,271 | **2,908** | 2.1% |
| --force-recompute all (128/128) | 3,018 | 2,797 | 0.5% |
| (bs8ga8 save-all baseline) | 2,521 | 2,717 | 12.7% |

Three load-bearing observations:

1. **bs16ga4 + planner recompute = 2,908 wall (+7.0% over baseline,
   -2.5% from flextrain's 2,981) with zero model-code changes.**
2. The planner's RANKING is right even though its absolute numbers are
   optimistic: it predicted rc-64 > rc-all (3,271 vs 3,018) and reality
   agreed (2,908 vs 2,797). The greedy simulator-verified loop works
   within its model; the model just doesn't price contention.
3. The fidelity gap is monotone in traffic volume — save-all 12.7%,
   rc-64 2.1%, rc-all 0.5% — direct confirmation that per-task
   real-vs-profiled inflation IS bus contention, not mis-profiling.

bs32ga2 (2 rounds, flextrain's shape) is blocked structurally: the
monolithic head task materializes 32,768 x 248,320 bf16 logits =
15.2 GiB in ONE buffer (profiling OOMs the card; placement would refuse
too). flextrain necessarily chunks its head/CE. A token-chunked
head+loss lowering would unlock bs32 and shrink the logits object
16x — candidate M5 item, NOT required for the current scoreboard.

Bycatch: bs16 profiling exposed an int32 overflow in the fused CE
kernel (row x 248,320 vocab crosses 2^31 at row 8,650; bs8 sat 5%
under the line). Fixed + regression-tested (commit 42cdadc).

## 6. Where this lands, and the remaining levers

**Applied (config-only, no model-code changes): qwen35 s1k sweeps run
bs16ga4 with --recompute.** 2,717 → 2,908 wall (+7.0%), -2.5% from
flextrain at comparable memory. Per Shein's bar (2026-07-03: "if the
gap isn't substantial and the fix needs refactoring, keep things as
they are — but know why"), the WHY is fully attributed and the cheap
fix is taken; the remaining levers are recorded, not built:

1. **Contention-aware plan costing** (sim-side). The compute sum is
   17.5 s → a 3,745 tok/s ceiling if transfers were fully hidden; the
   sim already predicts 3,271 for the current best plan and reality
   pays ~11% contention tax on it. The planner's ranking is correct —
   absolute calibration (e.g. profile under --contend as the
   pessimistic bracket, or a contention coefficient on windows where
   transfer streams run hot) would let it optimize INTO the contention
   regime instead of stopping at its blind spot. This is a real
   planner-objective design item — flag for Shein before building.
2. **Token-chunked head+loss lowering — DONE (a30681a, 2026-07-04)**:
   head_fwd/loss_bwd/head_bwd fused into one token-chunked head_loss
   task (flextrain's memory contract: no (t, vocab) tensor ever).
   Results (artifacts/m5/head-loss, ledger-quoted):
   bs32ga2 @14 = **3,001 wall tok/s — new qwen35 record, BEATS
   flextrain's 2,981** at its own 2-round shape (fidelity 0.34%: the
   2-round weight streaming is tiny, as §5b predicted); bs32ga2 @16 =
   2,965; bs16ga4 @20 = 2,935 (was 2,897-2,908 pre-fusion). Honest
   caveat: bs32's torch block-bwd scratch is ~4x bs8's (the ledger-20
   attempt OOMed the card; 14-16 GiB ledgers fit), so DEVICE usage is
   ~28-29 GiB vs flextrain's 24.7 — the del-at-last-use scratch fix
   (M4.6 queue) is what converts this into a device-terms win too.
3. **Window-oracle re-run for qwen35** (tools/window_plans.py, minutes):
   the "weight residency rejected / B0 optimal" verdict was computed on
   llama's traffic profile; qwen35's h2d-critical profile may flip it.
4. M5 items unchanged (VMM extent tax ×1.17 here, CUDA-graph host tax)
   — real, but they are NOT this gap.

## 7. Headline rows — and an envelope-machinery caveat the runs exposed

artifacts/m5/qwen35-s1k-dev (bs16ga4, --recompute, --device-gib 25/29):
**both legs violated their envelopes** — actual peaks 27.15 (>25) and
29.92 (>29) GiB, caught by the post-run verifier. Cause: the scratch
reserve is max-task workspace from ISOLATED profiling (+256 MiB pad),
but the Session's per-stream torch scratch caches ADD across streams —
at bs16-with-recompute the real aggregate is ~7 GiB vs the 5.0 reserved.
This is the documented "serial-scratch assumption" caveat of
--device-gib biting for real; until the reserve is stream-aware, the
device-quoted rows below are labeled by their MEASURED actual peaks:

| plan (all bs16ga4 + planner recompute 64/128) | wall tok/s | actual device peak |
|---|---:|---:|
| ledger 18.45 (the "dev-25" leg) | 2,782 | 27.15 GiB |
| ledger 19.72 (the "dev-29" leg) | 2,865 | 29.92 GiB |
| ledger 20.00 (§5b row; peak not instrumented) | 2,908 | ~30 GiB (est.) |
| flextrain (measured, same workload) | 2,981 | 24.7 GiB |

Honest summary: **at flextrain-matched memory (~27 GiB actual, still
2.4 GiB above them) we are -6.7%; at best-on-card we are -2.5%.** The
memory side is itself the next lever: our ~7 GiB of multi-stream torch
scratch dwarfs flextrain's 5 GiB leeway that absorbs the same class of
allocations, and the M4.6-queued del-at-last-use halving of block_bwd
intermediates plus VMM chunk-backing (extent tax ×1.08-1.17 here) both
convert directly into feasible-ledger headroom at fixed device budget.

Follow-ups recorded, not built (in addition to §6): make the
--device-gib scratch reserve stream/plan-aware so envelope quotes hold
under recompute plans.
