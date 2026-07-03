# M5.2 handoff — qwen3.5-9B perf gap vs flextrain (2026-07-03)

Session handoff: correctness for the qwen3.5 family is DONE and committed;
the open task is a ~14–19% throughput gap vs flextrain that the evidence
places in per-task KERNEL costs, not scheduling. Read this top to bottom,
then start at "Next steps".

## Where M5.2 stands (all committed, suite 144 green)

- Full correctness ladder green for BOTH block kinds, both embedding
  modes: kernel contracts pinned, per-kind ladder 2, model-step vs golden
  (untied tiny + tiny_tied), plan-invariance ×3, batch=2 packed-sequence
  cu_seqlens reset E2E, 3-step multistep golden, poison-on-free,
  interleaving stress, measured-cost replan. `tests/tasks/test_qwen35_math.py`.
- **The 9B is UNTIED** (Shein's catch; config.json `tie_word_embeddings:
  false`; untied lowers to 8.95B params = the "9B", tied would be 7.94B).
  Untied = default (`ShapedQwen35Config`); tied stays a config choice
  (`tiny_tied()`, the 2B-style path) with its own E2E test. Commit 9fff09e.
- Fused-kernel pass done: `gated_rmsnorm_fwd/bwd` + `causal_conv1d_silu_fwd/bwd`
  registry families (fla fused Triton defaults, eager fallbacks,
  `DATAFLOW_KERNELS=eager` bisection verified). Fusion was an architecture
  win, perf-neutral at 9B scale (~0.3% of step).
- Key commits this stretch: 28eb89c (ladder 2 + contiguity contract),
  81b8ebf (family registry + ladder 3 + gates), ddcaa3c (fused gated
  rmsnorm), 89bb1f2 (conv registry), 9fff09e (untied correction),
  25d0902 (extending.md §6 qwen35 variants).

## Hard-won contracts (do not relearn these)

- **Every tensor handed to an fla/conv Triton kernel must be
  `.contiguous()`** — a strided column slice (e.g. packed `ba[:, HV:]`)
  is read with the wrong row stride and corrupts gate grads SILENTLY.
  This was the multi-day ladder-2 divergence; one line fixed it.
- fla `layer_norm_gated_bwd(recompute_output=True)` returns the PRE-GATE
  norm `rms(o)·w` — the silu gate must be composed on top before using it
  for the out-projection grad.
- The qwen35 packed layouts have alignment-padding gaps (8-byte
  A_log/dt_bias at tiny scale); AdamW updates padding from undefined dW
  padding → NaN in padding under poison, NEVER in fields. Engine-gate
  readbacks mask padding (`_run35`).
- Untied tiny at 8 MiB fast budget: known M4.9-class timing corner —
  interleaved-optimizer O_i prefetch strands the ledger, eviction valve
  ping-pongs to a loud DeadlockError (no far-future Belady candidate at
  tiny scale). Plan-invariance tight legs run at 12 MiB. 9B budgets: 0
  evictions everywhere.

## The perf gap (open task)

All numbers wall tok/s, RTX 5090, static placement, interleaved
optimizer, task0 preplace, 65,536 tok/step where noted.

| workload | ours | sim (measured costs) | flextrain (Shein recorded) |
|---|---:|---:|---:|
| qwen35-9b s1k bs8ga8 @16 GiB | 2,572 | 2,521 | ~3,160 |
| qwen35-9b s1k bs8ga8 @20 GiB | 2,717 | 2,651 | ~3,160 |
| (context) llama3-8b s1k @16/20 | 3,341/3,365 | ~3,4xx | 3,410–3,435 |
| (context) qwen3-8b s1k @16/20 | 3,221/3,289 | 3,438/3,487 | — |

Reading: real ≈ sim within +2–3% ⇒ the plan executes its measured costs
faithfully — **the measured per-task costs themselves are too slow**.
flextrain's hybrid runs only ~7% under its llama; ours runs ~19% under
our llama. Shein's read (agreed): kernel-level issue.

Secondary oddity worth a look while in there: replay_fidelity_gap_pct is
9–13% on this config (llama/qwen3 run ~0.5–2%) — the per-task replay gap
is unusually high even though the aggregate matches; consistent with a
few task families being mis-costed (profiled shape ≠ run shape behavior,
e.g. clocks or varlen autotune variance).

**Prime hypothesis — varlen vs dense kernel invocation.** Our lin blocks
flatten the batch to `(1, B·T)` + `cu_seqlens` for fla's
`chunk_gated_delta_rule_fwd/bwd` and `causal_conv1d_fwd/bwd`
(`_cu_seqlens(dims, device)` in `src/dataflow/tasks/qwen35_blocks.py`).
flextrain calls the same kernels DENSE `(B, T, H, D)`. Our lowering only
ever produces UNIFORM sequences (batch × seq_len), so a dense reshape is
EXACT — cu_seqlens is only needed for true ragged packing, which we never
generate. Varlen mode can cost real throughput (chunk_indices bookkeeping,
different autotune configs, reduced CTA parallelism on the recurrent
scan). Check the gattn flash path too (`ops.flash_fwd` — llama3 uses the
same helper and hits 3.3k, so it's probably already dense; verify).

Secondary suspects (likely minor): conv registry `out.copy_` +
`.contiguous()` staging copies (~134 MB × few per lin layer ≈ ms/step,
not percent-level); B=1 grid shapes for the fla scan kernels.

## Next steps (in order)

1. **Reproduce flextrain's number** so the target is measured, not
   recalled: conda env `flextrain` (exists), `refs/flextrain/train.py`
   (argparse from line ~374; `--steps`, `--device-id`, mode full vs lora,
   `SyntheticTokenSource` available; M4.6 note: `--leeway-gpu-mem-gib`
   default 5.0 explains flextrain's unused VRAM). Model at
   `refs/flextrain/models/Qwen3.5-9B/` (config.json present — check
   whether weights are downloaded or synthetic init is supported;
   `runs/Qwen3.5-9B_full_sl1024/` exists but is EMPTY, it was Shein's
   prior run label). Workload: full bf16 finetune, seq 1024, 64
   sequences/step = 65,536 tok/step, ~10+ steps, record steady tok/s.
   Do NOT run while a dataflow sweep is running (VRAM).
2. **Microbench A/B (the killer experiment)**: fla chunk fwd+bwd and
   conv fwd+bwd at `(1, 8192)+cu_seqlens` vs `(8, 1024)` dense, exact 9B
   shapes (HK=16, HV=32, K=V=128, conv_dim 8192, k=4), CUDA-event timed,
   both directions. If dense wins meaningfully → refactor
   `qwen35_blocks.py` to dense mode: reshape `(t, …)` → `(batch, seq, …)`
   at the fla/conv boundaries when `t == batch·seq_len` (always true for
   us); the conv window/recurrence reset then comes FREE from the batch
   dim (drop cu_seqlens entirely except for a future true-packing path).
   Keep the batch=2 E2E test — it pins the reset semantics either way.
3. **Per-task cost forensics** (if the A/B is not decisive): our profile
   cache stores per-task measured µs — compare linattn_fwd/bwd vs
   gattn_fwd/bwd vs llama3 block_fwd/bwd at same token counts; nsys the
   s1k run (`tools/nsys_profile.py`, DATAFLOW_NVTX=1, remember
   NSYS_NVTX_PROFILER_REGISTER_ONLY=0) and compare kernel names/durations
   against an nsys of the flextrain run — the diff names the slow kernel.
4. **After the fix**: re-run ladder (`pytest tests/tasks/test_qwen35_math.py`),
   full suite, re-profile + re-sweep s1k 16/20
   (`python tools/m4_train.py --config qwen35-9b-s1k-bs8ga8 --budgets
   16,20 --steps 3 --refresh-profiles --out artifacts/m5/qwen35-s1k-v2`),
   then promote `results/m5/qwen35-v1/` (README modeled on
   `results/m5/qwen3-v1/`), update design-doc §7 + auto-memory.

## Artifacts map

- `artifacts/m5/qwen35-untied-s1k/` — current honest s1k rows (the table
  above). `artifacts/m5/qwen35-untied/` — 4k/ga1 rows (1,379/1,375;
  transfer-floor-bound, not scoreboard material).
- `artifacts/m5/qwen35-{first,fused,s1k}/` — WRONG-ARCH (tied) runs,
  kept only as measurement history; never promote.
- Design/spec: `docs/notes/qwen35-design.md` (§7 = status checklist).
  Family how-to: `docs/extending.md` §6.

## Conventions reminders (for a fresh session)

- Commits in BOTH repos as `als244 <andrew.sheinberg@gmail.com>`
  (`git -c user.name=als244 -c user.email=andrew.sheinberg@gmail.com commit`).
- flextrain/prior_attempt are REFERENCES — concepts only, own code, own
  dependency pins.
- Plan docs are versioned files; never overwrite an old plan version.
- Env: `conda activate dataflow` (py3.12, torch 2.12.1+cu130,
  fla 0.5.1, causal-conv1d 1.6.2.post1 sm_120 source-build).
- Quote WALL tok/s in tables (`wall_tokens_per_s`).
- CUDA-graph-per-task is explicitly PAUSED by Shein; don't resurrect it
  as a fix for this gap without asking.
