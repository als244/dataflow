# Deep correctness-compare treatment

How we establish that the engine trains a model family correctly, by
comparison against that family's isolated pure-torch reference twin
(`reference_models/`). This document is the playbook for giving ANY
family — existing or future — the full treatment, and the catalog of
gotchas that make naive comparisons either vacuous or falsely alarming.
Every gotcha below was hit for real; the probes cited found them.

## Philosophy

The twin is an INDEPENDENT implementation (fp32 SDPA/norm paths, plain
autograd) loaded with the engine's exact init bytes. Independence is the
point: shared kernels would hide shared bugs. The price is that the two
legs differ by a legitimate, systematic cross-implementation distance —
so every gate must know what "healthy disagreement" looks like, or it
will either alarm on physics or get widened until it covers real bugs.
Never widen a tolerance without a root-caused minimal example of the
mechanism it prices in.

## The instrument ladder

Run in this order; each level sharpens the previous one.

1. **Loss-curve parity** (multi-step, calibrated bands): catches gross
   wiring/semantics errors. Blunt — passes do not certify much.
2. **Ladder-3 per-field** (`check_model_step`): one real engine step vs
   one twin step from identical bytes; compares loss, EVERY final
   param/buffer (rel_l2 + cosine), and EVERY gradient in dW space
   (`grad:{name}` entries). dW space is the sharp instrument — see
   gotchas 1–2 for why params and updates are not.
3. **Deep compare** (`tools/deep_compare.py`): forward divergence by
   depth with per-token hot-row decomposition; per-block gradient
   medians; MoE counts parity. Separates smooth precision drift from
   discrete relocations and localizes anomalies to a block.
4. **Per-block isolation** (`deep_compare.py --isolate N[,M...]`): feed
   the ENGINE's own block-(N-1) output into the twin's block N and
   compare outputs. Removes all upstream divergence — certifies a
   block's math even when the model-level compare is saturated (gotcha
   8). Isolate leader+follower JOINTLY for blocks that share discrete
   selections (gotcha 9).
5. **Control pair**: run the engine leg twice in separate processes and
   diff. Single-step dW is BITWISE reproducible (measured), so the
   control costs nothing and proves any engine-vs-twin gap is
   systematic, never "noise".
6. **Margin/counts probes**: routing-margin distributions and engine
   Aux counts vs twin step counters — turn "the MoE fields disagree"
   into "token 82 flipped experts 6↔7 at margin 3.7e-4".

## The gotcha catalog

**1. Init dominance (~200×).** Final params = init + one-step update,
and |init| / |update| ≈ 0.02 / 1e-4. Param-space rel_l2 divides the
real disagreement by ~200, and param-space cosine is init-vs-init.
A harness that fed the twin WRONG attention boundaries passed every
param comparison; only a zero-init field caught it. Corollary:
zero-init fields (balance biases) are the sharpest tripwires — expect
them to fail first, and investigate rather than envelope.

**2. bf16 weight-quantum contamination.** Comparing updates
(final − init) at training lr puts the update at ~1 ulp of the stored
bf16 weight; both legs collapse to rounding lottery (measured:
nonzero-but-orthogonal router updates, cos ≈ 0). Small-gradient fields
stay sub-quantum at ANY usable lr. Compare gradients (dW), not stored
deltas. To read dW after a run, add the dW ids to
`Program.final_locations` — objects omitted there are recycled after
last use (`initial_buffers` does NOT intercept task outputs).

**3. Engine determinism.** Single-step dW is bitwise identical across
processes. There is no ambient noise to hide behind: any engine-vs-twin
distance is systematic and must decompose into named mechanisms
(precision paths + discrete flips). If it does not, keep digging.

**4. Near-tie flips at discrete choosers.** Any top-k over scores
(MoE routing, group-limited selection, DSA index selection) flips
tokens whose selection margin is below the cross-implementation
activation distance (~1e-3). One flipped token relocates its whole
contribution: hot forward rows (row-rel ~0.3), whole-token grad
relocation in expert stacks, ±1 count deltas, and a bias sign flip when
a count sits within 1 of the exact integer mean. Irreducible between
independent implementations. Gates must BUDGET flips — counts totals
exact (tokens × top_k), per-expert |Δ| ≤ budget, hot-row-aware
continuous comparisons — never blanket-envelope the affected fields.

**5. Exact ties are a convention, not noise.** A margin of exactly 0.0
(bit-equal scores, common in bf16 at random init) is decided by
tie-break convention. Twins must PIN the engine's convention
(smallest-index via stable descending sort — see the dsv3 twin's
`_route`); a plain `torch.topk` is unpinned and flips exact ties
gratuitously.

**6. Flip propagation depends on the mixer.**
- Softmax attention: a flipped token contaminates other queries only
  through bounded attention weights — flips stay local (dsv3: exclude
  2 hot rows and the floor returns).
- Recurrent state (deltanet/SSM hybrids): a flipped token enters the
  running state and contaminates EVERY subsequent token — hot sets grow
  by depth and medians inflate model-wide (qwen35moe). The model-level
  matrix number is dominated by this amplification, not by per-op
  error.
- Elevated grad medians in EARLY blocks can be inherited from LATE
  forward divergence through the backward chain — localize with
  isolation, not by reading the grad table alone.

**7. CE/softmax amplification at random init.** Near-uniform logits
make per-class probabilities tiny; ~1e-3 logit divergence becomes
~1e-2 relative divergence in dlogits and head/embed grads. The FIRST
gradient of backprop already carries it — this is expected, not a bug.

**8. Flip-density saturation.** Many discrete choosers × many layers ×
tiny-config margins → flip cascades (glm52: ~35/128 tokens hot by
block 5; even hot-row exclusion stops recovering the floor because
every query attends over several flipped keys). Tiny smoke configs
LOSE model-level verification power for such families. Certification
then comes from per-block isolation (which stays at the few-e-3 floor
if the math is right) plus counts/margin probes.

**9. Shared-selection followers.** Blocks that reuse another block's
discrete selections (DSA followers reusing the leader's index top-k)
show phantom broadband divergence when isolated SOLO — the twin's
leader computed selections from twin activations. Isolate the leader
together with the follower (`--isolate 4,5`); the phantom vanishes if
the math is clean (measured: 2.0e-2 → 4.3e-3).

**10. The aux/LBL channel needs explicit driving.** Twin `loss()`
defaults to pure CE; if the harness never passes the live `aux_coef`
while the engine lowers with it, the entire load-balance gradient
channel goes untested (and hides inside gotcha-1 blindness). Rules:
- The loss CHANNEL stays CE-only on both legs; aux enters gradients.
- Sequence-wise LBL (sigmoid/noaux trio) decomposes over per-sequence
  forwards: add `aux_coef * load_balance_loss()` UNSCALED per sequence
  (the engine applies full alpha per round; never scale aux by the CE
  valid ratio).
- Round-global LBL (softmax default) does NOT decompose per sequence;
  until the twin can express it jointly, zero aux on BOTH legs
  (symmetric) and say so — a silently mismatched channel is worse.
- The ga-invariant retained mode drops non-router aux gradients BY
  DESIGN; its twin replica must compute the aux term from
  `router(x.detach())` so autograd matches that truncation.
- Twins declare their form via `AUX_FORM` ("sequence_wise" /
  "forward_global"); the harness dispatches on it.

**11. Twins are varlen-NATIVE — and new ones must be too.** Every twin
accepts a packed round: `forward/loss(tokens (1, ΣL), seq_lens=...)`
builds per-sequence rope positions and a block-diagonal additive mask
(masked SDPA; note torch's flash kernel rejects `attn_mask`, so packed
mode runs the efficient/math backend — twins trade speed for clarity).
Recurrent mixers (DeltaNet conv + delta-rule state) restart per segment
by slicing — exact, since pad and state are zero-initialized. The
sequence-wise aux and the DSA KL live-sets follow the segment bounds /
mask automatically. Feeding a packed row WITHOUT `seq_lens` computes
boundary-crossing attention — the original harness bug that gotcha-1
hid and the zero-init bias exposed. When porting or writing a twin,
gate the packed path with the CPU fp32 equivalence check: packed
outputs vs per-sequence loop must agree to ~1e-6 (logits AND every
auxiliary term — seq-aux, round-global LBL, indexer KL).

**12. Optimizer-time mechanisms are not autograd.** The noaux balance
bias updates by sign(mean−c) on STEP-AGGREGATE counts at
optimizer time; its dW slot carries counts, not gradients
(policy-frozen, absent from grad layouts). The twin accumulates
`step_counts` across the round's forwards and applies the identical
rule once per step. Integer counts at an exact integer mean produce
sign(0)=0 — both sides must share tie semantics.

**13. Frozen/warm-up phases.** Configs that freeze fields (dense
warm-up) need the twin restricted to the trainable set
(`reference_train_only`), or the twin updates params the engine holds
fixed.

**14. Grad-layout assumptions.** The generic engine-grad extraction
fabricates weight-layout buffers filled with grad values so the family
bridges' name maps apply unchanged; it asserts grad dtype == param
dtype per field and NaN-poisons gradient-free fields. A family with
split param/grad dtypes will trip the assert loudly — extend the shim
then, never silently.

**15. Every training objective must be driven, not just CE.** The
aux/LBL channel (gotcha 10) is one instance of a general rule: any
gradient source the engine wires (LBL, the DSA indexer KL, future
auxiliary objectives) needs an explicit twin replica with the engine's
exact detach semantics, driven by the harness, or the affected fields
go dark — the gradient comparison skips fields the twin has no grad
for, and final-param comparisons hide the movement (gotchas 1, 16).
The DSA indexer KL was dark this way: engine `train_indexer` defaults
True while the twins omitted the objective. Twin replica rules for it:
target = head-summed live-set attention probs, L1-renormalized,
DETACHED; indexer input DETACHED (gradient reaches only the indexer
weights); follower targets aggregate into their LEADER's objective as
(1/N)·Σ_members KL(p_member ‖ σ_leader). When onboarding a family,
enumerate every dW the engine produces and demand a twin gradient for
each — a skipped name in the grad comparison is a finding, not a
convenience.

**16. Zero-denominator rel_l2 masks moved-vs-zero.** rel_l2 with an
all-zero reference used to return the ABSOLUTE norm of the other side —
so an engine-trained field against a zero twin read as ~lr ≈ 1e-4,
"passing". It now returns inf (loud) when exactly one side is zero;
zero-vs-zero stays 0. Cosine already treated one-sided zero as 0.0 —
keep both, they fail differently.

**17. Compare on the box you think you're comparing on.** If gates run
on a remote overlay, the sync set must include EVERYTHING the harness
imports (`reference_models/` lives at the repo root, outside `src/`).
A stale twin produces failures that look exactly like engine bugs.

## Onboarding a new family

1. Twin in `reference_models/`: self-contained, paper semantics,
   per-row causal AND varlen-native (packed `seq_lens` mode with the
   equivalence check of gotcha 11); pin tie-break conventions at every
   discrete chooser;
   expose `step_counts`-style counters for every counted selection;
   declare `AUX_FORM`; stash per-forward aux terms.
2. Bridge in `pretrain/bridges/`: `to_reference_state_dict` + strict
   load + byte-identity assert. The grad extraction reuses it as-is.
3. Gates: family rows in the parity suite (uniform + ragged), per-field
   ladder entries, counts parity if MoE, freeze allowlists if phased.
4. Run the deep treatment: full matrix row (`tools/sweep_ladder3.py`),
   then `deep_compare.py` both shapes, then `--isolate` any block whose
   kind is new. Compare against the reference table below; every excess
   over the floor must decompose into cataloged mechanisms (flips you
   can count, propagation you can name) before tolerances are set.
5. Calibrate bands from repeated runs (engine side is bitwise; twin
   side varies only with intended config changes), at most 2× the
   worst observed, each with its mechanism named.

## Reference healthy numbers (tiny smoke configs, sm86)

Gradient (dW-space) medians / worst / min cosine, uniform ≈ ragged:

| tier | families | grad median | worst field | min cos |
|---|---|---|---|---|
| dense + attention | llama3, qwen3 | 0.7–1.2e-2 | ~1.7e-2 | ≥0.9998 |
| hybrid dense | qwen35 | ~1.6e-2 | ~5e-2 (state params) | ≥0.9990 |
| MoE, local flips | qwen3moe, olmoe, dsv3, dsv32 | 1–10e-2 | 0.05–0.22 (router/experts) | ≥0.976 |
| flip-amplified | glm52 (cascade), qwen35moe (recurrent state) | 0.1–0.17 | 0.28–0.39 | ≥0.937 |

Forward per-block floor (hot rows excluded): 3–9e-3 growing smoothly
with depth; isolated-block floor 3–7e-3 with hot rows only at
countable tie margins. Loss rel ≤ ~1e-3 (flip-amplified) and ≤ ~1e-4
otherwise. Numbers above these tiers demand investigation, not wider
bands; numbers within them still require the flip counts to reconcile.
