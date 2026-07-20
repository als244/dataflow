# tools/verify/ — correctness gates

Instruments that pin the runtime's math and plans against references.

## verify_family.py — one-command family correctness

Runs the family's canonical test module (per-op pins, per-task
fwd/recompute/bwd ladders vs the reference, per-model step vs the
isolated twin) and audits it for the ladder canon
([extending.md](../../docs/extending.md) §8).

| flag | meaning |
|---|---|
| `--family` | family name (`--list` enumerates) |
| `--module` | path to the canonical test module (external families) |
| `--audit-only` | coverage audit without running the tests |
| `--plugin` | external family plugin module(s) |

## engine_gate.py — real GPU vs simulator prediction

Synthetic-task execution on the real engine compared against the
sim's makespan/overlap expectations; writes
`<stem>.{summary,trace}.json`.

| flag | meaning |
|---|---|
| `--config {mini,small,8b}` | synthetic chain shape |
| `--fast-gib` | device budget (required) |
| `--recompute` | let the planner choose recompute |
| `--completion-mode {poll,hostfn}` | task-completion mechanism under test |
| `--bw-mode {bidi,uni}` | price transfers at bidirectional (concurrent-load) or unidirectional PCIe rates |
| `--out` | output directory |

## pressure_correctness.py — math invariance under memory pressure

The full annotated program through the REAL engine at descending
tight budgets; per-tensor relative-L2 of loss + every final weight vs
the plain-torch golden trajectory must stay at bf16 noise regardless
of budget (`--out`).

## deep_compare.py — deep correctness compare, one family × shape

The escalation instrument behind
[correctness_compare.md](../../docs/correctness_compare.md):
dW-space gates, follower isolation, near-tie flip classification.

| flag | meaning |
|---|---|
| `--family` | family under test (required) |
| `--shape {uniform,ragged}` | sequence shape |
| `--hot-mult` | hot-init multiplier for gradient visibility |
| `--isolate N` | feed the ENGINE's block N−1 output into BOTH sides — isolates block N from upstream drift |

## sweep_ladder3.py — ladder-3 measurement sweep

Every family × {uniform, ragged} through the model-step gate;
`--families a,b,c` narrows, `--top N` bounds the report.

## rdma_preflight.py — peer-plane RDMA preflight

Proves the load-bearing RDMA assumptions on a box before cross-box
work: device presence, pinned-memory registration, loopback
bandwidth (`--device mlx5_X`, `--mib N`).
