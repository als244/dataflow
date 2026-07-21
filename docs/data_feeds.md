# Data feeds: sources, tokenization, packing, and engine inputs

The data plane is PER-SEQUENCE: a **DataSource** yields sequences
(tokens + targets + metadata), a **DataFeed** hands them out through
a background ingest worker, a **Packer** groups them into the fixed
rounds and steps a training run consumes, and the drivers submit the
result over the engine's ordinary object/run_args wire. The engine
knows none of this — it moves named byte buffers.

```
DataSource ──► DataFeed (ingest worker + bounded queue) ──► Packer
   shards         tokenize (text sources), requeue,          per-round
   jsonl/txt      capture, cursor                            bins + seq_lens
   parquet                                                      │
   synthetic                                                    ▼
   capture                      tokens_0_r / targets_0_r bytes + run_args
                                {"seq_lens": bounds} + valid_rows ──► engine
```

## 1. Sequences

The unit of data is one document / record / window:

```python
Sequence(tokens,            # (n,) int32 ids
         targets,           # (n,) int32; -1 = masked (no loss)
         extras={...})      # optional named per-token arrays
```

- TARGETS ARE THE SOURCE'S JOB. Corpus sources emit the next-token
  shift, with a document's final position targeting the end-of-text
  id (the model learns to emit it). Sources without an end-of-text id
  mask the final position (-1).
- Masked positions ride the loss's ignore-index channel: the CE
  kernels skip `target < 0`, and `valid_rows` (the global loss
  denominator) counts only real targets — masking needs no engine
  surface.
- `extras` is the per-token extension seam (advantage scores,
  reference logprobs): threaded to additional per-round objects only
  when a program declares them; the standard training chain ignores
  it.

## 2. Sources

Every source obeys one contract: `sequences(cursor)` is a pure
function of (constructor args, cursor) — two iterations from equal
cursors yield byte-identical sequences — and `describe()` reports its
JSON-clean facts (vocab, end-of-text id, tokenizer identity, flags)
into run metadata. Corpus sources wrap epochs indefinitely.

Uniform creation flags: `max_seqlen` + `long_policy` —
`exclude` (default: drop over-long sequences), `trim` (truncate),
`chunk` (split into max_seqlen pieces, positions restarting),
`whole` (emit the full document — only for round-splitting packers,
the legacy-reproduction path).

| scheme | source | notes |
|---|---|---|
| `shards:ROOT[,window=N][,split=S]` | token-shard corpora (1024-byte header + uint16 ids) | per-DOCUMENT sequences by default (delimiter-split; the delimiter never appears as an input); `window=N` emits fixed-N windows with the global shift (the fixed-block scheme) |
| `jsonl:GLOB,tokenizer=T[,field=text]` | one document per JSON line | tokenized at ingest |
| `txt:GLOB,tokenizer=T[,delimiter=\n\n]` | delimiter-split plain text | tokenized at ingest |
| `parquet:GLOB,tokenizer=T[,column=text]` | parquet text column | row-group streaming |
| `synthetic:[vocab=V][,mean_len=L][,seed=K]` | seeded random sequences | benchmarking/tests without data |
| `capture:PATH` | replay of a capture file | finite; exactly the recorded hand-out |

HuggingFace hub datasets are NOT a special runtime source: fetch them
to local files first —

    python tools/train/fetch_dataset.py openai/gsm8k --config main
    # -> datasets/gsm8k/train.jsonl  (records always carry "text";
    #    prompt/response shapes also keep instruction/output)
    --data jsonl:datasets/gsm8k/train.jsonl,tokenizer=gpt2

`datasets/` is gitignored — fetched corpora are persistent local
data, naturally replayable.

## 3. Tokenizers

Text sources take `tokenizer=`: `gpt2` (tiktoken), `tiktoken:ENC`,
or `hf:NAME` (any hub tokenizer). `encode` is deterministic and never
truncates (length policy is the source's); the tokenizer identity
(backend, name, vocab, end-of-text id) lands in `RunResult.meta`, so
every curve records what tokenized its data. Token ids are checked
against the run config's vocab at ingest. The backends live in the
optional `[data]` extra (`pip install -e '.[data]'` — tokenizers,
tiktoken, datasets, pyarrow); token sources (shards, synthetic,
capture) need none of it.

## 4. DataFeed

`DataFeed(source, prefetch_sequences=256, capture=..., start_cursor=...)`
owns one background ingest worker filling a bounded queue
(backpressure blocks the worker). Concurrency is an implementation
detail: hand-out order equals source order, gated against the
synchronous path (`prefetch_sequences=0`). `requeue(seqs)` returns
packer remainders to the front of the line. Worker errors surface on
the next `next_sequence()` call.

`capture=PATH` logs every handed-out sequence (post-tokenize,
post-policy); `--data capture:PATH` replays the file exactly. This
is the reference-comparison convention: run the engine leg with
`--capture`, replay the capture into the reference leg — both legs
provably consume identical per-step sequences, across boxes and
sessions.

## 5. Packing

`Packer(feed, tokens_per_round, ga_rounds, max_seqlen, ...)` pulls a
lookahead window (4x the step's token budget, flag-tunable) and
bin-packs sequences into the step's rounds:

- `policy="ffd"` (default): first-fit-decreasing over the window —
  every round as close to full as possible. Sequences that fit
  nowhere requeue in pull order and LEAD the next step.
- `policy="greedy"`: straight feed-order fill, defer at the first
  non-fit — maximally order-preserving.
- `allow_round_split=True`: a sequence may split at a round edge
  (the head fills the round exactly; the tail continues as the next
  round's first segment, chunking at max_seqlen round-locally).
  Rounds are then always exactly full. Combined with
  `long_policy=whole` this reproduces the historical fixed-token
  packing byte-for-byte (pinned by hash gates).

UNDER-FULL ROUNDS (the no-split default): round buffers keep their
fixed planner size, but ONLY CONTENT EXECUTES — `seq_lens` stops at
the content edge and every task computes over views sliced to the
round's real rows (content re-view). The buffer tail is dead bytes:
no kernel reads or writes it. `--execute-padding` restores the
fallback lane where the tail rides as one final MASKED wire segment
(filler tokens, targets -1, exactly zero loss contribution) and
executes for real; the two modes are numerically equivalent (a
standing engine-vs-engine gate pins it). Fill is a first-class
metric: the step log prints it, and `RunResult.meta["data"]`
aggregates content tokens + mean/min fill. tok/s counts CONTENT
tokens only; effective FLOPs scale token-linear buckets to content,
and hardware FLOPs count what actually ran (content by default, tail
included under `--execute-padding`). Packing policies compete on the
fill metric and the loss curve; a policy change never lands
silently.

One current limitation: the tensor-parallel lane (`--tp-mlp`) runs
planner-sized collectives, so under-full rounds there require
`--execute-padding` (or exact-fill packing) — the fleet refuses the
combination rather than mis-communicating.

## 6. Engine inputs — the wire

Exactly what crosses to the daemon, per round `r` of a step:
`tokens_0_r` / `targets_0_r` byte buffers (fixed
`tokens_per_round` int32), plus
`run_args = {"step": k, "valid_rows": REAL-target count,
"seq_lens": {r: [0, b1, ..., content_r]}}` — the bounds END at the
round's content edge (under `--execute-padding` a final masked
segment extends them to the buffer size). The engine's first
consuming task materializes `Segments` (device `cu`/`positions`)
from the bounds — see [program_contract.md](program_contract.md) §9.
Extras become additional per-round objects only for programs that
declare them.

## 7. Determinism, resume, capture

The PackedStep sequence is a pure function of (source spec,
tokenizer spec, packing geometry + flags, cursor). Every
`PackedStep` carries `cursor_after`; the drivers stamp it into
checkpoint `client_meta` (solo + reference) and the fleet manifest,
so resume SEEKS the source — no dataloader state, no replay from
zero. Checkpoints without a cursor fast-forward the packer CPU-side.
The resume drill (checkpoint → kill → resume → curve continues) is a
standing gate, as are the pinned-hash gates on the two legacy
configurations.

## 8. CLI reference

`--data SPEC` on every data-taking tool (`train.py`
train/reference, `measure_step`, `trace_real_run`,
`eval_checkpoint` via its val-split pipeline), plus the packing
flags where geometry lives:

| flag | meaning |
|---|---|
| `--data scheme:main[,k=v...]` | the source (table above); default = the in-repo shard corpus, per-document |
| `--packing-policy {ffd,greedy}` | bin-packing policy |
| `--allow-round-split` | legacy exact-fill packing |
| `--capture PATH` | record the consumed sequences for replay |
| `--execute-padding` | execute under-full tails as one masked segment (debug/fallback; default computes content rows only) |

The two legacy configurations, spelled out:

    # fixed windows (uniform rounds — what plans price):
    --data "shards:,window=1024" --packing-policy greedy
    # historical doc-aware curves:
    --data "shards:,long_policy=whole" --packing-policy greedy \
        --allow-round-split

## 9. Extending: a new DataSource

Implement the two-method contract (`sequences(cursor)` pure-from-
cursor + `describe()`), enforce the Sequence invariants and the long
policy AT THE SOURCE, register a scheme in
`data/sources/resolve_data`, and add a determinism + cursor-resume
gate beside the existing source gates
(tests/dataflow_training/data/). External packages can construct
sources directly and hand a custom pipeline factory to the drivers —
the factory contract is `pipeline(cursor|None) -> stepper` with
`stepper.next_step() -> PackedStep`
([extending_external.md](extending_external.md)).
