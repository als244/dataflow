# llama3: tasks, objects, kernels

GENERATED from `ShapedLlamaConfig.llama3_8b()` at the standard documentation run shape (seq 4096 × microbatch 16) — regenerate with `python tools/gen_model_docs.py --family llama3`. Presets: [builtin_models.md](../builtin_models.md); task-kind fleet index: [task_kinds.md](../task_kinds.md).

Layer kinds (32 layers): `block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block`

**Run shape of this documentation preset**: microbatch 16 × seq_len 4096 = **65,536 tokens per round** (× 1 grad-accum round(s) per step). `A_*`/`M_*` objects are sized per round; their bytes/token figures below transfer to any run shape.

## Dims (documentation preset)

| field | value |
|---|---|
| `d_model` | 4096 |
| `n_heads` | 32 |
| `n_kv_heads` | 8 |
| `d_ff` | 14336 |
| `vocab_size` | 128256 |
| `tokens` | 65536 |
| `seq_len` | 4096 |
| `rope_base` | 500000.0 |
| `opt_policy` | adamw |

## Object summary

At the documentation run shape (65,536 tokens/round). Token-scaled objects show bytes/token in parens. Details per kind below.

| object | scope | bytes |
|---|---|---|
| `W_i (block)` | layer | 436,224,000 |
| `dW_i (block)` | layer/step | 436,224,000 |
| `O_i (block)` | layer | 872,448,000 |
| `A (block)` | layer × round | 5,646,057,472 (86,152.0/token) |
| `W_head` | run | 1,050,681,344 |
| `W_embed` | run | 1,050,673,152 |
| `O_embed` | run | 2,101,346,304 |
| `O_head` | run | 2,101,362,688 |
| `hidden state (y)` | boundary buffer | 536,870,912 (8,192.0/token) |

### Aggregate totals (all layers, this run shape)

| type | objects | total bytes |
|---|---|---|
| W (all weights, incl. embed/head) | 34 | 16,060,522,496 |
| dW (all gradients, per step) | 34 | 16,060,522,496 |
| O (all optimizer state) | 34 | 32,121,044,992 |
| A (all saved contexts, one round) | 32 | 180,673,839,104 (2,756,864.0/token) |

## Objects, per layer kind

`dW_i` mirrors `W_i`'s fields at the grad dtypes; `O_i` holds the optimizer policy's state slots per field (adamw default: `m_f`+`v_f` at the opt dtype; sgd fields contribute none — see extending.md §6). `A_i`/`M_i` exist per (step, round).

### kind `block` (e.g. layer 0)

**`W_0` weights** — 436,224,000 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (4096,) | 8,192 |
| `wq` | bf16 | (4096, 4096) | 33,554,432 |
| `wk` | bf16 | (4096, 1024) | 8,388,608 |
| `wv` | bf16 | (4096, 1024) | 8,388,608 |
| `wo` | bf16 | (4096, 4096) | 33,554,432 |
| `ffn_norm_w` | bf16 | (4096,) | 8,192 |
| `w1` | bf16 | (4096, 14336) | 117,440,512 |
| `w3` | bf16 | (4096, 14336) | 117,440,512 |
| `w2` | bf16 | (14336, 4096) | 117,440,512 |

**`A_.._0` saved context** — 5,646,057,472 bytes = **86,152.0 bytes/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 262,144 |
| `q` | bf16 | (65536, 4096) | 536,870,912 |
| `k` | bf16 | (65536, 1024) | 134,217,728 |
| `v` | bf16 | (65536, 1024) | 134,217,728 |
| `lse` | fp32 | (512, 4096) | 8,388,608 |
| `attn_out` | bf16 | (65536, 4096) | 536,870,912 |
| `h_mid` | bf16 | (65536, 4096) | 536,870,912 |
| `rstd_ffn` | fp32 | (65536,) | 262,144 |
| `x1` | bf16 | (65536, 14336) | 1,879,048,192 |
| `x3` | bf16 | (65536, 14336) | 1,879,048,192 |

**`W_head`** — 1,050,681,344 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `w` | bf16 | (128256, 4096) | 1,050,673,152 |
| `final_norm_w` | bf16 | (4096,) | 8,192 |

## Tasks

### `embed_fwd` — `EmbedFwd`

- example task: `embed_fwd_0_0`
- inputs: `tokens_0_0` (262,144B), `W_embed` (1,050,673,152B)
- outputs: `y_embed_0_0` (536,870,912B)
- mutates: —

### `block_fwd` — `BlockFwd`

- example task: `block_fwd_0_0_0`
- inputs: `y_embed_0_0` (536,870,912B), `W_0` (436,224,000B)
- outputs: `y_0_0_0` (536,870,912B), `A_0_0_0` (5,646,057,472B)
- mutates: —
- stages (name — emitted ctx fields):
    0. `attn_norm` — rstd_attn
    1. `qkv_rope` — q, k, v
    2. `attn` — lse, attn_out
    3. `resid1_norm2` — h_mid, rstd_ffn
    4. `up_proj` — x1, x3  ← derived recompute boundary
    5. `swiglu` — —
    6. `down_resid` — —
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): rmsnorm_fwd → rope_fwd×2 → rmsnorm_fwd → swiglu_fwd_out

### `head_loss` — `HeadLoss`

- example task: `head_loss_0_0`
- inputs: `y_0_0_31` (536,870,912B), `targets_0_0` (262,144B), `W_head` (1,050,681,344B)
- outputs: `dy_0_0_31` (536,870,912B), `loss_0_0` (4B), `dW_head_0` (1,050,681,344B)
- mutates: —
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): rmsnorm_fwd → ce_loss_fwd_bwd → rmsnorm_bwd

### `optimizer_head` — `AdamWStep`

- example task: `optimizer_head_0`
- inputs: `W_head` (1,050,681,344B), `dW_head_0` (1,050,681,344B), `O_head` (2,101,362,688B)
- outputs: —
- mutates: `W_head`, `O_head`
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): adamw_step×2

### `block_bwd` — `BlockBwd`

- example task: `block_bwd_0_0_31`
- inputs: `dy_0_0_31` (536,870,912B), `A_0_0_31` (5,646,057,472B), `y_0_0_30` (536,870,912B), `W_31` (436,224,000B)
- outputs: `dy_0_0_30` (536,870,912B), `dW_0_31` (436,224,000B)
- mutates: —
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): rmsnorm_apply → swiglu_fwd_out → swiglu_bwd → rmsnorm_bwd → rope_bwd×2 → rmsnorm_apply → rmsnorm_bwd

### `optimizer_block` — `AdamWStep`

- example task: `optimizer_0_31`
- inputs: `W_31` (436,224,000B), `dW_0_31` (436,224,000B), `O_31` (872,448,000B)
- outputs: —
- mutates: `W_31`, `O_31`
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): adamw_step×9

### `embed_bwd` — `EmbedBwd`

- example task: `embed_bwd_0_0`
- inputs: `dy_embed_0_0` (536,870,912B), `tokens_0_0` (262,144B)
- outputs: `dW_embed_0` (1,050,673,152B)
- mutates: —
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): embed_bwd_accum

### `optimizer_embed` — `AdamWStep`

- example task: `optimizer_embed_0`
- inputs: `W_embed` (1,050,673,152B), `dW_embed_0` (1,050,673,152B), `O_embed` (2,101,346,304B)
- outputs: —
- mutates: `W_embed`, `O_embed`
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): adamw_step

