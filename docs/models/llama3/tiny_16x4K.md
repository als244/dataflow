# llama3 / `tiny` @ 16x4K: tasks, objects, kernels

GENERATED from `ShapedLlamaConfig.tiny()` at run shape microbatch 16 × seq 4096 — regenerate with `python tools/gen_model_page.py --preset tiny --microbatch 16 --seq-len 4096`. All presets: [builtin_models.md](../../builtin_models.md); task-kind fleet index: [task_kinds.md](../../task_kinds.md).

Layer kinds (2 layers): `block block`

**Run shape**: microbatch 16 × seq_len 4096 = **65,536 tokens per round** (× 1 grad-accum round(s) per step). `A_*`/`M_*` objects are sized per round; bytes/token figures transfer to any run shape.

## Object summary

At this run shape (65,536 tokens/round). Token-scaled objects show bytes/token in parens. Details per kind below.

| object | scope | bytes |
|---|---|---|
| `W_i (block)` | layer | 86,528 |
| `dW_i (block)` | layer/step | 86,528 |
| `O_i (block)` | layer | 173,056 |
| `A (block)` | layer × round | 77,070,336 (1,176.0/token) |
| `W_head` | run | 65,792 |
| `W_embed` | run | 65,536 |
| `O_embed` | run | 131,072 |
| `O_head` | run | 131,584 |
| `hidden state (y)` | boundary buffer | 8,388,608 (128.0/token) |

### Aggregate totals (all layers, this run shape)

| type | objects | total bytes |
|---|---|---|
| W (all weights, incl. embed/head) | 4 | 304,384 |
| dW (all gradients, incl. metadata grads, per step) | 4 | 304,384 |
| O (all optimizer state) | 4 | 608,768 |
| A (all saved contexts, one round) | 2 | 154,140,672 (2,352.0/token) |

## Dims

| field | value |
|---|---|
| `d_model` | 64 |
| `n_heads` | 4 |
| `n_kv_heads` | 2 |
| `d_ff` | 160 |
| `vocab_size` | 512 |
| `tokens` | 65536 |
| `seq_len` | 4096 |
| `rope_base` | 500000.0 |
| `opt_policy` | adamw |

## Objects, per layer kind

`dW_i` mirrors `W_i`'s fields at the grad dtypes; `O_i` holds the optimizer policy's state slots per field (adamw default: `m_f`+`v_f` at the opt dtype; sgd fields contribute none — see extending.md §6). `A_i`/`M_i` exist per (step, round).

### kind `block` (e.g. layer 0)

**`W_0` weights** — 86,528 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (64,) | 128 |
| `wq` | bf16 | (64, 64) | 8,192 |
| `wk` | bf16 | (64, 32) | 4,096 |
| `wv` | bf16 | (64, 32) | 4,096 |
| `wo` | bf16 | (64, 64) | 8,192 |
| `ffn_norm_w` | bf16 | (64,) | 128 |
| `w1` | bf16 | (64, 160) | 20,480 |
| `w3` | bf16 | (64, 160) | 20,480 |
| `w2` | bf16 | (160, 64) | 20,480 |

**`A_.._0` saved context** — 77,070,336 bytes = **1,176.0 bytes/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 262,144 |
| `q` | bf16 | (65536, 64) | 8,388,608 |
| `k` | bf16 | (65536, 32) | 4,194,304 |
| `v` | bf16 | (65536, 32) | 4,194,304 |
| `lse` | fp32 | (64, 4096) | 1,048,576 |
| `attn_out` | bf16 | (65536, 64) | 8,388,608 |
| `h_mid` | bf16 | (65536, 64) | 8,388,608 |
| `rstd_ffn` | fp32 | (65536,) | 262,144 |
| `x1` | bf16 | (65536, 160) | 20,971,520 |
| `x3` | bf16 | (65536, 160) | 20,971,520 |

**`W_head`** — 65,792 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `w` | bf16 | (512, 64) | 65,536 |
| `final_norm_w` | bf16 | (64,) | 128 |

## Tasks

### `embed_fwd` — `EmbedFwd`

- example task: `embed_fwd_0_0`
- inputs: `tokens_0_0` (262,144B), `W_embed` (65,536B)
- outputs: `y_embed_0_0` (8,388,608B)
- mutates: —
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): index_select

### `block_fwd` — `BlockFwd`

- example task: `block_fwd_0_0_0`
- inputs: `y_embed_0_0` (8,388,608B), `W_0` (86,528B)
- outputs: `y_0_0_0` (8,388,608B), `A_0_0_0` (77,070,336B)
- mutates: —
- stages (name — emitted ctx fields):
    0. `attn_norm` — rstd_attn
    1. `qkv_rope` — q, k, v
    2. `attn` — lse, attn_out
    3. `resid1_norm2` — h_mid, rstd_ffn
    4. `up_proj` — x1, x3  ← derived recompute boundary
    5. `swiglu` — —
    6. `down_resid` — —
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): rmsnorm_fwd → mm → rope_fwd → mm → rope_fwd → mm → _scaled_dot_product_flash_attention → addmm → rmsnorm_fwd → mm×2 → swiglu_fwd_out → addmm

### `head_loss` — `HeadLoss`

- example task: `head_loss_0_0`
- inputs: `y_0_0_1` (8,388,608B), `targets_0_0` (262,144B), `W_head` (65,792B)
- outputs: `dy_0_0_1` (8,388,608B), `loss_0_0` (4B), `dW_head_0` (65,792B)
- mutates: —
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): rmsnorm_fwd → mm → ce_loss_fwd_bwd → mm×2 → rmsnorm_bwd

### `optimizer_head` — `AdamWStep`

- example task: `optimizer_head_0`
- inputs: `W_head` (65,792B), `dW_head_0` (65,792B), `O_head` (131,584B)
- outputs: —
- mutates: `W_head`, `O_head`
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): adamw_step×2

### `block_bwd` — `BlockBwd`

- example task: `block_bwd_0_0_1`
- inputs: `dy_0_0_1` (8,388,608B), `A_0_0_1` (77,070,336B), `y_0_0_0` (8,388,608B), `W_1` (86,528B)
- outputs: `dy_0_0_0` (8,388,608B), `dW_0_1` (86,528B)
- mutates: —
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): rmsnorm_apply → swiglu_fwd_out → mm×2 → swiglu_bwd → mm×3 → rmsnorm_bwd → mm×2 → _scaled_dot_product_flash_attention_backward → rope_bwd×2 → rmsnorm_apply → mm×4 → rmsnorm_bwd

### `optimizer_block` — `AdamWStep`

- example task: `optimizer_0_1`
- inputs: `W_1` (86,528B), `dW_0_1` (86,528B), `O_1` (173,056B)
- outputs: —
- mutates: `W_1`, `O_1`
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): adamw_step×9

### `embed_bwd` — `EmbedBwd`

- example task: `embed_bwd_0_0`
- inputs: `dy_embed_0_0` (8,388,608B), `tokens_0_0` (262,144B)
- outputs: `dW_embed_0` (65,536B)
- mutates: —
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): embed_bwd_accum

### `optimizer_embed` — `AdamWStep`

- example task: `optimizer_embed_0`
- inputs: `W_embed` (65,536B), `dW_embed_0` (65,536B), `O_embed` (131,072B)
- outputs: —
- mutates: `W_embed`, `O_embed`
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): adamw_step

