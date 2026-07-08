# llama3: tasks, objects, kernels

GENERATED from `ShapedLlamaConfig.tiny()` — regenerate with `python tools/gen_model_docs.py --family llama3`. Presets: [builtin_models.md](../builtin_models.md); task-kind fleet index: [task_kinds.md](../task_kinds.md).

Layer kinds (2 layers): `block block`

**Run shape of this documentation preset**: microbatch 1 × seq_len 64 = **64 tokens per round** (× 1 grad-accum round(s) per step). `A_*`/`M_*` objects are sized per round; their bytes/token figures below transfer to any run shape.

## Dims (documentation preset)

| field | value |
|---|---|
| `d_model` | 64 |
| `n_heads` | 4 |
| `n_kv_heads` | 2 |
| `d_ff` | 160 |
| `vocab_size` | 512 |
| `tokens` | 64 |
| `seq_len` | 64 |
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

**`A_.._0` saved context** — 75,264 bytes = **1,176.0 bytes/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (64,) | 256 |
| `q` | bf16 | (64, 64) | 8,192 |
| `k` | bf16 | (64, 32) | 4,096 |
| `v` | bf16 | (64, 32) | 4,096 |
| `lse` | fp32 | (4, 64) | 1,024 |
| `attn_out` | bf16 | (64, 64) | 8,192 |
| `h_mid` | bf16 | (64, 64) | 8,192 |
| `rstd_ffn` | fp32 | (64,) | 256 |
| `x1` | bf16 | (64, 160) | 20,480 |
| `x3` | bf16 | (64, 160) | 20,480 |

**`W_head`** — 65,792 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `w` | bf16 | (512, 64) | 65,536 |
| `final_norm_w` | bf16 | (64,) | 128 |

## Tasks

### `embed_fwd` — `EmbedFwd`

- example task: `embed_fwd_0_0`
- inputs: `tokens_0_0` (256B), `W_embed` (65,536B)
- outputs: `y_embed_0_0` (8,192B)
- mutates: —

### `block_fwd` — `BlockFwd`

- example task: `block_fwd_0_0_0`
- inputs: `y_embed_0_0` (8,192B), `W_0` (86,528B)
- outputs: `y_0_0_0` (8,192B), `A_0_0_0` (75,264B)
- mutates: —
- stages (name — emitted ctx fields):
    0. `attn_norm` — rstd_attn
    1. `qkv_rope` — q, k, v
    2. `attn` — lse, attn_out
    3. `resid1_norm2` — h_mid, rstd_ffn
    4. `up_proj` — x1, x3  ← derived recompute boundary
    5. `swiglu` — —
    6. `down_resid` — —
- kernel calls (measured, one launch): rmsnorm_fwd → rope_fwd×2 → rmsnorm_fwd → swiglu_fwd_out

### `head_loss` — `HeadLoss`

- example task: `head_loss_0_0`
- inputs: `y_0_0_1` (8,192B), `targets_0_0` (256B), `W_head` (65,792B)
- outputs: `dy_0_0_1` (8,192B), `loss_0_0` (4B), `dW_head_0` (65,792B)
- mutates: —
- kernel calls (measured, one launch): rmsnorm_fwd → ce_loss_fwd_bwd → rmsnorm_bwd

### `optimizer_head` — `AdamWStep`

- example task: `optimizer_head_0`
- inputs: `W_head` (65,792B), `dW_head_0` (65,792B), `O_head` (131,584B)
- outputs: —
- mutates: `W_head`, `O_head`
- kernel calls (measured, one launch): adamw_step×2

### `block_bwd` — `BlockBwd`

- example task: `block_bwd_0_0_1`
- inputs: `dy_0_0_1` (8,192B), `A_0_0_1` (75,264B), `y_0_0_0` (8,192B), `W_1` (86,528B)
- outputs: `dy_0_0_0` (8,192B), `dW_0_1` (86,528B)
- mutates: —
- kernel calls (measured, one launch): rmsnorm_apply → swiglu_fwd_out → swiglu_bwd → rmsnorm_bwd → rope_bwd×2 → rmsnorm_apply → rmsnorm_bwd

### `optimizer_block` — `AdamWStep`

- example task: `optimizer_0_1`
- inputs: `W_1` (86,528B), `dW_0_1` (86,528B), `O_1` (173,056B)
- outputs: —
- mutates: `W_1`, `O_1`
- kernel calls (measured, one launch): adamw_step×9

### `embed_bwd` — `EmbedBwd`

- example task: `embed_bwd_0_0`
- inputs: `dy_embed_0_0` (8,192B), `tokens_0_0` (256B)
- outputs: `dW_embed_0` (65,536B)
- mutates: —
- kernel calls (measured, one launch): embed_bwd_accum

### `optimizer_embed` — `AdamWStep`

- example task: `optimizer_embed_0`
- inputs: `W_embed` (65,536B), `dW_embed_0` (65,536B), `O_embed` (131,072B)
- outputs: —
- mutates: `W_embed`, `O_embed`
- kernel calls (measured, one launch): adamw_step

