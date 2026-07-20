# llama3 / `tiny` @ 16x4K: tasks, objects, kernels

GENERATED from `ShapedLlamaConfig.tiny()` at run shape microbatch 16 × seq 4096 — regenerate with `python tools/gen_model_page.py --preset tiny --microbatch 16 --seq-len 4096`. All presets: [builtin_models.md](../../builtin_models.md); task-kind fleet index: [task_kinds.md](../../task_kinds.md).

Layer kinds (2 layers): `block block`

**Run shape**: microbatch 16 × seq_len 4096 = **65,536 tokens per round** (× 1 grad-accum round(s) per step). `A_*`/`AuxTemp_*` objects are sized per round; bytes/token figures transfer to any run shape.

## Object summary

At this run shape (65,536 tokens/round). Token-scaled objects show per-token size in parens. Details per kind below.

| object | scope | size |
|---|---|---|
| `W_i (block)` | layer | 84.50 KiB |
| `dW_i (block)` | layer/step | 84.50 KiB |
| `O_i (block)` | layer | 169.00 KiB |
| `A (block)` | layer × round | 73.50 MiB (1.15 KiB/token) |
| `W_head` | run | 64.25 KiB |
| `W_embed` | run | 64.00 KiB |
| `O_embed` | run | 128.00 KiB |
| `O_head` | run | 128.50 KiB |
| `hidden state (y)` | boundary buffer | 8.00 MiB (128.0 B/token) |

### Aggregate totals (all layers, this run shape)

| type | objects | total size |
|---|---|---|
| W (all weights, incl. embed/head) | 4 | 297.25 KiB |
| dW (all gradients, per step) | 4 | 297.25 KiB |
| O (all optimizer state) | 4 | 594.50 KiB |
| A (all saved activations, one round) | 2 | 147.00 MiB (2.30 KiB/token) |

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

**`W_0` weights** — 84.50 KiB

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (64,) | 128 B |
| `wq` | bf16 | (64, 64) | 8.00 KiB |
| `wk` | bf16 | (64, 32) | 4.00 KiB |
| `wv` | bf16 | (64, 32) | 4.00 KiB |
| `wo` | bf16 | (64, 64) | 8.00 KiB |
| `ffn_norm_w` | bf16 | (64,) | 128 B |
| `w1` | bf16 | (64, 160) | 20.00 KiB |
| `w3` | bf16 | (64, 160) | 20.00 KiB |
| `w2` | bf16 | (160, 64) | 20.00 KiB |

**`A_.._0` saved context** — 73.50 MiB = **1.15 KiB/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 256.00 KiB |
| `q` | bf16 | (65536, 64) | 8.00 MiB |
| `k` | bf16 | (65536, 32) | 4.00 MiB |
| `v` | bf16 | (65536, 32) | 4.00 MiB |
| `lse` | fp32 | (4, 65536) | 1.00 MiB |
| `attn_out` | bf16 | (65536, 64) | 8.00 MiB |
| `h_mid` | bf16 | (65536, 64) | 8.00 MiB |
| `rstd_ffn` | fp32 | (65536,) | 256.00 KiB |
| `x1` | bf16 | (65536, 160) | 20.00 MiB |
| `x3` | bf16 | (65536, 160) | 20.00 MiB |

**`W_head`** — 64.25 KiB

| field | dtype | shape | bytes |
|---|---|---|---|
| `w` | bf16 | (512, 64) | 64.00 KiB |
| `final_norm_w` | bf16 | (64,) | 128 B |

## Tasks

### `embed_fwd` — `EmbedFwd`

- example task: `embed_fwd_0_0`
- inputs: `tokens_0_0` (256.00 KiB), `W_embed` (64.00 KiB)
- outputs: `y_embed_0_0` (8.00 MiB)
- mutates: —
- kernel calls:
    0. `index_select`

### `block_fwd` — `BlockFwd`

- example task: `block_fwd_0_0_0`
- inputs: `y_embed_0_0` (8.00 MiB), `W_0` (84.50 KiB)
- outputs: `y_0_0_0` (8.00 MiB), `A_0_0_0` (73.50 MiB)
- mutates: —
- stages (name — emitted ctx fields):
    0. `attn_norm` — rstd_attn
    1. `qkv_rope` — q, k, v
    2. `attn` — lse, attn_out
    3. `resid1_norm2` — h_mid, rstd_ffn
    4. `up_proj` — x1, x3  ← derived recompute boundary
    5. `swiglu` — —
    6. `down_resid` — —
- kernel calls, by stage:
    - `attn_norm`:
        0. `rmsnorm_fwd`
    - `qkv_rope`:
        1. `mm`
        2. `rope_fwd`
        3. `mm`
        4. `rope_fwd`
        5. `mm`
    - `resid1_norm2`:
        6. `addmm`
        7. `rmsnorm_fwd`
    - `up_proj`:
        8. `mm ×2`
    - `swiglu`:
        9. `swiglu_fwd_out`
    - `down_resid`:
        10. `addmm`

### `head_loss` — `HeadLoss`

- example task: `head_loss_0_0`
- inputs: `y_0_0_1` (8.00 MiB), `targets_0_0` (256.00 KiB), `W_head` (64.25 KiB)
- outputs: `dy_0_0_1` (8.00 MiB), `loss_0_0` (4 B), `dW_head_0` (64.25 KiB)
- mutates: —
- kernel calls:
    0. `rmsnorm_fwd`
    1. `mm`
    2. `ce_loss_fwd_bwd`
    3. `mm ×2`
    4. `rmsnorm_bwd`

### `optimizer_head` — `AdamWStep`

- example task: `optimizer_head_0`
- inputs: `W_head` (64.25 KiB), `dW_head_0` (64.25 KiB), `O_head` (128.50 KiB)
- outputs: —
- mutates: `W_head`, `O_head`
- kernel calls:
    0. `adamw_step ×2`

### `block_bwd` — `BlockBwd`

- example task: `block_bwd_0_0_1`
- inputs: `dy_0_0_1` (8.00 MiB), `A_0_0_1` (73.50 MiB), `y_0_0_0` (8.00 MiB), `W_1` (84.50 KiB)
- outputs: `dy_0_0_0` (8.00 MiB), `dW_0_1` (84.50 KiB)
- mutates: —
- kernel calls:
    0. `rmsnorm_apply`
    1. `swiglu_fwd_out`
    2. `mm ×2`
    3. `swiglu_bwd`
    4. `mm ×3`
    5. `rmsnorm_bwd`
    6. `mm ×2`
    7. `_flash_attention_backward`
    8. `rope_bwd ×2`
    9. `rmsnorm_apply`
    10. `mm ×4`
    11. `rmsnorm_bwd`

### `optimizer_block` — `AdamWStep`

- example task: `optimizer_0_1`
- inputs: `W_1` (84.50 KiB), `dW_0_1` (84.50 KiB), `O_1` (169.00 KiB)
- outputs: —
- mutates: `W_1`, `O_1`
- kernel calls:
    0. `adamw_step ×9`

### `embed_bwd` — `EmbedBwd`

- example task: `embed_bwd_0_0`
- inputs: `dy_embed_0_0` (8.00 MiB), `tokens_0_0` (256.00 KiB)
- outputs: `dW_embed_0` (64.00 KiB)
- mutates: —
- kernel calls:
    0. `embed_bwd_accum`

### `optimizer_embed` — `AdamWStep`

- example task: `optimizer_embed_0`
- inputs: `W_embed` (64.00 KiB), `dW_embed_0` (64.00 KiB), `O_embed` (128.00 KiB)
- outputs: —
- mutates: `W_embed`, `O_embed`
- kernel calls:
    0. `adamw_step`

