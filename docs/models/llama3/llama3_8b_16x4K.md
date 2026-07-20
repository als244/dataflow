# llama3 / `llama3_8b` @ 16x4K: tasks, objects, kernels

GENERATED from `ShapedLlamaConfig.llama3_8b()` at run shape microbatch 16 × seq 4096 — regenerate with `python tools/gen_model_page.py --preset llama3_8b --microbatch 16 --seq-len 4096`. All presets: [builtin_models.md](../../builtin_models.md); task-kind fleet index: [task_kinds.md](../../task_kinds.md).

Layer kinds (32 layers): `block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block`

**Run shape**: microbatch 16 × seq_len 4096 = **65,536 tokens per round** (× 1 grad-accum round(s) per step). `A_*`/`AuxTemp_*` objects are sized per round; bytes/token figures transfer to any run shape.

## Object summary

At this run shape (65,536 tokens/round). Token-scaled objects show per-token size in parens. Details per kind below.

| object | scope | size |
|---|---|---|
| `W_i (block)` | layer | 416.02 MiB |
| `dW_i (block)` | layer/step | 416.02 MiB |
| `O_i (block)` | layer | 832.03 MiB |
| `A (block)` | layer × round | 5.26 GiB (84.13 KiB/token) |
| `W_head` | run | 1,002.01 MiB |
| `W_embed` | run | 1,002.00 MiB |
| `O_embed` | run | 1.96 GiB |
| `O_head` | run | 1.96 GiB |
| `hidden state (y)` | boundary buffer | 512.00 MiB (8.00 KiB/token) |

### Aggregate totals (all layers, this run shape)

| type | objects | total size |
|---|---|---|
| W (all weights, incl. embed/head) | 34 | 14.96 GiB |
| dW (all gradients, per step) | 34 | 14.96 GiB |
| O (all optimizer state) | 34 | 29.92 GiB |
| A (all saved activations, one round) | 32 | 168.27 GiB (2.63 MiB/token) |

## Dims

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

## Objects, per layer kind

`dW_i` mirrors `W_i`'s fields at the grad dtypes; `O_i` holds the optimizer policy's state slots per field (adamw default: `m_f`+`v_f` at the opt dtype; sgd fields contribute none — see extending.md §6). `A_i`/`M_i` exist per (step, round).

### kind `block` (e.g. layer 0)

**`W_0` weights** — 416.02 MiB

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (4096,) | 8.00 KiB |
| `wq` | bf16 | (4096, 4096) | 32.00 MiB |
| `wk` | bf16 | (4096, 1024) | 8.00 MiB |
| `wv` | bf16 | (4096, 1024) | 8.00 MiB |
| `wo` | bf16 | (4096, 4096) | 32.00 MiB |
| `ffn_norm_w` | bf16 | (4096,) | 8.00 KiB |
| `w1` | bf16 | (4096, 14336) | 112.00 MiB |
| `w3` | bf16 | (4096, 14336) | 112.00 MiB |
| `w2` | bf16 | (14336, 4096) | 112.00 MiB |

**`A_.._0` saved context** — 5.26 GiB = **84.13 KiB/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 256.00 KiB |
| `q` | bf16 | (65536, 4096) | 512.00 MiB |
| `k` | bf16 | (65536, 1024) | 128.00 MiB |
| `v` | bf16 | (65536, 1024) | 128.00 MiB |
| `lse` | fp32 | (32, 65536) | 8.00 MiB |
| `attn_out` | bf16 | (65536, 4096) | 512.00 MiB |
| `h_mid` | bf16 | (65536, 4096) | 512.00 MiB |
| `rstd_ffn` | fp32 | (65536,) | 256.00 KiB |
| `x1` | bf16 | (65536, 14336) | 1.75 GiB |
| `x3` | bf16 | (65536, 14336) | 1.75 GiB |

**`W_head`** — 1,002.01 MiB

| field | dtype | shape | bytes |
|---|---|---|---|
| `w` | bf16 | (128256, 4096) | 1,002.00 MiB |
| `final_norm_w` | bf16 | (4096,) | 8.00 KiB |

## Tasks

### `embed_fwd` — `EmbedFwd`

- example task: `embed_fwd_0_0`
- inputs: `tokens_0_0` (256.00 KiB), `W_embed` (1,002.00 MiB)
- outputs: `y_embed_0_0` (512.00 MiB)
- mutates: —
- kernel calls:
    0. `index_select`

### `block_fwd` — `BlockFwd`

- example task: `block_fwd_0_0_0`
- inputs: `y_embed_0_0` (512.00 MiB), `W_0` (416.02 MiB)
- outputs: `y_0_0_0` (512.00 MiB), `A_0_0_0` (5.26 GiB)
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
- inputs: `y_0_0_31` (512.00 MiB), `targets_0_0` (256.00 KiB), `W_head` (1,002.01 MiB)
- outputs: `dy_0_0_31` (512.00 MiB), `loss_0_0` (4 B), `dW_head_0` (1,002.01 MiB)
- mutates: —
- kernel calls:
    0. `rmsnorm_fwd`
    1. `mm`
    2. `ce_loss_fwd_bwd`
    3. `mm ×2`
    4. `rmsnorm_bwd`

### `optimizer_head` — `AdamWStep`

- example task: `optimizer_head_0`
- inputs: `W_head` (1,002.01 MiB), `dW_head_0` (1,002.01 MiB), `O_head` (1.96 GiB)
- outputs: —
- mutates: `W_head`, `O_head`
- kernel calls:
    0. `adamw_step ×2`

### `block_bwd` — `BlockBwd`

- example task: `block_bwd_0_0_31`
- inputs: `dy_0_0_31` (512.00 MiB), `A_0_0_31` (5.26 GiB), `y_0_0_30` (512.00 MiB), `W_31` (416.02 MiB)
- outputs: `dy_0_0_30` (512.00 MiB), `dW_0_31` (416.02 MiB)
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

- example task: `optimizer_0_31`
- inputs: `W_31` (416.02 MiB), `dW_0_31` (416.02 MiB), `O_31` (832.03 MiB)
- outputs: —
- mutates: `W_31`, `O_31`
- kernel calls:
    0. `adamw_step ×9`

### `embed_bwd` — `EmbedBwd`

- example task: `embed_bwd_0_0`
- inputs: `dy_embed_0_0` (512.00 MiB), `tokens_0_0` (256.00 KiB)
- outputs: `dW_embed_0` (1,002.00 MiB)
- mutates: —
- kernel calls:
    0. `embed_bwd_accum`

### `optimizer_embed` — `AdamWStep`

- example task: `optimizer_embed_0`
- inputs: `W_embed` (1,002.00 MiB), `dW_embed_0` (1,002.00 MiB), `O_embed` (1.96 GiB)
- outputs: —
- mutates: `W_embed`, `O_embed`
- kernel calls:
    0. `adamw_step`

