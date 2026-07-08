# qwen3 / `tiny` @ 16x4K: tasks, objects, kernels

GENERATED from `ShapedQwen3Config.tiny()` at run shape microbatch 16 × seq 4096 — regenerate with `python tools/gen_model_page.py --preset tiny --microbatch 16 --seq-len 4096`. All presets: [builtin_models.md](../../builtin_models.md); task-kind fleet index: [task_kinds.md](../../task_kinds.md).

Layer kinds (2 layers): `block block`

**Run shape**: microbatch 16 × seq_len 4096 = **65,536 tokens per round** (× 1 grad-accum round(s) per step). `A_*`/`M_*` objects are sized per round; bytes/token figures transfer to any run shape.

## Object summary

At this run shape (65,536 tokens/round). Token-scaled objects show bytes/token in parens. Details per kind below.

| object | scope | bytes |
|---|---|---|
| `W_i (block)` | layer | 1,181,184 |
| `dW_i (block)` | layer/step | 1,181,184 |
| `O_i (block)` | layer | 2,362,368 |
| `A (block)` | layer × round | 271,581,184 (4,144.0/token) |
| `W_head` | run | 262,656 |
| `W_embed` | run | 262,144 |
| `O_embed` | run | 524,288 |
| `O_head` | run | 525,312 |
| `hidden state (y)` | boundary buffer | 33,554,432 (512.0/token) |

### Aggregate totals (all layers, this run shape)

| type | objects | total bytes |
|---|---|---|
| W (all weights, incl. embed/head) | 4 | 2,887,168 |
| dW (all gradients, per step) | 4 | 2,887,168 |
| O (all optimizer state) | 4 | 5,774,336 |
| A (all saved activations, one round) | 2 | 543,162,368 (8,288.0/token) |

## Dims

| field | value |
|---|---|
| `d_model` | 256 |
| `n_heads` | 4 |
| `n_kv_heads` | 2 |
| `head_dim` | 64 |
| `d_ff` | 512 |
| `vocab_size` | 512 |
| `tokens` | 65536 |
| `seq_len` | 4096 |
| `rope_base` | 1000000.0 |
| `opt_policy` | adamw |

## Objects, per layer kind

`dW_i` mirrors `W_i`'s fields at the grad dtypes; `O_i` holds the optimizer policy's state slots per field (adamw default: `m_f`+`v_f` at the opt dtype; sgd fields contribute none — see extending.md §6). `A_i`/`M_i` exist per (step, round).

### kind `block` (e.g. layer 0)

**`W_0` weights** — 1,181,184 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (256,) | 512 |
| `wq` | bf16 | (256, 256) | 131,072 |
| `wk` | bf16 | (256, 128) | 65,536 |
| `wv` | bf16 | (256, 128) | 65,536 |
| `q_norm_w` | bf16 | (64,) | 128 |
| `k_norm_w` | bf16 | (64,) | 128 |
| `wo` | bf16 | (256, 256) | 131,072 |
| `ffn_norm_w` | bf16 | (256,) | 512 |
| `w1` | bf16 | (256, 512) | 262,144 |
| `w3` | bf16 | (256, 512) | 262,144 |
| `w2` | bf16 | (512, 256) | 262,144 |

**`A_.._0` saved context** — 271,581,184 bytes = **4,144.0 bytes/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 262,144 |
| `qm` | bf16 | (65536, 256) | 33,554,432 |
| `km` | bf16 | (65536, 128) | 16,777,216 |
| `rstd_q` | fp32 | (262144,) | 1,048,576 |
| `rstd_k` | fp32 | (131072,) | 524,288 |
| `v` | bf16 | (65536, 128) | 16,777,216 |
| `lse` | fp32 | (64, 4096) | 1,048,576 |
| `attn_out` | bf16 | (65536, 256) | 33,554,432 |
| `h_mid` | bf16 | (65536, 256) | 33,554,432 |
| `rstd_ffn` | fp32 | (65536,) | 262,144 |
| `x1` | bf16 | (65536, 512) | 67,108,864 |
| `x3` | bf16 | (65536, 512) | 67,108,864 |

**`W_head`** — 262,656 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `w` | bf16 | (512, 256) | 262,144 |
| `final_norm_w` | bf16 | (256,) | 512 |

## Tasks

### `embed_fwd` — `EmbedFwd`

- example task: `embed_fwd_0_0`
- inputs: `tokens_0_0` (262,144B), `W_embed` (262,144B)
- outputs: `y_embed_0_0` (33,554,432B)
- mutates: —
- kernel calls:
    0. `index_select`

### `block_fwd` — `Qwen3BlockFwd`

- example task: `block_fwd_0_0_0`
- inputs: `y_embed_0_0` (33,554,432B), `W_0` (1,181,184B)
- outputs: `y_0_0_0` (33,554,432B), `A_0_0_0` (271,581,184B)
- mutates: —
- stages (name — emitted ctx fields):
    0. `attn_norm` — rstd_attn
    1. `qkv_qknorm` — qm, km, rstd_q, rstd_k, v
    2. `rope` — —
    3. `attn` — lse, attn_out
    4. `resid1_norm2` — h_mid, rstd_ffn
    5. `up_proj` — x1, x3  ← derived recompute boundary
    6. `swiglu` — —
    7. `down_resid` — —
- kernel calls, by stage:
    - `attn_norm`:
        0. `rmsnorm_fwd`
    - `qkv_qknorm`:
        1. `mm ×3`
        2. `rmsnorm_fwd ×2`
    - `rope`:
        3. `rope_fwd ×2`
    - `attn`:
        4. `_scaled_dot_product_flash_attention`
    - `resid1_norm2`:
        5. `addmm`
        6. `rmsnorm_fwd`
    - `up_proj`:
        7. `mm ×2`
    - `swiglu`:
        8. `swiglu_fwd_out`
    - `down_resid`:
        9. `addmm`

### `head_loss` — `HeadLoss`

- example task: `head_loss_0_0`
- inputs: `y_0_0_1` (33,554,432B), `targets_0_0` (262,144B), `W_head` (262,656B)
- outputs: `dy_0_0_1` (33,554,432B), `loss_0_0` (4B), `dW_head_0` (262,656B)
- mutates: —
- kernel calls:
    0. `rmsnorm_fwd`
    1. `mm`
    2. `ce_loss_fwd_bwd`
    3. `mm ×2`
    4. `rmsnorm_bwd`

### `optimizer_head` — `AdamWStep`

- example task: `optimizer_head_0`
- inputs: `W_head` (262,656B), `dW_head_0` (262,656B), `O_head` (525,312B)
- outputs: —
- mutates: `W_head`, `O_head`
- kernel calls:
    0. `adamw_step ×2`

### `block_bwd` — `Qwen3BlockBwd`

- example task: `block_bwd_0_0_1`
- inputs: `dy_0_0_1` (33,554,432B), `A_0_0_1` (271,581,184B), `y_0_0_0` (33,554,432B), `W_1` (1,181,184B)
- outputs: `dy_0_0_0` (33,554,432B), `dW_0_1` (1,181,184B)
- mutates: —
- kernel calls:
    0. `rmsnorm_apply`
    1. `swiglu_fwd_out`
    2. `mm ×2`
    3. `swiglu_bwd`
    4. `mm ×3`
    5. `rmsnorm_bwd`
    6. `mm ×2`
    7. `rmsnorm_apply ×2`
    8. `rope_fwd ×2`
    9. `_scaled_dot_product_flash_attention_backward`
    10. `rope_bwd ×2`
    11. `rmsnorm_bwd ×2`
    12. `rmsnorm_apply`
    13. `mm ×4`
    14. `rmsnorm_bwd`

### `optimizer_block` — `AdamWStep`

- example task: `optimizer_0_1`
- inputs: `W_1` (1,181,184B), `dW_0_1` (1,181,184B), `O_1` (2,362,368B)
- outputs: —
- mutates: `W_1`, `O_1`
- kernel calls:
    0. `adamw_step ×11`

### `embed_bwd` — `EmbedBwd`

- example task: `embed_bwd_0_0`
- inputs: `dy_embed_0_0` (33,554,432B), `tokens_0_0` (262,144B)
- outputs: `dW_embed_0` (262,144B)
- mutates: —
- kernel calls:
    0. `embed_bwd_accum`

### `optimizer_embed` — `AdamWStep`

- example task: `optimizer_embed_0`
- inputs: `W_embed` (262,144B), `dW_embed_0` (262,144B), `O_embed` (524,288B)
- outputs: —
- mutates: `W_embed`, `O_embed`
- kernel calls:
    0. `adamw_step`

