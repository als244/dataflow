# qwen35 / `tiny_tied` @ 16x4K: tasks, objects, kernels

GENERATED from `ShapedQwen35Config.tiny_tied()` at run shape microbatch 16 × seq 4096 — regenerate with `python tools/gen_model_page.py --preset tiny_tied --microbatch 16 --seq-len 4096`. All presets: [builtin_models.md](../../builtin_models.md); task-kind fleet index: [task_kinds.md](../../task_kinds.md).

Layer kinds (4 layers): `lin lin lin full`

**Run shape**: microbatch 16 × seq_len 4096 = **65,536 tokens per round** (× 1 grad-accum round(s) per step). `A_*`/`M_*` objects are sized per round; bytes/token figures transfer to any run shape.

## Object summary

At this run shape (65,536 tokens/round). Token-scaled objects show bytes/token in parens. Details per kind below.

| object | scope | bytes |
|---|---|---|
| `W_i (lin)` | layer | 1,056,512 |
| `dW_i (lin)` | layer/step | 1,056,512 |
| `O_i (lin)` | layer | 2,113,024 |
| `A (lin)` | layer × round | 272,105,472 (4,152.0/token) |
| `W_i (full)` | layer | 1,312,256 |
| `dW_i (full)` | layer/step | 1,312,256 |
| `O_i (full)` | layer | 2,624,512 |
| `A (full)` | layer × round | 305,135,616 (4,656.0/token) |
| `W_head` | run | 262,656 |
| `W_embed` | run | 262,656 |
| `O_embed` | run | 525,312 |
| `hidden state (y)` | boundary buffer | 33,554,432 (512.0/token) |

### Aggregate totals (all layers, this run shape)

| type | objects | total bytes |
|---|---|---|
| W (all weights, incl. embed/head) | 5 | 4,744,448 |
| dW (all gradients, per step) | 5 | 4,744,448 |
| O (all optimizer state) | 5 | 9,488,896 |
| A (all saved activations, one round) | 4 | 1,121,452,032 (17,112.0/token) |

## Dims

| field | value |
|---|---|
| `d_model` | 256 |
| `n_layers` | 4 |
| `full_attention_interval` | 4 |
| `n_heads` | 4 |
| `n_kv_heads` | 2 |
| `head_dim` | 64 |
| `partial_rotary_factor` | 0.25 |
| `lin_k_heads` | 2 |
| `lin_v_heads` | 4 |
| `lin_k_head_dim` | 32 |
| `lin_v_head_dim` | 32 |
| `lin_conv_kernel` | 4 |
| `d_ff` | 512 |
| `vocab_size` | 512 |
| `tokens` | 65536 |
| `seq_len` | 4096 |
| `rope_base` | 10000000.0 |
| `opt_policy` | adamw |

## Objects, per layer kind

`dW_i` mirrors `W_i`'s fields at the grad dtypes; `O_i` holds the optimizer policy's state slots per field (adamw default: `m_f`+`v_f` at the opt dtype; sgd fields contribute none — see extending.md §6). `A_i`/`M_i` exist per (step, round).

### kind `lin` (e.g. layer 0)

**`W_0` weights** — 1,056,512 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (256,) | 512 |
| `w_qkvz` | bf16 | (256, 384) | 196,608 |
| `w_ba` | bf16 | (256, 8) | 4,096 |
| `w_conv` | bf16 | (256, 4) | 2,048 |
| `A_log` | bf16 | (4,) | 8 |
| `dt_bias` | bf16 | (4,) | 8 |
| `lin_norm_w` | bf16 | (32,) | 64 |
| `w_out` | bf16 | (128, 256) | 65,536 |
| `ffn_norm_w` | bf16 | (256,) | 512 |
| `w1` | bf16 | (256, 512) | 262,144 |
| `w3` | bf16 | (256, 512) | 262,144 |
| `w2` | bf16 | (512, 256) | 262,144 |

**`A_.._0` saved context** — 272,105,472 bytes = **4,152.0 bytes/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 262,144 |
| `qkvz` | bf16 | (65536, 384) | 50,331,648 |
| `ba` | bf16 | (65536, 8) | 1,048,576 |
| `g_post` | fp32 | (65536, 4) | 1,048,576 |
| `A_int` | bf16 | (65536, 4, 64) | 33,554,432 |
| `core_out` | bf16 | (65536, 4, 32) | 16,777,216 |
| `rstd_gate` | fp32 | (262144,) | 1,048,576 |
| `xo` | bf16 | (65536, 256) | 33,554,432 |
| `rstd_ffn` | fp32 | (65536,) | 262,144 |
| `x1` | bf16 | (65536, 512) | 67,108,864 |
| `x3` | bf16 | (65536, 512) | 67,108,864 |

### kind `full` (e.g. layer 3)

**`W_3` weights** — 1,312,256 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (256,) | 512 |
| `wq` | bf16 | (256, 512) | 262,144 |
| `wk` | bf16 | (256, 128) | 65,536 |
| `wv` | bf16 | (256, 128) | 65,536 |
| `q_norm_w` | bf16 | (64,) | 128 |
| `k_norm_w` | bf16 | (64,) | 128 |
| `wo` | bf16 | (256, 256) | 131,072 |
| `ffn_norm_w` | bf16 | (256,) | 512 |
| `w1` | bf16 | (256, 512) | 262,144 |
| `w3` | bf16 | (256, 512) | 262,144 |
| `w2` | bf16 | (512, 256) | 262,144 |

**`A_.._3` saved context** — 305,135,616 bytes = **4,656.0 bytes/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 262,144 |
| `qm` | bf16 | (65536, 256) | 33,554,432 |
| `km` | bf16 | (65536, 128) | 16,777,216 |
| `rstd_q` | fp32 | (262144,) | 1,048,576 |
| `rstd_k` | fp32 | (131072,) | 524,288 |
| `gate` | bf16 | (65536, 256) | 33,554,432 |
| `v` | bf16 | (65536, 128) | 16,777,216 |
| `lse` | fp32 | (64, 4096) | 1,048,576 |
| `attn_out` | bf16 | (65536, 256) | 33,554,432 |
| `xo` | bf16 | (65536, 256) | 33,554,432 |
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
- inputs: `tokens_0_0` (262,144B), `W_embed` (262,656B)
- outputs: `y_embed_0_0` (33,554,432B)
- mutates: —
- kernel calls:
    0. `index_select`

### `linattn_fwd` — `Qwen35LinBlockFwd`

- example task: `block_fwd_0_0_0`
- inputs: `y_embed_0_0` (33,554,432B), `W_0` (1,056,512B)
- outputs: `y_0_0_0` (33,554,432B), `A_0_0_0` (272,105,472B)
- mutates: —
- stages (name — emitted ctx fields):
    0. `attn_norm` — rstd_attn
    1. `proj` — qkvz, ba
    2. `conv` — —
    3. `heads_l2norm` — —
    4. `fla` — g_post, A_int, core_out
    5. `norm_out` — rstd_gate, xo
    6. `ffn_norm` — rstd_ffn
    7. `up_proj` — x1, x3  ← derived recompute boundary
    8. `swiglu` — —
    9. `down_resid` — —
- kernel calls, by stage:
    - `attn_norm`:
        0. `rmsnorm_fwd`
    - `proj`:
        1. `mm ×2`
    - `conv`:
        2. `causal_conv1d_silu_fwd`
    - `heads_l2norm`:
        3. `fla::l2norm_fwd ×2`
    - `fla`:
        4. `fla::chunk_gated_delta_rule_fwd`
    - `norm_out`:
        5. `gated_rmsnorm_fwd`
        6. `addmm`
    - `ffn_norm`:
        7. `rmsnorm_fwd`
    - `up_proj`:
        8. `mm ×2`
    - `swiglu`:
        9. `swiglu_fwd_out`
    - `down_resid`:
        10. `addmm`

### `gattn_fwd` — `Qwen35AttnBlockFwd`

- example task: `block_fwd_0_0_3`
- inputs: `y_0_0_2` (33,554,432B), `W_3` (1,312,256B)
- outputs: `y_0_0_3` (33,554,432B), `A_0_0_3` (305,135,616B)
- mutates: —
- stages (name — emitted ctx fields):
    0. `attn_norm` — rstd_attn
    1. `qkv_gate` — qm, km, gate, v
    2. `qknorm_rope` — rstd_q, rstd_k
    3. `attn` — lse, attn_out
    4. `gate_o` — xo
    5. `ffn_norm` — rstd_ffn
    6. `up_proj` — x1, x3  ← derived recompute boundary
    7. `swiglu` — —
    8. `down_resid` — —
- kernel calls, by stage:
    - `attn_norm`:
        0. `rmsnorm_fwd`
    - `qkv_gate`:
        1. `mm ×3`
    - `qknorm_rope`:
        2. `rmsnorm_fwd ×2`
        3. `rope_fwd ×2`
    - `attn`:
        4. `_scaled_dot_product_flash_attention`
    - `gate_o`:
        5. `addmm`
    - `ffn_norm`:
        6. `rmsnorm_fwd`
    - `up_proj`:
        7. `mm ×2`
    - `swiglu`:
        8. `swiglu_fwd_out`
    - `down_resid`:
        9. `addmm`

### `head_loss` — `HeadLoss`

- example task: `head_loss_0_0`
- inputs: `y_0_0_3` (33,554,432B), `targets_0_0` (262,144B), `W_embed` (262,656B)
- outputs: `dy_0_0_3` (33,554,432B), `loss_0_0` (4B), `dW_embed_0` (262,656B)
- mutates: —
- kernel calls:
    0. `rmsnorm_fwd`
    1. `mm`
    2. `ce_loss_fwd_bwd`
    3. `mm ×2`
    4. `rmsnorm_bwd`

### `gattn_bwd` — `Qwen35AttnBlockBwd`

- example task: `block_bwd_0_0_3`
- inputs: `dy_0_0_3` (33,554,432B), `A_0_0_3` (305,135,616B), `y_0_0_2` (33,554,432B), `W_3` (1,312,256B)
- outputs: `dy_0_0_2` (33,554,432B), `dW_0_3` (1,312,256B)
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

- example task: `optimizer_0_3`
- inputs: `W_3` (1,312,256B), `dW_0_3` (1,312,256B), `O_3` (2,624,512B)
- outputs: —
- mutates: `W_3`, `O_3`
- kernel calls:
    0. `adamw_step ×11`

### `linattn_bwd` — `Qwen35LinBlockBwd`

- example task: `block_bwd_0_0_2`
- inputs: `dy_0_0_2` (33,554,432B), `A_0_0_2` (272,105,472B), `y_0_0_1` (33,554,432B), `W_2` (1,056,512B)
- outputs: `dy_0_0_1` (33,554,432B), `dW_0_2` (1,056,512B)
- mutates: —
- kernel calls:
    0. `rmsnorm_apply`
    1. `swiglu_fwd_out`
    2. `mm ×2`
    3. `swiglu_bwd`
    4. `mm ×3`
    5. `rmsnorm_bwd`
    6. `mm`
    7. `gated_rmsnorm_bwd`
    8. `mm`
    9. `causal_conv1d_silu_fwd`
    10. `fla::l2norm_fwd ×2`
    11. `fla::chunk_gated_delta_rule_bwd`
    12. `fla::l2norm_bwd ×2`
    13. `causal_conv1d_silu_bwd`
    14. `rmsnorm_apply`
    15. `mm ×3`
    16. `rmsnorm_bwd`

### `embed_bwd` — `EmbedBwd`

- example task: `embed_bwd_0_0`
- inputs: `dy_embed_0_0` (33,554,432B), `tokens_0_0` (262,144B), `dW_embed_0` (262,656B)
- outputs: —
- mutates: `dW_embed_0`
- kernel calls:
    0. `embed_bwd_accum`

### `optimizer_embed` — `AdamWStep`

- example task: `optimizer_embed_0`
- inputs: `W_embed` (262,656B), `dW_embed_0` (262,656B), `O_embed` (525,312B)
- outputs: —
- mutates: `W_embed`, `O_embed`
- kernel calls:
    0. `adamw_step`

