# qwen35 / `tiny_tied` @ 16x4K: tasks, objects, kernels

GENERATED from `ShapedQwen35Config.tiny_tied()` at run shape microbatch 16 × seq 4096 — regenerate with `python tools/gen_model_page.py --preset tiny_tied --microbatch 16 --seq-len 4096`. All presets: [builtin_models.md](../../builtin_models.md); task-kind fleet index: [task_kinds.md](../../task_kinds.md).

Layer kinds (4 layers): `lin lin lin full`

**Run shape**: microbatch 16 × seq_len 4096 = **65,536 tokens per round** (× 1 grad-accum round(s) per step). `A_*`/`M_*` objects are sized per round; bytes/token figures transfer to any run shape.

## Object summary

At this run shape (65,536 tokens/round). Token-scaled objects show per-token size in parens. Details per kind below.

| object | scope | size |
|---|---|---|
| `W_i (lin)` | layer | 1.01 MiB |
| `dW_i (lin)` | layer/step | 1.01 MiB |
| `O_i (lin)` | layer | 2.02 MiB |
| `A (lin)` | layer × round | 259.50 MiB (4.05 KiB/token) |
| `W_i (full)` | layer | 1.25 MiB |
| `dW_i (full)` | layer/step | 1.25 MiB |
| `O_i (full)` | layer | 2.50 MiB |
| `A (full)` | layer × round | 291.00 MiB (4.55 KiB/token) |
| `W_head` | run | 256.50 KiB |
| `W_embed` | run | 256.50 KiB |
| `O_embed` | run | 513.00 KiB |
| `hidden state (y)` | boundary buffer | 32.00 MiB (512.0 B/token) |

### Aggregate totals (all layers, this run shape)

| type | objects | total size |
|---|---|---|
| W (all weights, incl. embed/head) | 5 | 4.52 MiB |
| dW (all gradients, per step) | 5 | 4.52 MiB |
| O (all optimizer state) | 5 | 9.05 MiB |
| A (all saved activations, one round) | 4 | 1.04 GiB (16.71 KiB/token) |

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

**`W_0` weights** — 1.01 MiB

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (256,) | 512 B |
| `w_qkvz` | bf16 | (256, 384) | 192.00 KiB |
| `w_ba` | bf16 | (256, 8) | 4.00 KiB |
| `w_conv` | bf16 | (256, 4) | 2.00 KiB |
| `A_log` | bf16 | (4,) | 8 B |
| `dt_bias` | bf16 | (4,) | 8 B |
| `lin_norm_w` | bf16 | (32,) | 64 B |
| `w_out` | bf16 | (128, 256) | 64.00 KiB |
| `ffn_norm_w` | bf16 | (256,) | 512 B |
| `w1` | bf16 | (256, 512) | 256.00 KiB |
| `w3` | bf16 | (256, 512) | 256.00 KiB |
| `w2` | bf16 | (512, 256) | 256.00 KiB |

**`A_.._0` saved context** — 259.50 MiB = **4.05 KiB/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 256.00 KiB |
| `qkvz` | bf16 | (65536, 384) | 48.00 MiB |
| `ba` | bf16 | (65536, 8) | 1.00 MiB |
| `g_post` | fp32 | (65536, 4) | 1.00 MiB |
| `A_int` | bf16 | (65536, 4, 64) | 32.00 MiB |
| `core_out` | bf16 | (65536, 4, 32) | 16.00 MiB |
| `rstd_gate` | fp32 | (262144,) | 1.00 MiB |
| `xo` | bf16 | (65536, 256) | 32.00 MiB |
| `rstd_ffn` | fp32 | (65536,) | 256.00 KiB |
| `x1` | bf16 | (65536, 512) | 64.00 MiB |
| `x3` | bf16 | (65536, 512) | 64.00 MiB |

### kind `full` (e.g. layer 3)

**`W_3` weights** — 1.25 MiB

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (256,) | 512 B |
| `wq` | bf16 | (256, 512) | 256.00 KiB |
| `wk` | bf16 | (256, 128) | 64.00 KiB |
| `wv` | bf16 | (256, 128) | 64.00 KiB |
| `q_norm_w` | bf16 | (64,) | 128 B |
| `k_norm_w` | bf16 | (64,) | 128 B |
| `wo` | bf16 | (256, 256) | 128.00 KiB |
| `ffn_norm_w` | bf16 | (256,) | 512 B |
| `w1` | bf16 | (256, 512) | 256.00 KiB |
| `w3` | bf16 | (256, 512) | 256.00 KiB |
| `w2` | bf16 | (512, 256) | 256.00 KiB |

**`A_.._3` saved context** — 291.00 MiB = **4.55 KiB/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 256.00 KiB |
| `qm` | bf16 | (65536, 256) | 32.00 MiB |
| `km` | bf16 | (65536, 128) | 16.00 MiB |
| `rstd_q` | fp32 | (262144,) | 1.00 MiB |
| `rstd_k` | fp32 | (131072,) | 512.00 KiB |
| `gate` | bf16 | (65536, 256) | 32.00 MiB |
| `v` | bf16 | (65536, 128) | 16.00 MiB |
| `lse` | fp32 | (64, 4096) | 1.00 MiB |
| `attn_out` | bf16 | (65536, 256) | 32.00 MiB |
| `xo` | bf16 | (65536, 256) | 32.00 MiB |
| `rstd_ffn` | fp32 | (65536,) | 256.00 KiB |
| `x1` | bf16 | (65536, 512) | 64.00 MiB |
| `x3` | bf16 | (65536, 512) | 64.00 MiB |

**`W_head`** — 256.50 KiB

| field | dtype | shape | bytes |
|---|---|---|---|
| `w` | bf16 | (512, 256) | 256.00 KiB |
| `final_norm_w` | bf16 | (256,) | 512 B |

## Tasks

### `embed_fwd` — `EmbedFwd`

- example task: `embed_fwd_0_0`
- inputs: `tokens_0_0` (256.00 KiB), `W_embed` (256.50 KiB)
- outputs: `y_embed_0_0` (32.00 MiB)
- mutates: —
- kernel calls:
    0. `index_select`

### `linattn_fwd` — `Qwen35LinBlockFwd`

- example task: `block_fwd_0_0_0`
- inputs: `y_embed_0_0` (32.00 MiB), `W_0` (1.01 MiB)
- outputs: `y_0_0_0` (32.00 MiB), `A_0_0_0` (259.50 MiB)
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
- inputs: `y_0_0_2` (32.00 MiB), `W_3` (1.25 MiB)
- outputs: `y_0_0_3` (32.00 MiB), `A_0_0_3` (291.00 MiB)
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
- inputs: `y_0_0_3` (32.00 MiB), `targets_0_0` (256.00 KiB), `W_embed` (256.50 KiB)
- outputs: `dy_0_0_3` (32.00 MiB), `loss_0_0` (4 B), `dW_embed_0` (256.50 KiB)
- mutates: —
- kernel calls:
    0. `rmsnorm_fwd`
    1. `mm`
    2. `ce_loss_fwd_bwd`
    3. `mm ×2`
    4. `rmsnorm_bwd`

### `gattn_bwd` — `Qwen35AttnBlockBwd`

- example task: `block_bwd_0_0_3`
- inputs: `dy_0_0_3` (32.00 MiB), `A_0_0_3` (291.00 MiB), `y_0_0_2` (32.00 MiB), `W_3` (1.25 MiB)
- outputs: `dy_0_0_2` (32.00 MiB), `dW_0_3` (1.25 MiB)
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
- inputs: `W_3` (1.25 MiB), `dW_0_3` (1.25 MiB), `O_3` (2.50 MiB)
- outputs: —
- mutates: `W_3`, `O_3`
- kernel calls:
    0. `adamw_step ×11`

### `linattn_bwd` — `Qwen35LinBlockBwd`

- example task: `block_bwd_0_0_2`
- inputs: `dy_0_0_2` (32.00 MiB), `A_0_0_2` (259.50 MiB), `y_0_0_1` (32.00 MiB), `W_2` (1.01 MiB)
- outputs: `dy_0_0_1` (32.00 MiB), `dW_0_2` (1.01 MiB)
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
- inputs: `dy_embed_0_0` (32.00 MiB), `tokens_0_0` (256.00 KiB), `dW_embed_0` (256.50 KiB)
- outputs: —
- mutates: `dW_embed_0`
- kernel calls:
    0. `embed_bwd_accum`

### `optimizer_embed` — `AdamWStep`

- example task: `optimizer_embed_0`
- inputs: `W_embed` (256.50 KiB), `dW_embed_0` (256.50 KiB), `O_embed` (513.00 KiB)
- outputs: —
- mutates: `W_embed`, `O_embed`
- kernel calls:
    0. `adamw_step`

