# qwen35 / `qwen35_9b` @ 16x4K: tasks, objects, kernels

GENERATED from `ShapedQwen35Config.qwen35_9b()` at run shape microbatch 16 × seq 4096 — regenerate with `python tools/gen_model_docs/gen_model_page.py --preset qwen35_9b --microbatch 16 --seq-len 4096`. All presets: [builtin_models.md](../../builtin_models.md); task-kind fleet index: [task_kinds.md](../../task_kinds.md).

Layer kinds (32 layers): `lin lin lin full lin lin lin full lin lin lin full lin lin lin full lin lin lin full lin lin lin full lin lin lin full lin lin lin full`

**Run shape**: microbatch 16 × seq_len 4096 = **65,536 tokens per round** (× 1 grad-accum round(s) per step). `A_*`/`AuxTemp_*` objects are sized per round; bytes/token figures transfer to any run shape.

## Object summary

At this run shape (65,536 tokens/round). Token-scaled objects show per-token size in parens. Details per kind below.

| object | scope | size |
|---|---|---|
| `W_i (lin)` | layer | 416.58 MiB |
| `dW_i (lin)` | layer/step | 416.58 MiB |
| `O_i (lin)` | layer | 833.16 MiB |
| `A (lin)` | layer × round | 5.77 GiB (92.38 KiB/token) |
| `W_i (full)` | layer | 400.02 MiB |
| `dW_i (full)` | layer/step | 400.02 MiB |
| `O_i (full)` | layer | 800.03 MiB |
| `A (full)` | layer × round | 5.26 GiB (84.15 KiB/token) |
| `W_head` | run | 1.89 GiB |
| `W_embed` | run | 1.89 GiB |
| `O_embed` | run | 3.79 GiB |
| `O_head` | run | 3.79 GiB |
| `hidden state (y)` | boundary buffer | 512.00 MiB (8.00 KiB/token) |

### Aggregate totals (all layers, this run shape)

| type | objects | total size |
|---|---|---|
| W (all weights, incl. embed/head) | 34 | 16.68 GiB |
| dW (all gradients, per step) | 34 | 16.68 GiB |
| O (all optimizer state) | 34 | 33.36 GiB |
| A (all saved activations, one round) | 32 | 180.65 GiB (2.82 MiB/token) |

## Dims

| field | value |
|---|---|
| `d_model` | 4096 |
| `n_layers` | 32 |
| `full_attention_interval` | 4 |
| `n_heads` | 16 |
| `n_kv_heads` | 4 |
| `head_dim` | 256 |
| `partial_rotary_factor` | 0.25 |
| `lin_k_heads` | 16 |
| `lin_v_heads` | 32 |
| `lin_k_head_dim` | 128 |
| `lin_v_head_dim` | 128 |
| `lin_conv_kernel` | 4 |
| `d_ff` | 12288 |
| `vocab_size` | 248320 |
| `tokens` | 65536 |
| `seq_len` | 4096 |
| `rope_base` | 10000000.0 |
| `opt_policy` | adamw |

## Objects, per layer kind

`dW_i` mirrors `W_i`'s fields at the grad dtypes; `O_i` holds the optimizer policy's state slots per field (adamw default: `m_f`+`v_f` at the opt dtype; sgd fields contribute none — see extending.md §6). `A_i`/`M_i` exist per (step, round).

### kind `lin` (e.g. layer 0)

**`W_0` weights** — 416.58 MiB

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (4096,) | 8.00 KiB |
| `w_qkvz` | bf16 | (4096, 12288) | 96.00 MiB |
| `w_ba` | bf16 | (4096, 64) | 512.00 KiB |
| `w_conv` | bf16 | (8192, 4) | 64.00 KiB |
| `A_log` | bf16 | (32,) | 64 B |
| `dt_bias` | bf16 | (32,) | 64 B |
| `lin_norm_w` | bf16 | (128,) | 256 B |
| `w_out` | bf16 | (4096, 4096) | 32.00 MiB |
| `ffn_norm_w` | bf16 | (4096,) | 8.00 KiB |
| `w1` | bf16 | (4096, 12288) | 96.00 MiB |
| `w3` | bf16 | (4096, 12288) | 96.00 MiB |
| `w2` | bf16 | (12288, 4096) | 96.00 MiB |

**`A_.._0` saved context** — 5.77 GiB = **92.38 KiB/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 256.00 KiB |
| `qkvz` | bf16 | (65536, 12288) | 1.50 GiB |
| `ba` | bf16 | (65536, 64) | 8.00 MiB |
| `g_post` | fp32 | (65536, 32) | 8.00 MiB |
| `A_int` | bf16 | (65536, 32, 64) | 256.00 MiB |
| `core_out` | bf16 | (65536, 32, 128) | 512.00 MiB |
| `rstd_gate` | fp32 | (2097152,) | 8.00 MiB |
| `xo` | bf16 | (65536, 4096) | 512.00 MiB |
| `rstd_ffn` | fp32 | (65536,) | 256.00 KiB |
| `x1` | bf16 | (65536, 12288) | 1.50 GiB |
| `x3` | bf16 | (65536, 12288) | 1.50 GiB |

### kind `full` (e.g. layer 3)

**`W_3` weights** — 400.02 MiB

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (4096,) | 8.00 KiB |
| `wq` | bf16 | (4096, 8192) | 64.00 MiB |
| `wk` | bf16 | (4096, 1024) | 8.00 MiB |
| `wv` | bf16 | (4096, 1024) | 8.00 MiB |
| `q_norm_w` | bf16 | (256,) | 512 B |
| `k_norm_w` | bf16 | (256,) | 512 B |
| `wo` | bf16 | (4096, 4096) | 32.00 MiB |
| `ffn_norm_w` | bf16 | (4096,) | 8.00 KiB |
| `w1` | bf16 | (4096, 12288) | 96.00 MiB |
| `w3` | bf16 | (4096, 12288) | 96.00 MiB |
| `w2` | bf16 | (12288, 4096) | 96.00 MiB |

**`A_.._3` saved context** — 5.26 GiB = **84.15 KiB/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 256.00 KiB |
| `qm` | bf16 | (65536, 4096) | 512.00 MiB |
| `km` | bf16 | (65536, 1024) | 128.00 MiB |
| `rstd_q` | fp32 | (1048576,) | 4.00 MiB |
| `rstd_k` | fp32 | (262144,) | 1.00 MiB |
| `gate` | bf16 | (65536, 4096) | 512.00 MiB |
| `v` | bf16 | (65536, 1024) | 128.00 MiB |
| `lse` | fp32 | (16, 65536) | 4.00 MiB |
| `attn_out` | bf16 | (65536, 4096) | 512.00 MiB |
| `xo` | bf16 | (65536, 4096) | 512.00 MiB |
| `rstd_ffn` | fp32 | (65536,) | 256.00 KiB |
| `x1` | bf16 | (65536, 12288) | 1.50 GiB |
| `x3` | bf16 | (65536, 12288) | 1.50 GiB |

**`W_head`** — 1.89 GiB

| field | dtype | shape | bytes |
|---|---|---|---|
| `w` | bf16 | (248320, 4096) | 1.89 GiB |
| `final_norm_w` | bf16 | (4096,) | 8.00 KiB |

## Tasks

### `embed_fwd` — `EmbedFwd`

- example task: `embed_fwd_0_0`
- inputs: `tokens_0_0` (256.00 KiB), `W_embed` (1.89 GiB)
- outputs: `y_embed_0_0` (512.00 MiB)
- mutates: —
- kernel calls:
    0. `index_select`

### `linattn_fwd` — `Qwen35LinBlockFwd`

- example task: `block_fwd_0_0_0`
- inputs: `y_embed_0_0` (512.00 MiB), `W_0` (416.58 MiB)
- outputs: `y_0_0_0` (512.00 MiB), `A_0_0_0` (5.77 GiB)
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
- inputs: `y_0_0_2` (512.00 MiB), `W_3` (400.02 MiB)
- outputs: `y_0_0_3` (512.00 MiB), `A_0_0_3` (5.26 GiB)
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
    - `gate_o`:
        4. `addmm`
    - `ffn_norm`:
        5. `rmsnorm_fwd`
    - `up_proj`:
        6. `mm ×2`
    - `swiglu`:
        7. `swiglu_fwd_out`
    - `down_resid`:
        8. `addmm`

### `head_loss` — `HeadLoss`

- example task: `head_loss_0_0`
- inputs: `y_0_0_31` (512.00 MiB), `targets_0_0` (256.00 KiB), `W_head` (1.89 GiB)
- outputs: `dy_0_0_31` (512.00 MiB), `loss_0_0` (4 B), `dW_head_0` (1.89 GiB)
- mutates: —
- kernel calls:
    0. `rmsnorm_fwd`
    1. `mm`
    2. `ce_loss_fwd_bwd`
    3. `mm ×2`
    4. `rmsnorm_bwd`

### `optimizer_head` — `AdamWStep`

- example task: `optimizer_head_0`
- inputs: `W_head` (1.89 GiB), `dW_head_0` (1.89 GiB), `O_head` (3.79 GiB)
- outputs: —
- mutates: `W_head`, `O_head`
- kernel calls:
    0. `adamw_step ×2`

### `gattn_bwd` — `Qwen35AttnBlockBwd`

- example task: `block_bwd_0_0_31`
- inputs: `dy_0_0_31` (512.00 MiB), `A_0_0_31` (5.26 GiB), `y_0_0_30` (512.00 MiB), `W_31` (400.02 MiB)
- outputs: `dy_0_0_30` (512.00 MiB), `dW_0_31` (400.02 MiB)
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
    9. `_flash_attention_backward`
    10. `rope_bwd ×2`
    11. `rmsnorm_bwd ×2`
    12. `rmsnorm_apply`
    13. `mm ×4`
    14. `rmsnorm_bwd`

### `optimizer_block` — `AdamWStep`

- example task: `optimizer_0_31`
- inputs: `W_31` (400.02 MiB), `dW_0_31` (400.02 MiB), `O_31` (800.03 MiB)
- outputs: —
- mutates: `W_31`, `O_31`
- kernel calls:
    0. `adamw_step ×11`

### `linattn_bwd` — `Qwen35LinBlockBwd`

- example task: `block_bwd_0_0_30`
- inputs: `dy_0_0_30` (512.00 MiB), `A_0_0_30` (5.77 GiB), `y_0_0_29` (512.00 MiB), `W_30` (416.58 MiB)
- outputs: `dy_0_0_29` (512.00 MiB), `dW_0_30` (416.58 MiB)
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
- inputs: `dy_embed_0_0` (512.00 MiB), `tokens_0_0` (256.00 KiB)
- outputs: `dW_embed_0` (1.89 GiB)
- mutates: —
- kernel calls:
    0. `embed_bwd_accum`

### `optimizer_embed` — `AdamWStep`

- example task: `optimizer_embed_0`
- inputs: `W_embed` (1.89 GiB), `dW_embed_0` (1.89 GiB), `O_embed` (3.79 GiB)
- outputs: —
- mutates: `W_embed`, `O_embed`
- kernel calls:
    0. `adamw_step`

