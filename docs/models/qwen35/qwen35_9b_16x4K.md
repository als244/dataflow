# qwen35 / `qwen35_9b` @ 16x4K: tasks, objects, kernels

GENERATED from `ShapedQwen35Config.qwen35_9b()` at run shape microbatch 16 × seq 4096 — regenerate with `python tools/gen_model_page.py --preset qwen35_9b --microbatch 16 --seq-len 4096`. All presets: [builtin_models.md](../../builtin_models.md); task-kind fleet index: [task_kinds.md](../../task_kinds.md).

Layer kinds (32 layers): `lin lin lin full lin lin lin full lin lin lin full lin lin lin full lin lin lin full lin lin lin full lin lin lin full lin lin lin full`

**Run shape**: microbatch 16 × seq_len 4096 = **65,536 tokens per round** (× 1 grad-accum round(s) per step). `A_*`/`M_*` objects are sized per round; bytes/token figures transfer to any run shape.

## Object summary

At this run shape (65,536 tokens/round). Token-scaled objects show bytes/token in parens. Details per kind below.

| object | scope | bytes |
|---|---|---|
| `W_i (lin)` | layer | 436,814,592 |
| `dW_i (lin)` | layer/step | 436,814,592 |
| `O_i (lin)` | layer | 873,629,184 |
| `A (lin)` | layer × round | 6,199,705,600 (94,600.0/token) |
| `W_i (full)` | layer | 419,447,808 |
| `dW_i (full)` | layer/step | 419,447,808 |
| `O_i (full)` | layer | 838,895,616 |
| `A (full)` | layer × round | 5,647,106,048 (86,168.0/token) |
| `W_head` | run | 2,034,245,632 |
| `W_embed` | run | 2,034,237,440 |
| `O_embed` | run | 4,068,474,880 |
| `O_head` | run | 4,068,491,264 |
| `hidden state (y)` | boundary buffer | 536,870,912 (8,192.0/token) |

### Aggregate totals (all layers, this run shape)

| type | objects | total bytes |
|---|---|---|
| W (all weights, incl. embed/head) | 34 | 17,907,615,744 |
| dW (all gradients, per step) | 34 | 17,907,615,744 |
| O (all optimizer state) | 34 | 35,815,231,488 |
| A (all saved activations, one round) | 32 | 193,969,782,784 (2,959,744.0/token) |

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

**`W_0` weights** — 436,814,592 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (4096,) | 8,192 |
| `w_qkvz` | bf16 | (4096, 12288) | 100,663,296 |
| `w_ba` | bf16 | (4096, 64) | 524,288 |
| `w_conv` | bf16 | (8192, 4) | 65,536 |
| `A_log` | bf16 | (32,) | 64 |
| `dt_bias` | bf16 | (32,) | 64 |
| `lin_norm_w` | bf16 | (128,) | 256 |
| `w_out` | bf16 | (4096, 4096) | 33,554,432 |
| `ffn_norm_w` | bf16 | (4096,) | 8,192 |
| `w1` | bf16 | (4096, 12288) | 100,663,296 |
| `w3` | bf16 | (4096, 12288) | 100,663,296 |
| `w2` | bf16 | (12288, 4096) | 100,663,296 |

**`A_.._0` saved context** — 6,199,705,600 bytes = **94,600.0 bytes/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 262,144 |
| `qkvz` | bf16 | (65536, 12288) | 1,610,612,736 |
| `ba` | bf16 | (65536, 64) | 8,388,608 |
| `g_post` | fp32 | (65536, 32) | 8,388,608 |
| `A_int` | bf16 | (65536, 32, 64) | 268,435,456 |
| `core_out` | bf16 | (65536, 32, 128) | 536,870,912 |
| `rstd_gate` | fp32 | (2097152,) | 8,388,608 |
| `xo` | bf16 | (65536, 4096) | 536,870,912 |
| `rstd_ffn` | fp32 | (65536,) | 262,144 |
| `x1` | bf16 | (65536, 12288) | 1,610,612,736 |
| `x3` | bf16 | (65536, 12288) | 1,610,612,736 |

### kind `full` (e.g. layer 3)

**`W_3` weights** — 419,447,808 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (4096,) | 8,192 |
| `wq` | bf16 | (4096, 8192) | 67,108,864 |
| `wk` | bf16 | (4096, 1024) | 8,388,608 |
| `wv` | bf16 | (4096, 1024) | 8,388,608 |
| `q_norm_w` | bf16 | (256,) | 512 |
| `k_norm_w` | bf16 | (256,) | 512 |
| `wo` | bf16 | (4096, 4096) | 33,554,432 |
| `ffn_norm_w` | bf16 | (4096,) | 8,192 |
| `w1` | bf16 | (4096, 12288) | 100,663,296 |
| `w3` | bf16 | (4096, 12288) | 100,663,296 |
| `w2` | bf16 | (12288, 4096) | 100,663,296 |

**`A_.._3` saved context** — 5,647,106,048 bytes = **86,168.0 bytes/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 262,144 |
| `qm` | bf16 | (65536, 4096) | 536,870,912 |
| `km` | bf16 | (65536, 1024) | 134,217,728 |
| `rstd_q` | fp32 | (1048576,) | 4,194,304 |
| `rstd_k` | fp32 | (262144,) | 1,048,576 |
| `gate` | bf16 | (65536, 4096) | 536,870,912 |
| `v` | bf16 | (65536, 1024) | 134,217,728 |
| `lse` | fp32 | (256, 4096) | 4,194,304 |
| `attn_out` | bf16 | (65536, 4096) | 536,870,912 |
| `xo` | bf16 | (65536, 4096) | 536,870,912 |
| `rstd_ffn` | fp32 | (65536,) | 262,144 |
| `x1` | bf16 | (65536, 12288) | 1,610,612,736 |
| `x3` | bf16 | (65536, 12288) | 1,610,612,736 |

**`W_head`** — 2,034,245,632 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `w` | bf16 | (248320, 4096) | 2,034,237,440 |
| `final_norm_w` | bf16 | (4096,) | 8,192 |

## Tasks

### `embed_fwd` — `EmbedFwd`

- example task: `embed_fwd_0_0`
- inputs: `tokens_0_0` (262,144B), `W_embed` (2,034,237,440B)
- outputs: `y_embed_0_0` (536,870,912B)
- mutates: —
- kernel calls:
    0. `index_select`

### `linattn_fwd` — `Qwen35LinBlockFwd`

- example task: `block_fwd_0_0_0`
- inputs: `y_embed_0_0` (536,870,912B), `W_0` (436,814,592B)
- outputs: `y_0_0_0` (536,870,912B), `A_0_0_0` (6,199,705,600B)
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
    - `attn_norm`: `rmsnorm_fwd`
    - `proj`: `mm ×2`
    - `conv`: `causal_conv1d_silu_fwd`
    - `heads_l2norm`: `fla::l2norm_fwd ×2`
    - `fla`: `fla::chunk_gated_delta_rule_fwd`
    - `norm_out`: `gated_rmsnorm_fwd`, `addmm`
    - `ffn_norm`: `rmsnorm_fwd`
    - `up_proj`: `mm ×2`
    - `swiglu`: `swiglu_fwd_out`
    - `down_resid`: `addmm`

### `gattn_fwd` — `Qwen35AttnBlockFwd`

- example task: `block_fwd_0_0_3`
- inputs: `y_0_0_2` (536,870,912B), `W_3` (419,447,808B)
- outputs: `y_0_0_3` (536,870,912B), `A_0_0_3` (5,647,106,048B)
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
    - `attn_norm`: `rmsnorm_fwd`
    - `qkv_gate`: `mm ×3`
    - `qknorm_rope`: `rmsnorm_fwd ×2`, `rope_fwd ×2`
    - `attn`: `_scaled_dot_product_flash_attention`
    - `gate_o`: `addmm`
    - `ffn_norm`: `rmsnorm_fwd`
    - `up_proj`: `mm ×2`
    - `swiglu`: `swiglu_fwd_out`
    - `down_resid`: `addmm`

### `head_loss` — `HeadLoss`

- example task: `head_loss_0_0`
- inputs: `y_0_0_31` (536,870,912B), `targets_0_0` (262,144B), `W_head` (2,034,245,632B)
- outputs: `dy_0_0_31` (536,870,912B), `loss_0_0` (4B), `dW_head_0` (2,034,245,632B)
- mutates: —
- kernel calls:
    0. `rmsnorm_fwd`
    1. `mm`
    2. `ce_loss_fwd_bwd`
    3. `mm ×2`
    4. `rmsnorm_bwd`

### `optimizer_head` — `AdamWStep`

- example task: `optimizer_head_0`
- inputs: `W_head` (2,034,245,632B), `dW_head_0` (2,034,245,632B), `O_head` (4,068,491,264B)
- outputs: —
- mutates: `W_head`, `O_head`
- kernel calls:
    0. `adamw_step ×2`

### `gattn_bwd` — `Qwen35AttnBlockBwd`

- example task: `block_bwd_0_0_31`
- inputs: `dy_0_0_31` (536,870,912B), `A_0_0_31` (5,647,106,048B), `y_0_0_30` (536,870,912B), `W_31` (419,447,808B)
- outputs: `dy_0_0_30` (536,870,912B), `dW_0_31` (419,447,808B)
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

- example task: `optimizer_0_31`
- inputs: `W_31` (419,447,808B), `dW_0_31` (419,447,808B), `O_31` (838,895,616B)
- outputs: —
- mutates: `W_31`, `O_31`
- kernel calls:
    0. `adamw_step ×11`

### `linattn_bwd` — `Qwen35LinBlockBwd`

- example task: `block_bwd_0_0_30`
- inputs: `dy_0_0_30` (536,870,912B), `A_0_0_30` (6,199,705,600B), `y_0_0_29` (536,870,912B), `W_30` (436,814,592B)
- outputs: `dy_0_0_29` (536,870,912B), `dW_0_30` (436,814,592B)
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
- inputs: `dy_embed_0_0` (536,870,912B), `tokens_0_0` (262,144B)
- outputs: `dW_embed_0` (2,034,237,440B)
- mutates: —
- kernel calls:
    0. `embed_bwd_accum`

### `optimizer_embed` — `AdamWStep`

- example task: `optimizer_embed_0`
- inputs: `W_embed` (2,034,237,440B), `dW_embed_0` (2,034,237,440B), `O_embed` (4,068,474,880B)
- outputs: —
- mutates: `W_embed`, `O_embed`
- kernel calls:
    0. `adamw_step`

