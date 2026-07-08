# qwen35moe / `qwen35moe_35b` @ 16x4K: tasks, objects, kernels

GENERATED from `ShapedQwen35MoeConfig.qwen35moe_35b()` at run shape microbatch 16 × seq 4096 — regenerate with `python tools/gen_model_page.py --preset qwen35moe_35b --microbatch 16 --seq-len 4096`. All presets: [builtin_models.md](../../builtin_models.md); task-kind fleet index: [task_kinds.md](../../task_kinds.md).

Layer kinds (40 layers): `lin lin lin full lin lin lin full lin lin lin full lin lin lin full lin lin lin full lin lin lin full lin lin lin full lin lin lin full lin lin lin full lin lin lin full`

**Run shape**: microbatch 16 × seq_len 4096 = **65,536 tokens per round** (× 1 grad-accum round(s) per step). `A_*`/`M_*` objects are sized per round; bytes/token figures transfer to any run shape.

## Object summary

At this run shape (65,536 tokens/round). Token-scaled objects show bytes/token in parens. Details per kind below.

| object | scope | bytes |
|---|---|---|
| `W_i (lin)` | layer | 1,685,402,368 |
| `dW_i (lin)` | layer/step | 1,685,402,368 |
| `O_i (lin)` | layer | 3,370,804,736 |
| `A (lin)` | layer × round | 3,951,689,728 (60,298.0/token) |
| `W_i (full)` | layer | 1,672,492,032 |
| `dW_i (full)` | layer/step | 1,672,492,032 |
| `O_i (full)` | layer | 3,344,984,064 |
| `A (full)` | layer × round | 3,264,348,160 (49,810.0/token) |
| `W_head` | run | 1,017,122,816 |
| `W_embed` | run | 1,017,118,720 |
| `O_embed` | run | 2,034,237,440 |
| `O_head` | run | 2,034,245,632 |
| `hidden state (y)` | boundary buffer | 268,435,456 (4,096.0/token) |

### Aggregate totals (all layers, this run shape)

| type | objects | total bytes |
|---|---|---|
| W (all weights, incl. embed/head) | 42 | 69,321,232,896 |
| dW (all gradients, per step) | 42 | 69,321,232,896 |
| O (all optimizer state) | 42 | 138,642,465,792 |
| A (all saved activations, one round) | 40 | 151,194,173,440 (2,307,040.0/token) |
| M (all metadata, one round) | 40 | 209,766,400 (3,200.8/token) |

## Dims

| field | value |
|---|---|
| `d_model` | 2048 |
| `n_layers` | 40 |
| `full_attention_interval` | 4 |
| `n_heads` | 16 |
| `n_kv_heads` | 2 |
| `head_dim` | 256 |
| `partial_rotary_factor` | 0.25 |
| `lin_k_heads` | 16 |
| `lin_v_heads` | 32 |
| `lin_k_head_dim` | 128 |
| `lin_v_head_dim` | 128 |
| `lin_conv_kernel` | 4 |
| `d_ff` | 512 |
| `vocab_size` | 248320 |
| `tokens` | 65536 |
| `seq_len` | 4096 |
| `rope_base` | 10000000.0 |
| `opt_policy` | adamw |

## Objects, per layer kind

`dW_i` mirrors `W_i`'s fields at the grad dtypes; `O_i` holds the optimizer policy's state slots per field (adamw default: `m_f`+`v_f` at the opt dtype; sgd fields contribute none — see extending.md §6). `A_i`/`M_i` exist per (step, round).

### kind `lin` (e.g. layer 0)

**`W_0` weights** — 1,685,402,368 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (2048,) | 4,096 |
| `w_qkvz` | bf16 | (2048, 12288) | 50,331,648 |
| `w_ba` | bf16 | (2048, 64) | 262,144 |
| `w_conv` | bf16 | (8192, 4) | 65,536 |
| `A_log` | bf16 | (32,) | 64 |
| `dt_bias` | bf16 | (32,) | 64 |
| `lin_norm_w` | bf16 | (128,) | 256 |
| `w_out` | bf16 | (4096, 2048) | 16,777,216 |
| `ffn_norm_w` | bf16 | (2048,) | 4,096 |
| `w_router` | bf16 | (2048, 256) | 1,048,576 |
| `w13_experts` | bf16 | (256, 2048, 1024) | 1,073,741,824 |
| `w2_experts` | bf16 | (256, 512, 2048) | 536,870,912 |
| `w_shared_gate` | bf16 | (2048, 1) | 4,096 |
| `w_s13` | bf16 | (2048, 1024) | 4,194,304 |
| `w_s2` | bf16 | (512, 2048) | 2,097,152 |

**`A_.._0` saved context** — 3,951,689,728 bytes = **60,298.0 bytes/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 262,144 |
| `qkvz` | bf16 | (65536, 12288) | 1,610,612,736 |
| `ba` | bf16 | (65536, 64) | 8,388,608 |
| `g_post` | fp32 | (65536, 32) | 8,388,608 |
| `A_int` | bf16 | (65536, 32, 64) | 268,435,456 |
| `core_out` | bf16 | (65536, 32, 128) | 536,870,912 |
| `rstd_gate` | fp32 | (2097152,) | 8,388,608 |
| `xo` | bf16 | (65536, 2048) | 268,435,456 |
| `rstd_ffn` | fp32 | (65536,) | 262,144 |
| `router_logits` | bf16 | (65536, 256) | 33,554,432 |
| `h13` | bf16 | (524288, 1024) | 1,073,741,824 |
| `gate_pre` | bf16 | (65536, 1) | 131,072 |
| `s13` | bf16 | (65536, 1024) | 134,217,728 |

### kind `full` (e.g. layer 3)

**`W_3` weights** — 1,672,492,032 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (2048,) | 4,096 |
| `wq` | bf16 | (2048, 8192) | 33,554,432 |
| `wk` | bf16 | (2048, 512) | 2,097,152 |
| `wv` | bf16 | (2048, 512) | 2,097,152 |
| `q_norm_w` | bf16 | (256,) | 512 |
| `k_norm_w` | bf16 | (256,) | 512 |
| `wo` | bf16 | (4096, 2048) | 16,777,216 |
| `ffn_norm_w` | bf16 | (2048,) | 4,096 |
| `w_router` | bf16 | (2048, 256) | 1,048,576 |
| `w13_experts` | bf16 | (256, 2048, 1024) | 1,073,741,824 |
| `w2_experts` | bf16 | (256, 512, 2048) | 536,870,912 |
| `w_shared_gate` | bf16 | (2048, 1) | 4,096 |
| `w_s13` | bf16 | (2048, 1024) | 4,194,304 |
| `w_s2` | bf16 | (512, 2048) | 2,097,152 |

**`A_.._3` saved context** — 3,264,348,160 bytes = **49,810.0 bytes/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 262,144 |
| `qm` | bf16 | (65536, 4096) | 536,870,912 |
| `km` | bf16 | (65536, 512) | 67,108,864 |
| `rstd_q` | fp32 | (1048576,) | 4,194,304 |
| `rstd_k` | fp32 | (131072,) | 524,288 |
| `gate` | bf16 | (65536, 4096) | 536,870,912 |
| `v` | bf16 | (65536, 512) | 67,108,864 |
| `lse` | fp32 | (256, 4096) | 4,194,304 |
| `attn_out` | bf16 | (65536, 4096) | 536,870,912 |
| `xo` | bf16 | (65536, 2048) | 268,435,456 |
| `rstd_ffn` | fp32 | (65536,) | 262,144 |
| `router_logits` | bf16 | (65536, 256) | 33,554,432 |
| `h13` | bf16 | (524288, 1024) | 1,073,741,824 |
| `gate_pre` | bf16 | (65536, 1) | 131,072 |
| `s13` | bf16 | (65536, 1024) | 134,217,728 |

**`W_head`** — 1,017,122,816 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `w` | bf16 | (248320, 2048) | 1,017,118,720 |
| `final_norm_w` | bf16 | (2048,) | 4,096 |

## Tasks

### `embed_fwd` — `EmbedFwd`

- example task: `embed_fwd_0_0`
- inputs: `tokens_0_0` (262,144B), `W_embed` (1,017,118,720B)
- outputs: `y_embed_0_0` (268,435,456B)
- mutates: —
- kernel calls:
    0. `index_select`

### `linmoe_fwd` — `Qwen35MoeLinBlockFwd`

- example task: `block_fwd_0_0_0`
- inputs: `y_embed_0_0` (268,435,456B), `W_0` (1,685,402,368B)
- outputs: `y_0_0_0` (268,435,456B), `A_0_0_0` (3,951,689,728B), `M_0_0_0` (5,244,160B)
- mutates: —
- stages (name — emitted ctx fields):
    0. `attn_norm` — rstd_attn
    1. `proj` — qkvz, ba
    2. `conv` — —
    3. `heads_l2norm` — —
    4. `fla` — g_post, A_int, core_out
    5. `norm_out` — rstd_gate, xo
    6. `ffn_norm` — rstd_ffn
    7. `moe_route` — router_logits
    8. `moe_dispatch` — —
    9. `moe_experts13` — h13
    10. `moe_shared` — s13, gate_pre  ← derived recompute boundary
    11. `moe_experts2_combine` — —
- kernel calls:
    0. `rmsnorm_fwd`
    1. `mm ×2`
    2. `causal_conv1d_silu_fwd`
    3. `fla::l2norm_fwd ×2`
    4. `fla::chunk_gated_delta_rule_fwd`
    5. `gated_rmsnorm_fwd`
    6. `addmm`
    7. `rmsnorm_fwd`
    8. `mm`
    9. `moe_topk_softmax`
    10. `moe_sort`
    11. `moe_dispatch_fwd`
    12. `moe_grouped_mm_fwd`
    13. `mm ×2`
    14. `swiglu_packed_fwd`
    15. `moe_grouped_mm_fwd`
    16. `swiglu_packed_fwd`
    17. `mm`
    18. `moe_scale_rows`
    19. `moe_combine_fwd`

### `gattnmoe_fwd` — `Qwen35MoeAttnBlockFwd`

- example task: `block_fwd_0_0_3`
- inputs: `y_0_0_2` (268,435,456B), `W_3` (1,672,492,032B)
- outputs: `y_0_0_3` (268,435,456B), `A_0_0_3` (3,264,348,160B), `M_0_0_3` (5,244,160B)
- mutates: —
- stages (name — emitted ctx fields):
    0. `attn_norm` — rstd_attn
    1. `qkv_gate` — qm, km, gate, v
    2. `qknorm_rope` — rstd_q, rstd_k
    3. `attn` — lse, attn_out
    4. `gate_o` — xo
    5. `ffn_norm` — rstd_ffn
    6. `moe_route` — router_logits
    7. `moe_dispatch` — —
    8. `moe_experts13` — h13
    9. `moe_shared` — s13, gate_pre  ← derived recompute boundary
    10. `moe_experts2_combine` — —
- kernel calls:
    0. `rmsnorm_fwd`
    1. `mm ×3`
    2. `rmsnorm_fwd ×2`
    3. `rope_fwd ×2`
    4. `_scaled_dot_product_flash_attention`
    5. `addmm`
    6. `rmsnorm_fwd`
    7. `mm`
    8. `moe_topk_softmax`
    9. `moe_sort`
    10. `moe_dispatch_fwd`
    11. `moe_grouped_mm_fwd`
    12. `mm ×2`
    13. `swiglu_packed_fwd`
    14. `moe_grouped_mm_fwd`
    15. `swiglu_packed_fwd`
    16. `mm`
    17. `moe_scale_rows`
    18. `moe_combine_fwd`

### `head_loss` — `HeadLoss`

- example task: `head_loss_0_0`
- inputs: `y_0_0_39` (268,435,456B), `targets_0_0` (262,144B), `W_head` (1,017,122,816B)
- outputs: `dy_0_0_39` (268,435,456B), `loss_0_0` (4B), `dW_head_0` (1,017,122,816B)
- mutates: —
- kernel calls:
    0. `rmsnorm_fwd`
    1. `mm`
    2. `ce_loss_fwd_bwd`
    3. `mm ×2`
    4. `rmsnorm_bwd`

### `optimizer_head` — `AdamWStep`

- example task: `optimizer_head_0`
- inputs: `W_head` (1,017,122,816B), `dW_head_0` (1,017,122,816B), `O_head` (2,034,245,632B)
- outputs: —
- mutates: `W_head`, `O_head`
- kernel calls:
    0. `adamw_step ×2`

### `gattnmoe_bwd` — `Qwen35MoeAttnBlockBwd`

- example task: `block_bwd_0_0_39`
- inputs: `dy_0_0_39` (268,435,456B), `A_0_0_39` (3,264,348,160B), `y_0_0_38` (268,435,456B), `W_39` (1,672,492,032B), `M_0_0_39` (5,244,160B)
- outputs: `dy_0_0_38` (268,435,456B), `dW_0_39` (1,672,492,032B)
- mutates: —
- kernel calls:
    0. `rmsnorm_apply`
    1. `moe_dispatch_fwd ×2`
    2. `swiglu_packed_fwd`
    3. `moe_grouped_mm_dgrad`
    4. `moe_rowdot`
    5. `moe_scale_rows`
    6. `moe_grouped_mm_wgrad`
    7. `moe_scale_rows`
    8. `swiglu_packed_bwd`
    9. `moe_grouped_mm_wgrad`
    10. `moe_grouped_mm_dgrad`
    11. `moe_dispatch_bwd`
    12. `moe_router_bwd`
    13. `moe_aux_lb_grad`
    14. `mm`
    15. `swiglu_packed_fwd`
    16. `mm`
    17. `moe_rowdot`
    18. `moe_scale_rows`
    19. `mm ×3`
    20. `swiglu_packed_bwd`
    21. `mm`
    22. `rmsnorm_bwd`
    23. `mm ×2`
    24. `rmsnorm_apply ×2`
    25. `rope_fwd ×2`
    26. `_scaled_dot_product_flash_attention_backward`
    27. `rope_bwd ×2`
    28. `rmsnorm_bwd ×2`
    29. `rmsnorm_apply`
    30. `mm ×4`
    31. `rmsnorm_bwd`

### `optimizer_block` — `AdamWStep`

- example task: `optimizer_0_39`
- inputs: `W_39` (1,672,492,032B), `dW_0_39` (1,672,492,032B), `O_39` (3,344,984,064B)
- outputs: —
- mutates: `W_39`, `O_39`
- kernel calls:
    0. `adamw_step ×14`

### `linmoe_bwd` — `Qwen35MoeLinBlockBwd`

- example task: `block_bwd_0_0_38`
- inputs: `dy_0_0_38` (268,435,456B), `A_0_0_38` (3,951,689,728B), `y_0_0_37` (268,435,456B), `W_38` (1,685,402,368B), `M_0_0_38` (5,244,160B)
- outputs: `dy_0_0_37` (268,435,456B), `dW_0_38` (1,685,402,368B)
- mutates: —
- kernel calls:
    0. `rmsnorm_apply`
    1. `moe_dispatch_fwd ×2`
    2. `swiglu_packed_fwd`
    3. `moe_grouped_mm_dgrad`
    4. `moe_rowdot`
    5. `moe_scale_rows`
    6. `moe_grouped_mm_wgrad`
    7. `moe_scale_rows`
    8. `swiglu_packed_bwd`
    9. `moe_grouped_mm_wgrad`
    10. `moe_grouped_mm_dgrad`
    11. `moe_dispatch_bwd`
    12. `moe_router_bwd`
    13. `moe_aux_lb_grad`
    14. `mm`
    15. `swiglu_packed_fwd`
    16. `mm`
    17. `moe_rowdot`
    18. `moe_scale_rows`
    19. `mm ×3`
    20. `swiglu_packed_bwd`
    21. `mm`
    22. `rmsnorm_bwd`
    23. `mm`
    24. `gated_rmsnorm_bwd`
    25. `mm`
    26. `causal_conv1d_silu_fwd`
    27. `fla::l2norm_fwd ×2`
    28. `fla::chunk_gated_delta_rule_bwd`
    29. `fla::l2norm_bwd ×2`
    30. `causal_conv1d_silu_bwd`
    31. `rmsnorm_apply`
    32. `mm ×3`
    33. `rmsnorm_bwd`

### `embed_bwd` — `EmbedBwd`

- example task: `embed_bwd_0_0`
- inputs: `dy_embed_0_0` (268,435,456B), `tokens_0_0` (262,144B)
- outputs: `dW_embed_0` (1,017,118,720B)
- mutates: —
- kernel calls:
    0. `embed_bwd_accum`

### `optimizer_embed` — `AdamWStep`

- example task: `optimizer_embed_0`
- inputs: `W_embed` (1,017,118,720B), `dW_embed_0` (1,017,118,720B), `O_embed` (2,034,237,440B)
- outputs: —
- mutates: `W_embed`, `O_embed`
- kernel calls:
    0. `adamw_step`

