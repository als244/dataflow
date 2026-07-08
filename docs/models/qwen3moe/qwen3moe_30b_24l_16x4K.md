# qwen3moe / `qwen3moe_30b_24l` @ 16x4K: tasks, objects, kernels

GENERATED from `ShapedQwen3MoeConfig.qwen3moe_30b_24l()` at run shape microbatch 16 × seq 4096 — regenerate with `python tools/gen_model_page.py --preset qwen3moe_30b_24l --microbatch 16 --seq-len 4096`. All presets: [builtin_models.md](../../builtin_models.md); task-kind fleet index: [task_kinds.md](../../task_kinds.md).

Layer kinds (24 layers): `block block block block block block block block block block block block block block block block block block block block block block block block`

**Run shape**: microbatch 16 × seq_len 4096 = **65,536 tokens per round** (× 1 grad-accum round(s) per step). `A_*`/`M_*` objects are sized per round; bytes/token figures transfer to any run shape.

## Object summary

At this run shape (65,536 tokens/round). Token-scaled objects show bytes/token in parens. Details per kind below.

| object | scope | bytes |
|---|---|---|
| `W_i (block)` | layer | 1,246,241,280 |
| `dW_i (block)` | layer/step | 1,246,241,280 |
| `O_i (block)` | layer | 2,492,482,560 |
| `A (block)` | layer × round | 3,122,135,040 (47,640.0/token) |
| `M (block)` | layer × round | 5,243,648 (80.0/token) |
| `W_head` | run | 622,333,952 |
| `W_embed` | run | 622,329,856 |
| `O_embed` | run | 1,244,659,712 |
| `O_head` | run | 1,244,667,904 |
| `hidden state (y)` | boundary buffer | 268,435,456 (4,096.0/token) |

### Aggregate totals (all layers, this run shape)

| type | objects | total bytes |
|---|---|---|
| W (all weights, incl. embed/head) | 26 | 31,154,454,528 |
| dW (all gradients, per step) | 26 | 31,154,454,528 |
| O (all optimizer state) | 26 | 62,308,909,056 |
| A (all saved activations, one round) | 24 | 74,931,240,960 (1,143,360.0/token) |
| M (all metadata, one round) | 24 | 125,847,552 (1,920.3/token) |

## Dims

| field | value |
|---|---|
| `d_model` | 2048 |
| `n_heads` | 32 |
| `n_kv_heads` | 4 |
| `head_dim` | 128 |
| `d_ff` | 768 |
| `vocab_size` | 151936 |
| `tokens` | 65536 |
| `seq_len` | 4096 |
| `rope_base` | 1000000.0 |
| `opt_policy` | adamw |

## Objects, per layer kind

`dW_i` mirrors `W_i`'s fields at the grad dtypes; `O_i` holds the optimizer policy's state slots per field (adamw default: `m_f`+`v_f` at the opt dtype; sgd fields contribute none — see extending.md §6). `A_i`/`M_i` exist per (step, round).

### kind `block` (e.g. layer 0)

**`W_0` weights** — 1,246,241,280 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (2048,) | 4,096 |
| `wq` | bf16 | (2048, 4096) | 16,777,216 |
| `wk` | bf16 | (2048, 512) | 2,097,152 |
| `wv` | bf16 | (2048, 512) | 2,097,152 |
| `q_norm_w` | bf16 | (128,) | 256 |
| `k_norm_w` | bf16 | (128,) | 256 |
| `wo` | bf16 | (4096, 2048) | 16,777,216 |
| `ffn_norm_w` | bf16 | (2048,) | 4,096 |
| `w_router` | bf16 | (2048, 128) | 524,288 |
| `w13_experts` | bf16 | (128, 2048, 1536) | 805,306,368 |
| `w2_experts` | bf16 | (128, 768, 2048) | 402,653,184 |

**`A_.._0` saved context** — 3,122,135,040 bytes = **47,640.0 bytes/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 262,144 |
| `qm` | bf16 | (65536, 4096) | 536,870,912 |
| `km` | bf16 | (65536, 512) | 67,108,864 |
| `rstd_q` | fp32 | (2097152,) | 8,388,608 |
| `rstd_k` | fp32 | (262144,) | 1,048,576 |
| `v` | bf16 | (65536, 512) | 67,108,864 |
| `lse` | fp32 | (512, 4096) | 8,388,608 |
| `attn_out` | bf16 | (65536, 4096) | 536,870,912 |
| `h_mid` | bf16 | (65536, 2048) | 268,435,456 |
| `rstd_ffn` | fp32 | (65536,) | 262,144 |
| `router_logits` | bf16 | (65536, 128) | 16,777,216 |
| `h13` | bf16 | (524288, 1536) | 1,610,612,736 |

**`M_.._0` metadata** — 5,243,648 bytes = **80.0 bytes/token** (never recomputed)

| field | dtype | shape | bytes |
|---|---|---|---|
| `route_w` | bf16 | (65536, 8) | 1,048,576 |
| `route_ids` | int32 | (65536, 8) | 2,097,152 |
| `route_order` | int32 | (524288,) | 2,097,152 |
| `route_offsets` | int32 | (129,) | 516 |

**`W_head`** — 622,333,952 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `w` | bf16 | (151936, 2048) | 622,329,856 |
| `final_norm_w` | bf16 | (2048,) | 4,096 |

## Tasks

### `embed_fwd` — `EmbedFwd`

- example task: `embed_fwd_0_0`
- inputs: `tokens_0_0` (262,144B), `W_embed` (622,329,856B)
- outputs: `y_embed_0_0` (268,435,456B)
- mutates: —
- kernel calls:
    0. `index_select`

### `q3moeattn_fwd` — `Qwen3MoeBlockFwd`

- example task: `block_fwd_0_0_0`
- inputs: `y_embed_0_0` (268,435,456B), `W_0` (1,246,241,280B)
- outputs: `y_0_0_0` (268,435,456B), `A_0_0_0` (3,122,135,040B), `M_0_0_0` (5,243,648B)
- mutates: —
- stages (name — emitted ctx fields):
    0. `attn_norm` — rstd_attn
    1. `qkv_qknorm` — qm, km, rstd_q, rstd_k, v
    2. `rope` — —
    3. `attn` — lse, attn_out
    4. `resid1_norm2` — h_mid, rstd_ffn
    5. `moe_route` — router_logits
    6. `moe_dispatch` — —
    7. `moe_experts13` — h13  ← derived recompute boundary
    8. `moe_experts2_combine` — —
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
    12. `swiglu_packed_fwd`
    13. `moe_grouped_mm_fwd`
    14. `moe_combine_fwd`

### `head_loss` — `HeadLoss`

- example task: `head_loss_0_0`
- inputs: `y_0_0_23` (268,435,456B), `targets_0_0` (262,144B), `W_head` (622,333,952B)
- outputs: `dy_0_0_23` (268,435,456B), `loss_0_0` (4B), `dW_head_0` (622,333,952B)
- mutates: —
- kernel calls:
    0. `rmsnorm_fwd`
    1. `mm`
    2. `ce_loss_fwd_bwd`
    3. `mm ×2`
    4. `rmsnorm_bwd`

### `optimizer_head` — `AdamWStep`

- example task: `optimizer_head_0`
- inputs: `W_head` (622,333,952B), `dW_head_0` (622,333,952B), `O_head` (1,244,667,904B)
- outputs: —
- mutates: `W_head`, `O_head`
- kernel calls:
    0. `adamw_step ×2`

### `q3moeattn_bwd` — `Qwen3MoeBlockBwd`

- example task: `block_bwd_0_0_23`
- inputs: `dy_0_0_23` (268,435,456B), `A_0_0_23` (3,122,135,040B), `y_0_0_22` (268,435,456B), `W_23` (1,246,241,280B), `M_0_0_23` (5,243,648B)
- outputs: `dy_0_0_22` (268,435,456B), `dW_0_23` (1,246,241,280B)
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
    15. `rmsnorm_bwd`
    16. `mm ×2`
    17. `rmsnorm_apply ×2`
    18. `rope_fwd ×2`
    19. `_scaled_dot_product_flash_attention_backward`
    20. `rope_bwd ×2`
    21. `rmsnorm_bwd ×2`
    22. `rmsnorm_apply`
    23. `mm ×4`
    24. `rmsnorm_bwd`

### `optimizer_block` — `AdamWStep`

- example task: `optimizer_0_23`
- inputs: `W_23` (1,246,241,280B), `dW_0_23` (1,246,241,280B), `O_23` (2,492,482,560B)
- outputs: —
- mutates: `W_23`, `O_23`
- kernel calls:
    0. `adamw_step ×11`

### `embed_bwd` — `EmbedBwd`

- example task: `embed_bwd_0_0`
- inputs: `dy_embed_0_0` (268,435,456B), `tokens_0_0` (262,144B)
- outputs: `dW_embed_0` (622,329,856B)
- mutates: —
- kernel calls:
    0. `embed_bwd_accum`

### `optimizer_embed` — `AdamWStep`

- example task: `optimizer_embed_0`
- inputs: `W_embed` (622,329,856B), `dW_embed_0` (622,329,856B), `O_embed` (1,244,659,712B)
- outputs: —
- mutates: `W_embed`, `O_embed`
- kernel calls:
    0. `adamw_step`

