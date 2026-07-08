# olmoe / `olmoe_7b` @ 16x4K: tasks, objects, kernels

GENERATED from `ShapedOlmoeConfig.olmoe_7b()` at run shape microbatch 16 × seq 4096 — regenerate with `python tools/gen_model_page.py --preset olmoe_7b --microbatch 16 --seq-len 4096`. All presets: [builtin_models.md](../../builtin_models.md); task-kind fleet index: [task_kinds.md](../../task_kinds.md).

Layer kinds (16 layers): `block block block block block block block block block block block block block block block block`

**Run shape**: microbatch 16 × seq_len 4096 = **65,536 tokens per round** (× 1 grad-accum round(s) per step). `A_*`/`M_*` objects are sized per round; bytes/token figures transfer to any run shape.

## Object summary

At this run shape (65,536 tokens/round). Token-scaled objects show bytes/token in parens. Details per kind below.

| object | scope | bytes |
|---|---|---|
| `W_i (block)` | layer | 839,139,328 |
| `dW_i (block)` | layer/step | 839,139,328 |
| `O_i (block)` | layer | 1,678,278,656 |
| `A (block)` | layer × round | 3,503,292,416 (53,456.0/token) |
| `M (block)` | layer × round | 5,243,392 (80.0/token) |
| `W_head` | run | 206,049,280 |
| `W_embed` | run | 206,045,184 |
| `O_embed` | run | 412,090,368 |
| `O_head` | run | 412,098,560 |
| `hidden state (y)` | boundary buffer | 268,435,456 (4,096.0/token) |

### Aggregate totals (all layers, this run shape)

| type | objects | total bytes |
|---|---|---|
| W (all weights, incl. embed/head) | 18 | 13,838,323,712 |
| dW (all gradients, per step) | 18 | 13,838,323,712 |
| O (all optimizer state) | 18 | 27,676,647,424 |
| A (all saved activations, one round) | 16 | 56,052,678,656 (855,296.0/token) |
| M (all metadata, one round) | 16 | 83,894,272 (1,280.1/token) |

## Dims

| field | value |
|---|---|
| `d_model` | 2048 |
| `n_heads` | 16 |
| `n_kv_heads` | 16 |
| `head_dim` | 128 |
| `d_ff` | 1024 |
| `vocab_size` | 50304 |
| `tokens` | 65536 |
| `seq_len` | 4096 |
| `rope_base` | 10000.0 |
| `opt_policy` | adamw |

## Objects, per layer kind

`dW_i` mirrors `W_i`'s fields at the grad dtypes; `O_i` holds the optimizer policy's state slots per field (adamw default: `m_f`+`v_f` at the opt dtype; sgd fields contribute none — see extending.md §6). `A_i`/`M_i` exist per (step, round).

### kind `block` (e.g. layer 0)

**`W_0` weights** — 839,139,328 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (2048,) | 4,096 |
| `wq` | bf16 | (2048, 2048) | 8,388,608 |
| `wk` | bf16 | (2048, 2048) | 8,388,608 |
| `wv` | bf16 | (2048, 2048) | 8,388,608 |
| `q_norm_w` | bf16 | (2048,) | 4,096 |
| `k_norm_w` | bf16 | (2048,) | 4,096 |
| `wo` | bf16 | (2048, 2048) | 8,388,608 |
| `ffn_norm_w` | bf16 | (2048,) | 4,096 |
| `w_router` | bf16 | (2048, 64) | 262,144 |
| `w13_experts` | bf16 | (64, 2048, 2048) | 536,870,912 |
| `w2_experts` | bf16 | (64, 1024, 2048) | 268,435,456 |

**`A_.._0` saved context** — 3,503,292,416 bytes = **53,456.0 bytes/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 262,144 |
| `qm` | bf16 | (65536, 2048) | 268,435,456 |
| `km` | bf16 | (65536, 2048) | 268,435,456 |
| `rstd_q` | fp32 | (65536,) | 262,144 |
| `rstd_k` | fp32 | (65536,) | 262,144 |
| `v` | bf16 | (65536, 2048) | 268,435,456 |
| `lse` | fp32 | (256, 4096) | 4,194,304 |
| `attn_out` | bf16 | (65536, 2048) | 268,435,456 |
| `h_mid` | bf16 | (65536, 2048) | 268,435,456 |
| `rstd_ffn` | fp32 | (65536,) | 262,144 |
| `router_logits` | bf16 | (65536, 64) | 8,388,608 |
| `h13` | bf16 | (524288, 2048) | 2,147,483,648 |

**`M_.._0` metadata** — 5,243,392 bytes = **80.0 bytes/token** (never recomputed)

| field | dtype | shape | bytes |
|---|---|---|---|
| `route_w` | bf16 | (65536, 8) | 1,048,576 |
| `route_ids` | int32 | (65536, 8) | 2,097,152 |
| `route_order` | int32 | (524288,) | 2,097,152 |
| `route_offsets` | int32 | (65,) | 260 |

**`W_head`** — 206,049,280 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `w` | bf16 | (50304, 2048) | 206,045,184 |
| `final_norm_w` | bf16 | (2048,) | 4,096 |

## Tasks

### `embed_fwd` — `EmbedFwd`

- example task: `embed_fwd_0_0`
- inputs: `tokens_0_0` (262,144B), `W_embed` (206,045,184B)
- outputs: `y_embed_0_0` (268,435,456B)
- mutates: —
- kernel calls:
    0. `index_select`

### `moeattn_fwd` — `OlmoeBlockFwd`

- example task: `block_fwd_0_0_0`
- inputs: `y_embed_0_0` (268,435,456B), `W_0` (839,139,328B)
- outputs: `y_0_0_0` (268,435,456B), `A_0_0_0` (3,503,292,416B), `M_0_0_0` (5,243,392B)
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
- kernel calls, by stage:
    - `attn_norm`: `rmsnorm_fwd`
    - `qkv_qknorm`: `mm ×3`, `rmsnorm_fwd ×2`
    - `rope`: `rope_fwd ×2`
    - `attn`: `_scaled_dot_product_flash_attention`
    - `resid1_norm2`: `addmm`, `rmsnorm_fwd`
    - `moe_route`: `mm`, `moe_topk_softmax`
    - `moe_dispatch`: `moe_sort`, `moe_dispatch_fwd`
    - `moe_experts13`: `moe_grouped_mm_fwd`
    - `moe_experts2_combine`: `swiglu_packed_fwd`, `moe_grouped_mm_fwd`, `moe_combine_fwd`

### `head_loss` — `HeadLoss`

- example task: `head_loss_0_0`
- inputs: `y_0_0_15` (268,435,456B), `targets_0_0` (262,144B), `W_head` (206,049,280B)
- outputs: `dy_0_0_15` (268,435,456B), `loss_0_0` (4B), `dW_head_0` (206,049,280B)
- mutates: —
- kernel calls:
    0. `rmsnorm_fwd`
    1. `mm`
    2. `ce_loss_fwd_bwd`
    3. `mm ×2`
    4. `rmsnorm_bwd`

### `optimizer_head` — `AdamWStep`

- example task: `optimizer_head_0`
- inputs: `W_head` (206,049,280B), `dW_head_0` (206,049,280B), `O_head` (412,098,560B)
- outputs: —
- mutates: `W_head`, `O_head`
- kernel calls:
    0. `adamw_step ×2`

### `moeattn_bwd` — `OlmoeBlockBwd`

- example task: `block_bwd_0_0_15`
- inputs: `dy_0_0_15` (268,435,456B), `A_0_0_15` (3,503,292,416B), `y_0_0_14` (268,435,456B), `W_15` (839,139,328B), `M_0_0_15` (5,243,392B)
- outputs: `dy_0_0_14` (268,435,456B), `dW_0_15` (839,139,328B)
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

- example task: `optimizer_0_15`
- inputs: `W_15` (839,139,328B), `dW_0_15` (839,139,328B), `O_15` (1,678,278,656B)
- outputs: —
- mutates: `W_15`, `O_15`
- kernel calls:
    0. `adamw_step ×11`

### `embed_bwd` — `EmbedBwd`

- example task: `embed_bwd_0_0`
- inputs: `dy_embed_0_0` (268,435,456B), `tokens_0_0` (262,144B)
- outputs: `dW_embed_0` (206,045,184B)
- mutates: —
- kernel calls:
    0. `embed_bwd_accum`

### `optimizer_embed` — `AdamWStep`

- example task: `optimizer_embed_0`
- inputs: `W_embed` (206,045,184B), `dW_embed_0` (206,045,184B), `O_embed` (412,090,368B)
- outputs: —
- mutates: `W_embed`, `O_embed`
- kernel calls:
    0. `adamw_step`

