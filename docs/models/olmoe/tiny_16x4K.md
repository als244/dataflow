# olmoe / `tiny` @ 16x4K: tasks, objects, kernels

GENERATED from `ShapedOlmoeConfig.tiny()` at run shape microbatch 16 × seq 4096 — regenerate with `python tools/gen_model_page.py --preset tiny --microbatch 16 --seq-len 4096`. All presets: [builtin_models.md](../../builtin_models.md); task-kind fleet index: [task_kinds.md](../../task_kinds.md).

Layer kinds (2 layers): `block block`

**Run shape**: microbatch 16 × seq_len 4096 = **65,536 tokens per round** (× 1 grad-accum round(s) per step). `A_*`/`M_*` objects are sized per round; bytes/token figures transfer to any run shape.

## Object summary

At this run shape (65,536 tokens/round). Token-scaled objects show bytes/token in parens. Details per kind below.

| object | scope | bytes |
|---|---|---|
| `W_i (block)` | layer | 920,576 |
| `dW_i (block)` | layer/step | 920,576 |
| `O_i (block)` | layer | 1,841,152 |
| `A (block)` | layer × round | 154,140,672 (2,352.0/token) |
| `M (block)` | layer × round | 1,310,976 (20.0/token) |
| `W_head` | run | 131,328 |
| `W_embed` | run | 131,072 |
| `O_embed` | run | 262,144 |
| `O_head` | run | 262,656 |
| `hidden state (y)` | boundary buffer | 16,777,216 (256.0/token) |

### Aggregate totals (all layers, this run shape)

| type | objects | total bytes |
|---|---|---|
| W (all weights, incl. embed/head) | 4 | 2,103,552 |
| dW (all gradients, per step) | 4 | 2,103,552 |
| O (all optimizer state) | 4 | 4,207,104 |
| A (all saved activations, one round) | 2 | 308,281,344 (4,704.0/token) |
| M (all metadata, one round) | 2 | 2,621,952 (40.0/token) |

## Dims

| field | value |
|---|---|
| `d_model` | 128 |
| `n_heads` | 4 |
| `n_kv_heads` | 4 |
| `head_dim` | 32 |
| `d_ff` | 128 |
| `vocab_size` | 512 |
| `tokens` | 65536 |
| `seq_len` | 4096 |
| `rope_base` | 10000.0 |
| `opt_policy` | adamw |

## Objects, per layer kind

`dW_i` mirrors `W_i`'s fields at the grad dtypes; `O_i` holds the optimizer policy's state slots per field (adamw default: `m_f`+`v_f` at the opt dtype; sgd fields contribute none — see extending.md §6). `A_i`/`M_i` exist per (step, round).

### kind `block` (e.g. layer 0)

**`W_0` weights** — 920,576 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (128,) | 256 |
| `wq` | bf16 | (128, 128) | 32,768 |
| `wk` | bf16 | (128, 128) | 32,768 |
| `wv` | bf16 | (128, 128) | 32,768 |
| `q_norm_w` | bf16 | (128,) | 256 |
| `k_norm_w` | bf16 | (128,) | 256 |
| `wo` | bf16 | (128, 128) | 32,768 |
| `ffn_norm_w` | bf16 | (128,) | 256 |
| `w_router` | bf16 | (128, 8) | 2,048 |
| `w13_experts` | bf16 | (8, 128, 256) | 524,288 |
| `w2_experts` | bf16 | (8, 128, 128) | 262,144 |

**`A_.._0` saved context** — 154,140,672 bytes = **2,352.0 bytes/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 262,144 |
| `qm` | bf16 | (65536, 128) | 16,777,216 |
| `km` | bf16 | (65536, 128) | 16,777,216 |
| `rstd_q` | fp32 | (65536,) | 262,144 |
| `rstd_k` | fp32 | (65536,) | 262,144 |
| `v` | bf16 | (65536, 128) | 16,777,216 |
| `lse` | fp32 | (64, 4096) | 1,048,576 |
| `attn_out` | bf16 | (65536, 128) | 16,777,216 |
| `h_mid` | bf16 | (65536, 128) | 16,777,216 |
| `rstd_ffn` | fp32 | (65536,) | 262,144 |
| `router_logits` | bf16 | (65536, 8) | 1,048,576 |
| `h13` | bf16 | (131072, 256) | 67,108,864 |

**`M_.._0` metadata** — 1,310,976 bytes = **20.0 bytes/token** (never recomputed)

| field | dtype | shape | bytes |
|---|---|---|---|
| `route_w` | bf16 | (65536, 2) | 262,144 |
| `route_ids` | int32 | (65536, 2) | 524,288 |
| `route_order` | int32 | (131072,) | 524,288 |
| `route_offsets` | int32 | (9,) | 36 |

**`W_head`** — 131,328 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `w` | bf16 | (512, 128) | 131,072 |
| `final_norm_w` | bf16 | (128,) | 256 |

## Tasks

### `embed_fwd` — `EmbedFwd`

- example task: `embed_fwd_0_0`
- inputs: `tokens_0_0` (262,144B), `W_embed` (131,072B)
- outputs: `y_embed_0_0` (16,777,216B)
- mutates: —
- kernel calls:
    0. `index_select`

### `moeattn_fwd` — `OlmoeBlockFwd`

- example task: `block_fwd_0_0_0`
- inputs: `y_embed_0_0` (16,777,216B), `W_0` (920,576B)
- outputs: `y_0_0_0` (16,777,216B), `A_0_0_0` (154,140,672B), `M_0_0_0` (1,310,976B)
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
- inputs: `y_0_0_1` (16,777,216B), `targets_0_0` (262,144B), `W_head` (131,328B)
- outputs: `dy_0_0_1` (16,777,216B), `loss_0_0` (4B), `dW_head_0` (131,328B)
- mutates: —
- kernel calls:
    0. `rmsnorm_fwd`
    1. `mm`
    2. `ce_loss_fwd_bwd`
    3. `mm ×2`
    4. `rmsnorm_bwd`

### `optimizer_head` — `AdamWStep`

- example task: `optimizer_head_0`
- inputs: `W_head` (131,328B), `dW_head_0` (131,328B), `O_head` (262,656B)
- outputs: —
- mutates: `W_head`, `O_head`
- kernel calls:
    0. `adamw_step ×2`

### `moeattn_bwd` — `OlmoeBlockBwd`

- example task: `block_bwd_0_0_1`
- inputs: `dy_0_0_1` (16,777,216B), `A_0_0_1` (154,140,672B), `y_0_0_0` (16,777,216B), `W_1` (920,576B), `M_0_0_1` (1,310,976B)
- outputs: `dy_0_0_0` (16,777,216B), `dW_0_1` (920,576B)
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

- example task: `optimizer_0_1`
- inputs: `W_1` (920,576B), `dW_0_1` (920,576B), `O_1` (1,841,152B)
- outputs: —
- mutates: `W_1`, `O_1`
- kernel calls:
    0. `adamw_step ×11`

### `embed_bwd` — `EmbedBwd`

- example task: `embed_bwd_0_0`
- inputs: `dy_embed_0_0` (16,777,216B), `tokens_0_0` (262,144B)
- outputs: `dW_embed_0` (131,072B)
- mutates: —
- kernel calls:
    0. `embed_bwd_accum`

### `optimizer_embed` — `AdamWStep`

- example task: `optimizer_embed_0`
- inputs: `W_embed` (131,072B), `dW_embed_0` (131,072B), `O_embed` (262,144B)
- outputs: —
- mutates: `W_embed`, `O_embed`
- kernel calls:
    0. `adamw_step`

