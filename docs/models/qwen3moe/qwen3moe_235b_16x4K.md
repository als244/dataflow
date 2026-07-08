# qwen3moe / `qwen3moe_235b` @ 16x4K: tasks, objects, kernels

GENERATED from `ShapedQwen3MoeConfig.qwen3moe_235b()` at run shape microbatch 16 × seq 4096 — regenerate with `python tools/gen_model_page.py --preset qwen3moe_235b --microbatch 16 --seq-len 4096`. All presets: [builtin_models.md](../../builtin_models.md); task-kind fleet index: [task_kinds.md](../../task_kinds.md).

Layer kinds (94 layers): `block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block`

**Run shape**: microbatch 16 × seq_len 4096 = **65,536 tokens per round** (× 1 grad-accum round(s) per step). `A_*`/`M_*` objects are sized per round; bytes/token figures transfer to any run shape.

## Object summary

At this run shape (65,536 tokens/round). Token-scaled objects show per-token size in parens. Details per kind below.

| object | scope | size |
|---|---|---|
| `W_i (block)` | layer | 4.63 GiB |
| `dW_i (block)` | layer/step | 4.63 GiB |
| `O_i (block)` | layer | 9.27 GiB |
| `A (block)` | layer × round | 5.67 GiB (90.77 KiB/token) |
| `M (block)` | layer × round | 5.00 MiB (80.0 B/token) |
| `W_head` | run | 1.16 GiB |
| `W_embed` | run | 1.16 GiB |
| `O_embed` | run | 2.32 GiB |
| `O_head` | run | 2.32 GiB |
| `hidden state (y)` | boundary buffer | 512.00 MiB (8.00 KiB/token) |

### Aggregate totals (all layers, this run shape)

| type | objects | total size |
|---|---|---|
| W (all weights, incl. embed/head) | 96 | 437.90 GiB |
| dW (all gradients, per step) | 96 | 437.90 GiB |
| O (all optimizer state) | 96 | 875.79 GiB |
| A (all saved activations, one round) | 94 | 533.29 GiB (8.33 MiB/token) |
| M (all metadata, one round) | 94 | 470.07 MiB (7.34 KiB/token) |

## Dims

| field | value |
|---|---|
| `d_model` | 4096 |
| `n_heads` | 64 |
| `n_kv_heads` | 4 |
| `head_dim` | 128 |
| `d_ff` | 1536 |
| `vocab_size` | 151936 |
| `tokens` | 65536 |
| `seq_len` | 4096 |
| `rope_base` | 1000000.0 |
| `opt_policy` | adamw |

## Objects, per layer kind

`dW_i` mirrors `W_i`'s fields at the grad dtypes; `O_i` holds the optimizer policy's state slots per field (adamw default: `m_f`+`v_f` at the opt dtype; sgd fields contribute none — see extending.md §6). `A_i`/`M_i` exist per (step, round).

### kind `block` (e.g. layer 0)

**`W_0` weights** — 4.63 GiB

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (4096,) | 8.00 KiB |
| `wq` | bf16 | (4096, 8192) | 64.00 MiB |
| `wk` | bf16 | (4096, 512) | 4.00 MiB |
| `wv` | bf16 | (4096, 512) | 4.00 MiB |
| `q_norm_w` | bf16 | (128,) | 256 B |
| `k_norm_w` | bf16 | (128,) | 256 B |
| `wo` | bf16 | (8192, 4096) | 64.00 MiB |
| `ffn_norm_w` | bf16 | (4096,) | 8.00 KiB |
| `w_router` | bf16 | (4096, 128) | 1.00 MiB |
| `w13_experts` | bf16 | (128, 4096, 3072) | 3.00 GiB |
| `w2_experts` | bf16 | (128, 1536, 4096) | 1.50 GiB |

**`A_.._0` saved context** — 5.67 GiB = **90.77 KiB/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 256.00 KiB |
| `qm` | bf16 | (65536, 8192) | 1.00 GiB |
| `km` | bf16 | (65536, 512) | 64.00 MiB |
| `rstd_q` | fp32 | (4194304,) | 16.00 MiB |
| `rstd_k` | fp32 | (262144,) | 1.00 MiB |
| `v` | bf16 | (65536, 512) | 64.00 MiB |
| `lse` | fp32 | (1024, 4096) | 16.00 MiB |
| `attn_out` | bf16 | (65536, 8192) | 1.00 GiB |
| `h_mid` | bf16 | (65536, 4096) | 512.00 MiB |
| `rstd_ffn` | fp32 | (65536,) | 256.00 KiB |
| `router_logits` | bf16 | (65536, 128) | 16.00 MiB |
| `h13` | bf16 | (524288, 3072) | 3.00 GiB |

**`M_.._0` metadata** — 5.00 MiB = **80.0 B/token** (never recomputed)

| field | dtype | shape | bytes |
|---|---|---|---|
| `route_w` | bf16 | (65536, 8) | 1.00 MiB |
| `route_ids` | int32 | (65536, 8) | 2.00 MiB |
| `route_order` | int32 | (524288,) | 2.00 MiB |
| `route_offsets` | int32 | (129,) | 516 B |

**`W_head`** — 1.16 GiB

| field | dtype | shape | bytes |
|---|---|---|---|
| `w` | bf16 | (151936, 4096) | 1.16 GiB |
| `final_norm_w` | bf16 | (4096,) | 8.00 KiB |

## Tasks

### `embed_fwd` — `EmbedFwd`

- example task: `embed_fwd_0_0`
- inputs: `tokens_0_0` (256.00 KiB), `W_embed` (1.16 GiB)
- outputs: `y_embed_0_0` (512.00 MiB)
- mutates: —
- kernel calls:
    0. `index_select`

### `q3moeattn_fwd` — `Qwen3MoeBlockFwd`

- example task: `block_fwd_0_0_0`
- inputs: `y_embed_0_0` (512.00 MiB), `W_0` (4.63 GiB)
- outputs: `y_0_0_0` (512.00 MiB), `A_0_0_0` (5.67 GiB), `M_0_0_0` (5.00 MiB)
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
    - `moe_route`:
        7. `mm`
        8. `moe_topk_softmax`
    - `moe_dispatch`:
        9. `moe_sort`
        10. `moe_dispatch_fwd`
    - `moe_experts13`:
        11. `moe_grouped_mm_fwd`
    - `moe_experts2_combine`:
        12. `swiglu_packed_fwd`
        13. `moe_grouped_mm_fwd`
        14. `moe_combine_fwd`

### `head_loss` — `HeadLoss`

- example task: `head_loss_0_0`
- inputs: `y_0_0_93` (512.00 MiB), `targets_0_0` (256.00 KiB), `W_head` (1.16 GiB)
- outputs: `dy_0_0_93` (512.00 MiB), `loss_0_0` (4 B), `dW_head_0` (1.16 GiB)
- mutates: —
- kernel calls:
    0. `rmsnorm_fwd`
    1. `mm`
    2. `ce_loss_fwd_bwd`
    3. `mm ×2`
    4. `rmsnorm_bwd`

### `optimizer_head` — `AdamWStep`

- example task: `optimizer_head_0`
- inputs: `W_head` (1.16 GiB), `dW_head_0` (1.16 GiB), `O_head` (2.32 GiB)
- outputs: —
- mutates: `W_head`, `O_head`
- kernel calls:
    0. `adamw_step ×2`

### `q3moeattn_bwd` — `Qwen3MoeBlockBwd`

- example task: `block_bwd_0_0_93`
- inputs: `dy_0_0_93` (512.00 MiB), `A_0_0_93` (5.67 GiB), `y_0_0_92` (512.00 MiB), `W_93` (4.63 GiB), `M_0_0_93` (5.00 MiB)
- outputs: `dy_0_0_92` (512.00 MiB), `dW_0_93` (4.63 GiB)
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

- example task: `optimizer_0_93`
- inputs: `W_93` (4.63 GiB), `dW_0_93` (4.63 GiB), `O_93` (9.27 GiB)
- outputs: —
- mutates: `W_93`, `O_93`
- kernel calls:
    0. `adamw_step ×11`

### `embed_bwd` — `EmbedBwd`

- example task: `embed_bwd_0_0`
- inputs: `dy_embed_0_0` (512.00 MiB), `tokens_0_0` (256.00 KiB)
- outputs: `dW_embed_0` (1.16 GiB)
- mutates: —
- kernel calls:
    0. `embed_bwd_accum`

### `optimizer_embed` — `AdamWStep`

- example task: `optimizer_embed_0`
- inputs: `W_embed` (1.16 GiB), `dW_embed_0` (1.16 GiB), `O_embed` (2.32 GiB)
- outputs: —
- mutates: `W_embed`, `O_embed`
- kernel calls:
    0. `adamw_step`

