# olmoe / `tiny` @ 16x4K: tasks, objects, kernels

GENERATED from `ShapedOlmoeConfig.tiny()` at run shape microbatch 16 × seq 4096 — regenerate with `python tools/gen_model_page.py --preset tiny --microbatch 16 --seq-len 4096`. All presets: [builtin_models.md](../../builtin_models.md); task-kind fleet index: [task_kinds.md](../../task_kinds.md).

Layer kinds (2 layers): `block block`

**Run shape**: microbatch 16 × seq_len 4096 = **65,536 tokens per round** (× 1 grad-accum round(s) per step). `A_*`/`M_*` objects are sized per round; bytes/token figures transfer to any run shape.

## Object summary

At this run shape (65,536 tokens/round). Token-scaled objects show per-token size in parens. Details per kind below.

| object | scope | size |
|---|---|---|
| `W_i (block)` | layer | 899.00 KiB |
| `dW_i (block)` | layer/step | 899.00 KiB |
| `O_i (block)` | layer | 1.76 MiB |
| `A (block)` | layer × round | 147.00 MiB (2.30 KiB/token) |
| `M (block)` | layer × round | 1.25 MiB (20.0 B/token) |
| `W_head` | run | 128.25 KiB |
| `W_embed` | run | 128.00 KiB |
| `O_embed` | run | 256.00 KiB |
| `O_head` | run | 256.50 KiB |
| `hidden state (y)` | boundary buffer | 16.00 MiB (256.0 B/token) |

### Aggregate totals (all layers, this run shape)

| type | objects | total size |
|---|---|---|
| W (all weights, incl. embed/head) | 4 | 2.01 MiB |
| dW (all gradients, per step) | 4 | 2.01 MiB |
| O (all optimizer state) | 4 | 4.01 MiB |
| A (all saved activations, one round) | 2 | 294.00 MiB (4.59 KiB/token) |
| M (all metadata, one round) | 2 | 2.50 MiB (40.0 B/token) |

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

**`W_0` weights** — 899.00 KiB

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (128,) | 256 B |
| `wq` | bf16 | (128, 128) | 32.00 KiB |
| `wk` | bf16 | (128, 128) | 32.00 KiB |
| `wv` | bf16 | (128, 128) | 32.00 KiB |
| `q_norm_w` | bf16 | (128,) | 256 B |
| `k_norm_w` | bf16 | (128,) | 256 B |
| `wo` | bf16 | (128, 128) | 32.00 KiB |
| `ffn_norm_w` | bf16 | (128,) | 256 B |
| `w_router` | bf16 | (128, 8) | 2.00 KiB |
| `w13_experts` | bf16 | (8, 128, 256) | 512.00 KiB |
| `w2_experts` | bf16 | (8, 128, 128) | 256.00 KiB |

**`A_.._0` saved context** — 147.00 MiB = **2.30 KiB/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 256.00 KiB |
| `qm` | bf16 | (65536, 128) | 16.00 MiB |
| `km` | bf16 | (65536, 128) | 16.00 MiB |
| `rstd_q` | fp32 | (65536,) | 256.00 KiB |
| `rstd_k` | fp32 | (65536,) | 256.00 KiB |
| `v` | bf16 | (65536, 128) | 16.00 MiB |
| `lse` | fp32 | (4, 65536) | 1.00 MiB |
| `attn_out` | bf16 | (65536, 128) | 16.00 MiB |
| `h_mid` | bf16 | (65536, 128) | 16.00 MiB |
| `rstd_ffn` | fp32 | (65536,) | 256.00 KiB |
| `router_logits` | bf16 | (65536, 8) | 1.00 MiB |
| `h13` | bf16 | (131072, 256) | 64.00 MiB |

**`M_.._0` metadata** — 1.25 MiB = **20.0 B/token** (never recomputed)

| field | dtype | shape | bytes |
|---|---|---|---|
| `route_w` | bf16 | (65536, 2) | 256.00 KiB |
| `route_ids` | int32 | (65536, 2) | 512.00 KiB |
| `route_order` | int32 | (131072,) | 512.00 KiB |
| `route_offsets` | int32 | (9,) | 36 B |

**`W_head`** — 128.25 KiB

| field | dtype | shape | bytes |
|---|---|---|---|
| `w` | bf16 | (512, 128) | 128.00 KiB |
| `final_norm_w` | bf16 | (128,) | 256 B |

## Tasks

### `prologue_round` — `RoundPrologue`

- example task: `prologue_round_0_0`
- inputs: `Aux_0` (512 B), `Aux_1` (512 B)
- outputs: `current_round_0_0` (4 B)
- mutates: `Aux_0`, `Aux_1`

### `embed_fwd` — `EmbedFwd`

- example task: `embed_fwd_0_0`
- inputs: `tokens_0_0` (256.00 KiB), `W_embed` (128.00 KiB)
- outputs: `y_embed_0_0` (16.00 MiB)
- mutates: —
- kernel calls:
    0. `index_select`

### `moeattn_fwd` — `OlmoeBlockFwd`

- example task: `block_fwd_0_0_0`
- inputs: `y_embed_0_0` (16.00 MiB), `W_0` (899.00 KiB), `current_round_0_0` (4 B), `Aux_0` (512 B)
- outputs: `y_0_0_0` (16.00 MiB), `A_0_0_0` (147.00 MiB), `AuxTemp_0_0_0` (1.25 MiB)
- mutates: `Aux_0`
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
    - `resid1_norm2`:
        4. `addmm`
        5. `rmsnorm_fwd`
    - `moe_route`:
        6. `mm`
        7. `moe_topk_softmax`
        8. `scatter_add_ ×2`
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
- inputs: `y_0_0_1` (16.00 MiB), `targets_0_0` (256.00 KiB), `W_head` (128.25 KiB)
- outputs: `dy_0_0_1` (16.00 MiB), `loss_0_0` (4 B), `dW_head_0` (128.25 KiB)
- mutates: —
- kernel calls:
    0. `rmsnorm_fwd`
    1. `mm`
    2. `ce_loss_fwd_bwd`
    3. `mm ×2`
    4. `rmsnorm_bwd`

### `optimizer_head` — `AdamWStep`

- example task: `optimizer_head_0`
- inputs: `W_head` (128.25 KiB), `dW_head_0` (128.25 KiB), `O_head` (256.50 KiB)
- outputs: —
- mutates: `W_head`, `O_head`
- kernel calls:
    0. `adamw_step ×2`

### `moeattn_bwd` — `OlmoeBlockBwd`

- example task: `block_bwd_0_0_1`
- inputs: `dy_0_0_1` (16.00 MiB), `A_0_0_1` (147.00 MiB), `y_0_0_0` (16.00 MiB), `W_1` (899.00 KiB), `AuxTemp_0_0_1` (1.25 MiB), `Aux_1` (512 B)
- outputs: `dy_0_0_0` (16.00 MiB), `dW_0_1` (899.00 KiB)
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
    19. `_flash_attention_backward`
    20. `rope_bwd ×2`
    21. `rmsnorm_bwd ×2`
    22. `rmsnorm_apply`
    23. `mm ×4`
    24. `rmsnorm_bwd`

### `optimizer_block` — `AdamWStep`

- example task: `optimizer_0_1`
- inputs: `W_1` (899.00 KiB), `dW_0_1` (899.00 KiB), `O_1` (1.76 MiB)
- outputs: —
- mutates: `W_1`, `O_1`
- kernel calls:
    0. `adamw_step ×11`

### `embed_bwd` — `EmbedBwd`

- example task: `embed_bwd_0_0`
- inputs: `dy_embed_0_0` (16.00 MiB), `tokens_0_0` (256.00 KiB)
- outputs: `dW_embed_0` (128.00 KiB)
- mutates: —
- kernel calls:
    0. `embed_bwd_accum`

### `optimizer_embed` — `AdamWStep`

- example task: `optimizer_embed_0`
- inputs: `W_embed` (128.00 KiB), `dW_embed_0` (128.00 KiB), `O_embed` (256.00 KiB)
- outputs: —
- mutates: `W_embed`, `O_embed`
- kernel calls:
    0. `adamw_step`

