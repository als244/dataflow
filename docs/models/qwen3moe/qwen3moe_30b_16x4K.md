# qwen3moe / `qwen3moe_30b` @ 16x4K: tasks, objects, kernels

GENERATED from `ShapedQwen3MoeConfig.qwen3moe_30b()` at run shape microbatch 16 × seq 4096 — regenerate with `python tools/gen_model_docs/gen_model_page.py --preset qwen3moe_30b --microbatch 16 --seq-len 4096`. All presets: [builtin_models.md](../../builtin_models.md); task-kind fleet index: [task_kinds.md](../../task_kinds.md).

Layer kinds (48 layers): `block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block`

**Run shape**: microbatch 16 × seq_len 4096 = **65,536 tokens per round** (× 1 grad-accum round(s) per step). `A_*`/`AuxTemp_*` objects are sized per round; bytes/token figures transfer to any run shape.

## Object summary

At this run shape (65,536 tokens/round). Token-scaled objects show per-token size in parens. Details per kind below.

| object | scope | size |
|---|---|---|
| `W_i (block)` | layer | 1.16 GiB |
| `dW_i (block)` | layer/step | 1.16 GiB |
| `O_i (block)` | layer | 2.32 GiB |
| `A (block)` | layer × round | 2.91 GiB (46.52 KiB/token) |
| `M (block)` | layer × round | 5.00 MiB (80.0 B/token) |
| `W_head` | run | 593.50 MiB |
| `W_embed` | run | 593.50 MiB |
| `O_embed` | run | 1.16 GiB |
| `O_head` | run | 1.16 GiB |
| `hidden state (y)` | boundary buffer | 256.00 MiB (4.00 KiB/token) |

### Aggregate totals (all layers, this run shape)

| type | objects | total size |
|---|---|---|
| W (all weights, incl. embed/head) | 50 | 56.87 GiB |
| dW (all gradients, per step) | 50 | 56.87 GiB |
| O (all optimizer state) | 50 | 113.74 GiB |
| A (all saved activations, one round) | 48 | 139.57 GiB (2.18 MiB/token) |
| M (all metadata, one round) | 48 | 240.04 MiB (3.75 KiB/token) |

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

**`W_0` weights** — 1.16 GiB

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (2048,) | 4.00 KiB |
| `wq` | bf16 | (2048, 4096) | 16.00 MiB |
| `wk` | bf16 | (2048, 512) | 2.00 MiB |
| `wv` | bf16 | (2048, 512) | 2.00 MiB |
| `q_norm_w` | bf16 | (128,) | 256 B |
| `k_norm_w` | bf16 | (128,) | 256 B |
| `wo` | bf16 | (4096, 2048) | 16.00 MiB |
| `ffn_norm_w` | bf16 | (2048,) | 4.00 KiB |
| `w_router` | bf16 | (2048, 128) | 512.00 KiB |
| `w13_experts` | bf16 | (128, 2048, 1536) | 768.00 MiB |
| `w2_experts` | bf16 | (128, 768, 2048) | 384.00 MiB |

**`A_.._0` saved context** — 2.91 GiB = **46.52 KiB/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 256.00 KiB |
| `qm` | bf16 | (65536, 4096) | 512.00 MiB |
| `km` | bf16 | (65536, 512) | 64.00 MiB |
| `rstd_q` | fp32 | (2097152,) | 8.00 MiB |
| `rstd_k` | fp32 | (262144,) | 1.00 MiB |
| `v` | bf16 | (65536, 512) | 64.00 MiB |
| `lse` | fp32 | (32, 65536) | 8.00 MiB |
| `attn_out` | bf16 | (65536, 4096) | 512.00 MiB |
| `h_mid` | bf16 | (65536, 2048) | 256.00 MiB |
| `rstd_ffn` | fp32 | (65536,) | 256.00 KiB |
| `router_logits` | bf16 | (65536, 128) | 16.00 MiB |
| `h13` | bf16 | (524288, 1536) | 1.50 GiB |

**`M_.._0` metadata** — 5.00 MiB = **80.0 B/token** (never recomputed)

| field | dtype | shape | bytes |
|---|---|---|---|
| `route_w` | bf16 | (65536, 8) | 1.00 MiB |
| `route_ids` | int32 | (65536, 8) | 2.00 MiB |
| `route_order` | int32 | (524288,) | 2.00 MiB |
| `route_offsets` | int32 | (129,) | 516 B |

**`W_head`** — 593.50 MiB

| field | dtype | shape | bytes |
|---|---|---|---|
| `w` | bf16 | (151936, 2048) | 593.50 MiB |
| `final_norm_w` | bf16 | (2048,) | 4.00 KiB |

## Tasks

### `prologue_round` — `RoundPrologue`

- example task: `prologue_round_0_0`
- inputs: `Aux_0` (1.50 KiB), `Aux_1` (1.50 KiB), `Aux_2` (1.50 KiB), `Aux_3` (1.50 KiB), `Aux_4` (1.50 KiB), `Aux_5` (1.50 KiB), `Aux_6` (1.50 KiB), `Aux_7` (1.50 KiB), `Aux_8` (1.50 KiB), `Aux_9` (1.50 KiB), `Aux_10` (1.50 KiB), `Aux_11` (1.50 KiB), `Aux_12` (1.50 KiB), `Aux_13` (1.50 KiB), `Aux_14` (1.50 KiB), `Aux_15` (1.50 KiB), `Aux_16` (1.50 KiB), `Aux_17` (1.50 KiB), `Aux_18` (1.50 KiB), `Aux_19` (1.50 KiB), `Aux_20` (1.50 KiB), `Aux_21` (1.50 KiB), `Aux_22` (1.50 KiB), `Aux_23` (1.50 KiB), `Aux_24` (1.50 KiB), `Aux_25` (1.50 KiB), `Aux_26` (1.50 KiB), `Aux_27` (1.50 KiB), `Aux_28` (1.50 KiB), `Aux_29` (1.50 KiB), `Aux_30` (1.50 KiB), `Aux_31` (1.50 KiB), `Aux_32` (1.50 KiB), `Aux_33` (1.50 KiB), `Aux_34` (1.50 KiB), `Aux_35` (1.50 KiB), `Aux_36` (1.50 KiB), `Aux_37` (1.50 KiB), `Aux_38` (1.50 KiB), `Aux_39` (1.50 KiB), `Aux_40` (1.50 KiB), `Aux_41` (1.50 KiB), `Aux_42` (1.50 KiB), `Aux_43` (1.50 KiB), `Aux_44` (1.50 KiB), `Aux_45` (1.50 KiB), `Aux_46` (1.50 KiB), `Aux_47` (1.50 KiB)
- outputs: `current_round_0_0` (4 B)
- mutates: `Aux_0`, `Aux_1`, `Aux_2`, `Aux_3`, `Aux_4`, `Aux_5`, `Aux_6`, `Aux_7`, `Aux_8`, `Aux_9`, `Aux_10`, `Aux_11`, `Aux_12`, `Aux_13`, `Aux_14`, `Aux_15`, `Aux_16`, `Aux_17`, `Aux_18`, `Aux_19`, `Aux_20`, `Aux_21`, `Aux_22`, `Aux_23`, `Aux_24`, `Aux_25`, `Aux_26`, `Aux_27`, `Aux_28`, `Aux_29`, `Aux_30`, `Aux_31`, `Aux_32`, `Aux_33`, `Aux_34`, `Aux_35`, `Aux_36`, `Aux_37`, `Aux_38`, `Aux_39`, `Aux_40`, `Aux_41`, `Aux_42`, `Aux_43`, `Aux_44`, `Aux_45`, `Aux_46`, `Aux_47`

### `embed_fwd` — `EmbedFwd`

- example task: `embed_fwd_0_0`
- inputs: `tokens_0_0` (256.00 KiB), `W_embed` (593.50 MiB)
- outputs: `y_embed_0_0` (256.00 MiB)
- mutates: —
- kernel calls:
    0. `index_select`

### `q3moeattn_fwd` — `Qwen3MoeBlockFwd`

- example task: `block_fwd_0_0_0`
- inputs: `y_embed_0_0` (256.00 MiB), `W_0` (1.16 GiB), `current_round_0_0` (4 B), `Aux_0` (1.50 KiB)
- outputs: `y_0_0_0` (256.00 MiB), `A_0_0_0` (2.91 GiB), `AuxTemp_0_0_0` (5.00 MiB)
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
- inputs: `y_0_0_47` (256.00 MiB), `targets_0_0` (256.00 KiB), `W_head` (593.50 MiB)
- outputs: `dy_0_0_47` (256.00 MiB), `loss_0_0` (4 B), `dW_head_0` (593.50 MiB)
- mutates: —
- kernel calls:
    0. `rmsnorm_fwd`
    1. `mm`
    2. `ce_loss_fwd_bwd`
    3. `mm ×2`
    4. `rmsnorm_bwd`

### `optimizer_head` — `AdamWStep`

- example task: `optimizer_head_0`
- inputs: `W_head` (593.50 MiB), `dW_head_0` (593.50 MiB), `O_head` (1.16 GiB)
- outputs: —
- mutates: `W_head`, `O_head`
- kernel calls:
    0. `adamw_step ×2`

### `q3moeattn_bwd` — `Qwen3MoeBlockBwd`

- example task: `block_bwd_0_0_47`
- inputs: `dy_0_0_47` (256.00 MiB), `A_0_0_47` (2.91 GiB), `y_0_0_46` (256.00 MiB), `W_47` (1.16 GiB), `AuxTemp_0_0_47` (5.00 MiB), `Aux_47` (1.50 KiB)
- outputs: `dy_0_0_46` (256.00 MiB), `dW_0_47` (1.16 GiB)
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

- example task: `optimizer_0_47`
- inputs: `W_47` (1.16 GiB), `dW_0_47` (1.16 GiB), `O_47` (2.32 GiB)
- outputs: —
- mutates: `W_47`, `O_47`
- kernel calls:
    0. `adamw_step ×11`

### `embed_bwd` — `EmbedBwd`

- example task: `embed_bwd_0_0`
- inputs: `dy_embed_0_0` (256.00 MiB), `tokens_0_0` (256.00 KiB)
- outputs: `dW_embed_0` (593.50 MiB)
- mutates: —
- kernel calls:
    0. `embed_bwd_accum`

### `optimizer_embed` — `AdamWStep`

- example task: `optimizer_embed_0`
- inputs: `W_embed` (593.50 MiB), `dW_embed_0` (593.50 MiB), `O_embed` (1.16 GiB)
- outputs: —
- mutates: `W_embed`, `O_embed`
- kernel calls:
    0. `adamw_step`

