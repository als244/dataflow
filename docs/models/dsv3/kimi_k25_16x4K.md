# dsv3 / `kimi_k25` @ 16x4K: tasks, objects, kernels

GENERATED from `ShapedDsv3Config.kimi_k25()` at run shape microbatch 16 × seq 4096 — regenerate with `python tools/gen_model_docs/gen_model_page.py --preset kimi_k25 --microbatch 16 --seq-len 4096`. All presets: [builtin_models.md](../../builtin_models.md); task-kind fleet index: [task_kinds.md](../../task_kinds.md).

Layer kinds (61 layers): `dense moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe`

**Run shape**: microbatch 16 × seq_len 4096 = **65,536 tokens per round** (× 1 grad-accum round(s) per step). `A_*`/`AuxTemp_*` objects are sized per round; bytes/token figures transfer to any run shape.

## Object summary

At this run shape (65,536 tokens/round). Token-scaled objects show per-token size in parens. Details per kind below.

| object | scope | size |
|---|---|---|
| `W_i (dense)` | layer | 948.91 MiB |
| `dW_i (dense)` | layer/step | 948.91 MiB |
| `O_i (dense)` | layer | 1.85 GiB |
| `A (dense)` | layer × round | 6.65 GiB (106.39 KiB/token) |
| `W_i (moe)` | layer | 31.78 GiB |
| `dW_i (moe)` | layer/step | 31.78 GiB |
| `O_i (moe)` | layer | 63.55 GiB |
| `A (moe)` | layer × round | 6.70 GiB (107.14 KiB/token) |
| `M (moe)` | layer × round | 5.00 MiB (80.0 B/token) |
| `W_head` | run | 2.19 GiB |
| `W_embed` | run | 2.19 GiB |
| `O_embed` | run | 4.38 GiB |
| `O_head` | run | 4.38 GiB |
| `hidden state (y)` | boundary buffer | 896.00 MiB (14.00 KiB/token) |

### Aggregate totals (all layers, this run shape)

| type | objects | total size |
|---|---|---|
| W (all weights, incl. embed/head) | 63 | 1,911.83 GiB |
| dW (all gradients, per step) | 63 | 1,911.83 GiB |
| O (all optimizer state) | 63 | 3,823.67 GiB |
| A (all saved activations, one round) | 61 | 408.43 GiB (6.38 MiB/token) |
| M (all metadata, one round) | 60 | 300.10 MiB (4.69 KiB/token) |

## Dims

| field | value |
|---|---|
| `d_model` | 7168 |
| `n_heads` | 64 |
| `q_lora_rank` | 1536 |
| `kv_lora_rank` | 512 |
| `qk_nope_dim` | 128 |
| `qk_rope_dim` | 64 |
| `v_head_dim` | 128 |
| `d_ff` | 18432 |
| `first_k_dense` | 1 |
| `vocab_size` | 163840 |
| `tokens` | 65536 |
| `seq_len` | 4096 |
| `rope_base` | 50000.0 |

## Objects, per layer kind

`dW_i` mirrors `W_i`'s fields at the grad dtypes; `O_i` holds the optimizer policy's state slots per field (adamw default: `m_f`+`v_f` at the opt dtype; sgd fields contribute none — see extending.md §6). `A_i`/`M_i` exist per (step, round).

### kind `dense` (e.g. layer 0)

**`W_0` weights** — 948.91 MiB

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (7168,) | 14.00 KiB |
| `w_q_a` | bf16 | (7168, 1536) | 21.00 MiB |
| `q_a_norm_w` | bf16 | (1536,) | 3.00 KiB |
| `w_q_b` | bf16 | (1536, 12288) | 36.00 MiB |
| `w_kv_a` | bf16 | (7168, 576) | 7.88 MiB |
| `kv_a_norm_w` | bf16 | (512,) | 1.00 KiB |
| `w_kv_b` | bf16 | (512, 16384) | 16.00 MiB |
| `wo` | bf16 | (8192, 7168) | 112.00 MiB |
| `ffn_norm_w` | bf16 | (7168,) | 14.00 KiB |
| `w1` | bf16 | (7168, 18432) | 252.00 MiB |
| `w3` | bf16 | (7168, 18432) | 252.00 MiB |
| `w2` | bf16 | (18432, 7168) | 252.00 MiB |

**`A_.._0` saved context** — 6.65 GiB = **106.39 KiB/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 256.00 KiB |
| `q_a` | bf16 | (65536, 1536) | 192.00 MiB |
| `rstd_qa` | fp32 | (65536,) | 256.00 KiB |
| `kv_a` | bf16 | (65536, 576) | 72.00 MiB |
| `rstd_kva` | fp32 | (65536,) | 256.00 KiB |
| `lse` | fp32 | (64, 65536) | 16.00 MiB |
| `attn_out` | bf16 | (65536, 8192) | 1.00 GiB |
| `h_mid` | bf16 | (65536, 7168) | 896.00 MiB |
| `rstd_ffn` | fp32 | (65536,) | 256.00 KiB |
| `x1` | bf16 | (65536, 18432) | 2.25 GiB |
| `x3` | bf16 | (65536, 18432) | 2.25 GiB |

### kind `moe` (e.g. layer 1)

**`W_1` weights** — 31.78 GiB

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (7168,) | 14.00 KiB |
| `w_q_a` | bf16 | (7168, 1536) | 21.00 MiB |
| `q_a_norm_w` | bf16 | (1536,) | 3.00 KiB |
| `w_q_b` | bf16 | (1536, 12288) | 36.00 MiB |
| `w_kv_a` | bf16 | (7168, 576) | 7.88 MiB |
| `kv_a_norm_w` | bf16 | (512,) | 1.00 KiB |
| `w_kv_b` | bf16 | (512, 16384) | 16.00 MiB |
| `wo` | bf16 | (8192, 7168) | 112.00 MiB |
| `ffn_norm_w` | bf16 | (7168,) | 14.00 KiB |
| `w_router` | bf16 | (7168, 384) | 5.25 MiB |
| `w_router_bias` | fp32 | (384,) | 1.50 KiB |
| `w13_experts` | bf16 | (384, 7168, 4096) | 21.00 GiB |
| `w2_experts` | bf16 | (384, 2048, 7168) | 10.50 GiB |
| `w_s13` | bf16 | (7168, 4096) | 56.00 MiB |
| `w_s2` | bf16 | (2048, 7168) | 28.00 MiB |

**`A_.._1` saved context** — 6.70 GiB = **107.14 KiB/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 256.00 KiB |
| `q_a` | bf16 | (65536, 1536) | 192.00 MiB |
| `rstd_qa` | fp32 | (65536,) | 256.00 KiB |
| `kv_a` | bf16 | (65536, 576) | 72.00 MiB |
| `rstd_kva` | fp32 | (65536,) | 256.00 KiB |
| `lse` | fp32 | (64, 65536) | 16.00 MiB |
| `attn_out` | bf16 | (65536, 8192) | 1.00 GiB |
| `h_mid` | bf16 | (65536, 7168) | 896.00 MiB |
| `rstd_ffn` | fp32 | (65536,) | 256.00 KiB |
| `router_logits` | bf16 | (65536, 384) | 48.00 MiB |
| `h13` | bf16 | (524288, 4096) | 4.00 GiB |
| `s13` | bf16 | (65536, 4096) | 512.00 MiB |

**`M_.._1` metadata** — 5.00 MiB = **80.0 B/token** (never recomputed)

| field | dtype | shape | bytes |
|---|---|---|---|
| `route_w` | bf16 | (65536, 8) | 1.00 MiB |
| `route_ids` | int32 | (65536, 8) | 2.00 MiB |
| `route_order` | int32 | (524288,) | 2.00 MiB |
| `route_offsets` | int32 | (385,) | 1.50 KiB |

**`W_head`** — 2.19 GiB

| field | dtype | shape | bytes |
|---|---|---|---|
| `w` | bf16 | (163840, 7168) | 2.19 GiB |
| `final_norm_w` | bf16 | (7168,) | 14.00 KiB |

## Tasks

### `prologue_round` — `RoundPrologue`

- example task: `prologue_round_0_0`
- inputs: `Aux_1` (4.50 KiB), `Aux_2` (4.50 KiB), `Aux_3` (4.50 KiB), `Aux_4` (4.50 KiB), `Aux_5` (4.50 KiB), `Aux_6` (4.50 KiB), `Aux_7` (4.50 KiB), `Aux_8` (4.50 KiB), `Aux_9` (4.50 KiB), `Aux_10` (4.50 KiB), `Aux_11` (4.50 KiB), `Aux_12` (4.50 KiB), `Aux_13` (4.50 KiB), `Aux_14` (4.50 KiB), `Aux_15` (4.50 KiB), `Aux_16` (4.50 KiB), `Aux_17` (4.50 KiB), `Aux_18` (4.50 KiB), `Aux_19` (4.50 KiB), `Aux_20` (4.50 KiB), `Aux_21` (4.50 KiB), `Aux_22` (4.50 KiB), `Aux_23` (4.50 KiB), `Aux_24` (4.50 KiB), `Aux_25` (4.50 KiB), `Aux_26` (4.50 KiB), `Aux_27` (4.50 KiB), `Aux_28` (4.50 KiB), `Aux_29` (4.50 KiB), `Aux_30` (4.50 KiB), `Aux_31` (4.50 KiB), `Aux_32` (4.50 KiB), `Aux_33` (4.50 KiB), `Aux_34` (4.50 KiB), `Aux_35` (4.50 KiB), `Aux_36` (4.50 KiB), `Aux_37` (4.50 KiB), `Aux_38` (4.50 KiB), `Aux_39` (4.50 KiB), `Aux_40` (4.50 KiB), `Aux_41` (4.50 KiB), `Aux_42` (4.50 KiB), `Aux_43` (4.50 KiB), `Aux_44` (4.50 KiB), `Aux_45` (4.50 KiB), `Aux_46` (4.50 KiB), `Aux_47` (4.50 KiB), `Aux_48` (4.50 KiB), `Aux_49` (4.50 KiB), `Aux_50` (4.50 KiB), `Aux_51` (4.50 KiB), `Aux_52` (4.50 KiB), `Aux_53` (4.50 KiB), `Aux_54` (4.50 KiB), `Aux_55` (4.50 KiB), `Aux_56` (4.50 KiB), `Aux_57` (4.50 KiB), `Aux_58` (4.50 KiB), `Aux_59` (4.50 KiB), `Aux_60` (4.50 KiB)
- outputs: `current_round_0_0` (4 B)
- mutates: `Aux_1`, `Aux_2`, `Aux_3`, `Aux_4`, `Aux_5`, `Aux_6`, `Aux_7`, `Aux_8`, `Aux_9`, `Aux_10`, `Aux_11`, `Aux_12`, `Aux_13`, `Aux_14`, `Aux_15`, `Aux_16`, `Aux_17`, `Aux_18`, `Aux_19`, `Aux_20`, `Aux_21`, `Aux_22`, `Aux_23`, `Aux_24`, `Aux_25`, `Aux_26`, `Aux_27`, `Aux_28`, `Aux_29`, `Aux_30`, `Aux_31`, `Aux_32`, `Aux_33`, `Aux_34`, `Aux_35`, `Aux_36`, `Aux_37`, `Aux_38`, `Aux_39`, `Aux_40`, `Aux_41`, `Aux_42`, `Aux_43`, `Aux_44`, `Aux_45`, `Aux_46`, `Aux_47`, `Aux_48`, `Aux_49`, `Aux_50`, `Aux_51`, `Aux_52`, `Aux_53`, `Aux_54`, `Aux_55`, `Aux_56`, `Aux_57`, `Aux_58`, `Aux_59`, `Aux_60`

### `embed_fwd` — `EmbedFwd`

- example task: `embed_fwd_0_0`
- inputs: `tokens_0_0` (256.00 KiB), `W_embed` (2.19 GiB)
- outputs: `y_embed_0_0` (896.00 MiB)
- mutates: —
- kernel calls:
    0. `index_select`

### `mladense_fwd` — `Dsv3DenseBlockFwd`

- example task: `block_fwd_0_0_0`
- inputs: `y_embed_0_0` (896.00 MiB), `W_0` (948.91 MiB)
- outputs: `y_0_0_0` (896.00 MiB), `A_0_0_0` (6.65 GiB)
- mutates: —
- stages (name — emitted ctx fields):
    0. `attn_norm` — rstd_attn
    1. `mla_q` — q_a, rstd_qa
    2. `mla_kv` — kv_a, rstd_kva
    3. `mla_attn` — lse, attn_out
    4. `resid1_norm2` — h_mid, rstd_ffn
    5. `up_proj` — x1, x3  ← derived recompute boundary
    6. `swiglu` — —
    7. `down_resid` — —
- kernel calls, by stage:
    - `attn_norm`:
        0. `rmsnorm_fwd`
    - `mla_q`:
        1. `mm`
        2. `rmsnorm_fwd`
        3. `mm`
        4. `rope_fwd`
    - `mla_kv`:
        5. `mm`
        6. `rmsnorm_fwd`
        7. `rope_fwd`
        8. `mm`
    - `resid1_norm2`:
        9. `addmm`
        10. `rmsnorm_fwd`
    - `up_proj`:
        11. `mm ×2`
    - `swiglu`:
        12. `swiglu_fwd_out`
    - `down_resid`:
        13. `addmm`

### `mlamoe_fwd` — `Dsv3MoeBlockFwd`

- example task: `block_fwd_0_0_1`
- inputs: `y_0_0_0` (896.00 MiB), `W_1` (31.78 GiB), `current_round_0_0` (4 B), `Aux_1` (4.50 KiB)
- outputs: `y_0_0_1` (896.00 MiB), `A_0_0_1` (6.70 GiB), `AuxTemp_0_0_1` (5.00 MiB)
- mutates: `Aux_1`
- stages (name — emitted ctx fields):
    0. `attn_norm` — rstd_attn
    1. `mla_q` — q_a, rstd_qa
    2. `mla_kv` — kv_a, rstd_kva
    3. `mla_attn` — lse, attn_out
    4. `resid1_norm2` — h_mid, rstd_ffn
    5. `moe_route` — router_logits
    6. `moe_dispatch` — —
    7. `moe_experts13` — h13
    8. `moe_shared` — s13  ← derived recompute boundary
    9. `moe_experts2_combine` — —
- kernel calls, by stage:
    - `attn_norm`:
        0. `rmsnorm_fwd`
    - `mla_q`:
        1. `mm`
        2. `rmsnorm_fwd`
        3. `mm`
        4. `rope_fwd`
    - `mla_kv`:
        5. `mm`
        6. `rmsnorm_fwd`
        7. `rope_fwd`
        8. `mm`
    - `resid1_norm2`:
        9. `addmm`
        10. `rmsnorm_fwd`
    - `moe_route`:
        11. `mm`
        12. `moe_topk_sigmoid_noaux`
        13. `scatter_add_ ×2`
    - `moe_dispatch`:
        14. `moe_sort`
        15. `moe_dispatch_fwd`
    - `moe_experts13`:
        16. `moe_grouped_mm_fwd`
    - `moe_shared`:
        17. `mm`
    - `moe_experts2_combine`:
        18. `swiglu_packed_fwd`
        19. `moe_grouped_mm_fwd`
        20. `swiglu_packed_fwd`
        21. `mm`
        22. `moe_combine_fwd`

### `head_loss` — `HeadLoss`

- example task: `head_loss_0_0`
- inputs: `y_0_0_60` (896.00 MiB), `targets_0_0` (256.00 KiB), `W_head` (2.19 GiB)
- outputs: `dy_0_0_60` (896.00 MiB), `loss_0_0` (4 B), `dW_head_0` (2.19 GiB)
- mutates: —
- kernel calls:
    0. `rmsnorm_fwd`
    1. `mm`
    2. `ce_loss_fwd_bwd`
    3. `mm ×2`
    4. `rmsnorm_bwd`

### `optimizer_head` — `AdamWStep`

- example task: `optimizer_head_0`
- inputs: `W_head` (2.19 GiB), `dW_head_0` (2.19 GiB), `O_head` (4.38 GiB)
- outputs: —
- mutates: `W_head`, `O_head`
- kernel calls:
    0. `adamw_step ×2`

### `mlamoe_bwd` — `Dsv3MoeBlockBwd`

- example task: `block_bwd_0_0_60`
- inputs: `dy_0_0_60` (896.00 MiB), `A_0_0_60` (6.70 GiB), `y_0_0_59` (896.00 MiB), `W_60` (31.78 GiB), `AuxTemp_0_0_60` (5.00 MiB), `Aux_60` (4.50 KiB)
- outputs: `dy_0_0_59` (896.00 MiB), `dW_0_60` (31.78 GiB)
- mutates: `W_60`
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
    12. `moe_router_bwd_sigmoid`
    13. `moe_seq_aux_grad`
    14. `mm`
    15. `swiglu_packed_fwd`
    16. `mm ×2`
    17. `swiglu_packed_bwd`
    18. `mm`
    19. `rmsnorm_bwd`
    20. `mm ×2`
    21. `rmsnorm_apply`
    22. `mm`
    23. `rope_fwd`
    24. `rmsnorm_apply`
    25. `rope_fwd`
    26. `mm`
    27. `_flash_attention_backward`
    28. `rope_bwd`
    29. `mm ×2`
    30. `rmsnorm_bwd`
    31. `rope_bwd`
    32. `mm ×2`
    33. `rmsnorm_bwd`
    34. `rmsnorm_apply`
    35. `mm ×3`
    36. `rmsnorm_bwd`

### `optimizer_block` — `AdamWStep`

- example task: `optimizer_0_60`
- inputs: `W_60` (31.78 GiB), `dW_0_60` (31.78 GiB), `O_60` (63.55 GiB)
- outputs: —
- mutates: `W_60`, `O_60`
- kernel calls:
    0. `adamw_step ×14`

### `mladense_bwd` — `Dsv3DenseBlockBwd`

- example task: `block_bwd_0_0_0`
- inputs: `dy_0_0_0` (896.00 MiB), `A_0_0_0` (6.65 GiB), `y_embed_0_0` (896.00 MiB), `W_0` (948.91 MiB)
- outputs: `dy_embed_0_0` (896.00 MiB), `dW_0_0` (948.91 MiB)
- mutates: —
- kernel calls:
    0. `rmsnorm_apply`
    1. `swiglu_fwd_out`
    2. `mm ×2`
    3. `swiglu_bwd`
    4. `mm ×3`
    5. `rmsnorm_bwd`
    6. `mm ×2`
    7. `rmsnorm_apply`
    8. `mm`
    9. `rope_fwd`
    10. `rmsnorm_apply`
    11. `rope_fwd`
    12. `mm`
    13. `_flash_attention_backward`
    14. `rope_bwd`
    15. `mm ×2`
    16. `rmsnorm_bwd`
    17. `rope_bwd`
    18. `mm ×2`
    19. `rmsnorm_bwd`
    20. `rmsnorm_apply`
    21. `mm ×3`
    22. `rmsnorm_bwd`

### `embed_bwd` — `EmbedBwd`

- example task: `embed_bwd_0_0`
- inputs: `dy_embed_0_0` (896.00 MiB), `tokens_0_0` (256.00 KiB)
- outputs: `dW_embed_0` (2.19 GiB)
- mutates: —
- kernel calls:
    0. `embed_bwd_accum`

### `optimizer_embed` — `AdamWStep`

- example task: `optimizer_embed_0`
- inputs: `W_embed` (2.19 GiB), `dW_embed_0` (2.19 GiB), `O_embed` (4.38 GiB)
- outputs: —
- mutates: `W_embed`, `O_embed`
- kernel calls:
    0. `adamw_step`

