# dsv32 / `glm5` @ 16x4K: tasks, objects, kernels

GENERATED from `ShapedDsv32Config.glm5()` at run shape microbatch 16 × seq 4096 — regenerate with `python tools/gen_model_page.py --preset glm5 --microbatch 16 --seq-len 4096`. All presets: [builtin_models.md](../../builtin_models.md); task-kind fleet index: [task_kinds.md](../../task_kinds.md).

Layer kinds (78 layers): `dense dense dense moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe`

**Run shape**: microbatch 16 × seq_len 4096 = **65,536 tokens per round** (× 1 grad-accum round(s) per step). `A_*`/`M_*` objects are sized per round; bytes/token figures transfer to any run shape.

## Object summary

At this run shape (65,536 tokens/round). Token-scaled objects show per-token size in parens. Details per kind below.

| object | scope | size |
|---|---|---|
| `W_i (dense)` | layer | 765.03 MiB |
| `dW_i (dense)` | layer/step | 765.03 MiB |
| `O_i (dense)` | layer | 1.49 GiB |
| `A (dense)` | layer × round | 6.09 GiB (97.39 KiB/token) |
| `M (dense)` | layer × round | 512.00 MiB (8.00 KiB/token) |
| `W_i (moe)` | layer | 18.40 GiB |
| `dW_i (moe)` | layer/step | 18.40 GiB |
| `O_i (moe)` | layer | 36.80 GiB |
| `A (moe)` | layer × round | 7.62 GiB (121.89 KiB/token) |
| `M (moe)` | layer × round | 517.00 MiB (8.08 KiB/token) |
| `W_head` | run | 1.77 GiB |
| `W_embed` | run | 1.77 GiB |
| `O_embed` | run | 3.54 GiB |
| `O_head` | run | 3.54 GiB |
| `hidden state (y)` | boundary buffer | 768.00 MiB (12.00 KiB/token) |

### Aggregate totals (all layers, this run shape)

| type | objects | total size |
|---|---|---|
| W (all weights, incl. embed/head) | 80 | 1,385.67 GiB |
| dW (all gradients, per step) | 80 | 1,385.67 GiB |
| O (all optimizer state) | 80 | 2,771.34 GiB |
| A (all saved activations, one round) | 78 | 589.62 GiB (9.21 MiB/token) |
| M (all metadata, one round) | 78 | 39.37 GiB (629.86 KiB/token) |

## Dims

| field | value |
|---|---|
| `d_model` | 6144 |
| `n_heads` | 64 |
| `q_lora_rank` | 2048 |
| `kv_lora_rank` | 512 |
| `qk_nope_dim` | 192 |
| `qk_rope_dim` | 64 |
| `v_head_dim` | 256 |
| `d_ff` | 12288 |
| `first_k_dense` | 3 |
| `vocab_size` | 154880 |
| `tokens` | 65536 |
| `seq_len` | 4096 |
| `rope_base` | 1000000.0 |
| `index_n_heads` | 32 |
| `index_head_dim` | 128 |
| `index_topk` | 2048 |
| `sparse_mode` | True |
| `train_indexer` | True |

## Objects, per layer kind

`dW_i` mirrors `W_i`'s fields at the grad dtypes; `O_i` holds the optimizer policy's state slots per field (adamw default: `m_f`+`v_f` at the opt dtype; sgd fields contribute none — see extending.md §6). `A_i`/`M_i` exist per (step, round).

### kind `dense` (e.g. layer 0)

**`W_0` weights** — 765.03 MiB

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (6144,) | 12.00 KiB |
| `w_q_a` | bf16 | (6144, 2048) | 24.00 MiB |
| `q_a_norm_w` | bf16 | (2048,) | 4.00 KiB |
| `w_q_b` | bf16 | (2048, 16384) | 64.00 MiB |
| `w_kv_a` | bf16 | (6144, 576) | 6.75 MiB |
| `kv_a_norm_w` | bf16 | (512,) | 1.00 KiB |
| `w_kv_b` | bf16 | (512, 28672) | 28.00 MiB |
| `wo` | bf16 | (16384, 6144) | 192.00 MiB |
| `w_idx_q` | bf16 | (2048, 4096) | 16.00 MiB |
| `w_idx_k` | bf16 | (6144, 128) | 1.50 MiB |
| `idx_k_ln_w` | bf16 | (128,) | 256 B |
| `idx_k_ln_b` | bf16 | (128,) | 256 B |
| `w_idx_w` | fp32 | (6144, 32) | 768.00 KiB |
| `ffn_norm_w` | bf16 | (6144,) | 12.00 KiB |
| `w1` | bf16 | (6144, 12288) | 144.00 MiB |
| `w3` | bf16 | (6144, 12288) | 144.00 MiB |
| `w2` | bf16 | (12288, 6144) | 144.00 MiB |

**`A_.._0` saved context** — 6.09 GiB = **97.39 KiB/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 256.00 KiB |
| `q_a` | bf16 | (65536, 2048) | 256.00 MiB |
| `rstd_qa` | fp32 | (65536,) | 256.00 KiB |
| `kv_a` | bf16 | (65536, 576) | 72.00 MiB |
| `rstd_kva` | fp32 | (65536,) | 256.00 KiB |
| `lse` | fp32 | (64, 65536) | 16.00 MiB |
| `attn_out` | bf16 | (65536, 16384) | 2.00 GiB |
| `h_mid` | bf16 | (65536, 6144) | 768.00 MiB |
| `rstd_ffn` | fp32 | (65536,) | 256.00 KiB |
| `x1` | bf16 | (65536, 12288) | 1.50 GiB |
| `x3` | bf16 | (65536, 12288) | 1.50 GiB |

**`M_.._0` metadata** — 512.00 MiB = **8.00 KiB/token** (never recomputed)

| field | dtype | shape | bytes |
|---|---|---|---|
| `dsa_idx` | int32 | (65536, 2048) | 512.00 MiB |

### kind `moe` (e.g. layer 3)

**`W_3` weights** — 18.40 GiB

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (6144,) | 12.00 KiB |
| `w_q_a` | bf16 | (6144, 2048) | 24.00 MiB |
| `q_a_norm_w` | bf16 | (2048,) | 4.00 KiB |
| `w_q_b` | bf16 | (2048, 16384) | 64.00 MiB |
| `w_kv_a` | bf16 | (6144, 576) | 6.75 MiB |
| `kv_a_norm_w` | bf16 | (512,) | 1.00 KiB |
| `w_kv_b` | bf16 | (512, 28672) | 28.00 MiB |
| `wo` | bf16 | (16384, 6144) | 192.00 MiB |
| `w_idx_q` | bf16 | (2048, 4096) | 16.00 MiB |
| `w_idx_k` | bf16 | (6144, 128) | 1.50 MiB |
| `idx_k_ln_w` | bf16 | (128,) | 256 B |
| `idx_k_ln_b` | bf16 | (128,) | 256 B |
| `w_idx_w` | fp32 | (6144, 32) | 768.00 KiB |
| `ffn_norm_w` | bf16 | (6144,) | 12.00 KiB |
| `w_router` | bf16 | (6144, 256) | 3.00 MiB |
| `w_router_bias` | fp32 | (256,) | 1.00 KiB |
| `w13_experts` | bf16 | (256, 6144, 4096) | 12.00 GiB |
| `w2_experts` | bf16 | (256, 2048, 6144) | 6.00 GiB |
| `w_s13` | bf16 | (6144, 4096) | 48.00 MiB |
| `w_s2` | bf16 | (2048, 6144) | 24.00 MiB |

**`A_.._3` saved context** — 7.62 GiB = **121.89 KiB/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 256.00 KiB |
| `q_a` | bf16 | (65536, 2048) | 256.00 MiB |
| `rstd_qa` | fp32 | (65536,) | 256.00 KiB |
| `kv_a` | bf16 | (65536, 576) | 72.00 MiB |
| `rstd_kva` | fp32 | (65536,) | 256.00 KiB |
| `lse` | fp32 | (64, 65536) | 16.00 MiB |
| `attn_out` | bf16 | (65536, 16384) | 2.00 GiB |
| `h_mid` | bf16 | (65536, 6144) | 768.00 MiB |
| `rstd_ffn` | fp32 | (65536,) | 256.00 KiB |
| `router_logits` | bf16 | (65536, 256) | 32.00 MiB |
| `h13` | bf16 | (524288, 4096) | 4.00 GiB |
| `s13` | bf16 | (65536, 4096) | 512.00 MiB |

**`M_.._3` metadata** — 517.00 MiB = **8.08 KiB/token** (never recomputed)

| field | dtype | shape | bytes |
|---|---|---|---|
| `dsa_idx` | int32 | (65536, 2048) | 512.00 MiB |
| `route_w` | bf16 | (65536, 8) | 1.00 MiB |
| `route_ids` | int32 | (65536, 8) | 2.00 MiB |
| `route_order` | int32 | (524288,) | 2.00 MiB |
| `route_offsets` | int32 | (257,) | 1.00 KiB |

**`W_head`** — 1.77 GiB

| field | dtype | shape | bytes |
|---|---|---|---|
| `w` | bf16 | (154880, 6144) | 1.77 GiB |
| `final_norm_w` | bf16 | (6144,) | 12.00 KiB |

## Tasks

### `prologue_round` — `RoundPrologue`

- example task: `prologue_round_0_0`
- inputs: `Aux_3` (3.00 KiB), `Aux_4` (3.00 KiB), `Aux_5` (3.00 KiB), `Aux_6` (3.00 KiB), `Aux_7` (3.00 KiB), `Aux_8` (3.00 KiB), `Aux_9` (3.00 KiB), `Aux_10` (3.00 KiB), `Aux_11` (3.00 KiB), `Aux_12` (3.00 KiB), `Aux_13` (3.00 KiB), `Aux_14` (3.00 KiB), `Aux_15` (3.00 KiB), `Aux_16` (3.00 KiB), `Aux_17` (3.00 KiB), `Aux_18` (3.00 KiB), `Aux_19` (3.00 KiB), `Aux_20` (3.00 KiB), `Aux_21` (3.00 KiB), `Aux_22` (3.00 KiB), `Aux_23` (3.00 KiB), `Aux_24` (3.00 KiB), `Aux_25` (3.00 KiB), `Aux_26` (3.00 KiB), `Aux_27` (3.00 KiB), `Aux_28` (3.00 KiB), `Aux_29` (3.00 KiB), `Aux_30` (3.00 KiB), `Aux_31` (3.00 KiB), `Aux_32` (3.00 KiB), `Aux_33` (3.00 KiB), `Aux_34` (3.00 KiB), `Aux_35` (3.00 KiB), `Aux_36` (3.00 KiB), `Aux_37` (3.00 KiB), `Aux_38` (3.00 KiB), `Aux_39` (3.00 KiB), `Aux_40` (3.00 KiB), `Aux_41` (3.00 KiB), `Aux_42` (3.00 KiB), `Aux_43` (3.00 KiB), `Aux_44` (3.00 KiB), `Aux_45` (3.00 KiB), `Aux_46` (3.00 KiB), `Aux_47` (3.00 KiB), `Aux_48` (3.00 KiB), `Aux_49` (3.00 KiB), `Aux_50` (3.00 KiB), `Aux_51` (3.00 KiB), `Aux_52` (3.00 KiB), `Aux_53` (3.00 KiB), `Aux_54` (3.00 KiB), `Aux_55` (3.00 KiB), `Aux_56` (3.00 KiB), `Aux_57` (3.00 KiB), `Aux_58` (3.00 KiB), `Aux_59` (3.00 KiB), `Aux_60` (3.00 KiB), `Aux_61` (3.00 KiB), `Aux_62` (3.00 KiB), `Aux_63` (3.00 KiB), `Aux_64` (3.00 KiB), `Aux_65` (3.00 KiB), `Aux_66` (3.00 KiB), `Aux_67` (3.00 KiB), `Aux_68` (3.00 KiB), `Aux_69` (3.00 KiB), `Aux_70` (3.00 KiB), `Aux_71` (3.00 KiB), `Aux_72` (3.00 KiB), `Aux_73` (3.00 KiB), `Aux_74` (3.00 KiB), `Aux_75` (3.00 KiB), `Aux_76` (3.00 KiB), `Aux_77` (3.00 KiB)
- outputs: `current_round_0_0` (4 B)
- mutates: `Aux_3`, `Aux_4`, `Aux_5`, `Aux_6`, `Aux_7`, `Aux_8`, `Aux_9`, `Aux_10`, `Aux_11`, `Aux_12`, `Aux_13`, `Aux_14`, `Aux_15`, `Aux_16`, `Aux_17`, `Aux_18`, `Aux_19`, `Aux_20`, `Aux_21`, `Aux_22`, `Aux_23`, `Aux_24`, `Aux_25`, `Aux_26`, `Aux_27`, `Aux_28`, `Aux_29`, `Aux_30`, `Aux_31`, `Aux_32`, `Aux_33`, `Aux_34`, `Aux_35`, `Aux_36`, `Aux_37`, `Aux_38`, `Aux_39`, `Aux_40`, `Aux_41`, `Aux_42`, `Aux_43`, `Aux_44`, `Aux_45`, `Aux_46`, `Aux_47`, `Aux_48`, `Aux_49`, `Aux_50`, `Aux_51`, `Aux_52`, `Aux_53`, `Aux_54`, `Aux_55`, `Aux_56`, `Aux_57`, `Aux_58`, `Aux_59`, `Aux_60`, `Aux_61`, `Aux_62`, `Aux_63`, `Aux_64`, `Aux_65`, `Aux_66`, `Aux_67`, `Aux_68`, `Aux_69`, `Aux_70`, `Aux_71`, `Aux_72`, `Aux_73`, `Aux_74`, `Aux_75`, `Aux_76`, `Aux_77`

### `embed_fwd` — `EmbedFwd`

- example task: `embed_fwd_0_0`
- inputs: `tokens_0_0` (256.00 KiB), `W_embed` (1.77 GiB)
- outputs: `y_embed_0_0` (768.00 MiB)
- mutates: —
- kernel calls:
    0. `index_select`

### `dsadense_fwd` — `Dsv32DenseBlockFwd`

- example task: `block_fwd_0_0_0`
- inputs: `y_embed_0_0` (768.00 MiB), `W_0` (765.03 MiB)
- outputs: `y_0_0_0` (768.00 MiB), `A_0_0_0` (6.09 GiB), `AuxTemp_0_0_0` (512.00 MiB)
- mutates: —
- stages (name — emitted ctx fields):
    0. `attn_norm` — rstd_attn
    1. `mla_q` — q_a, rstd_qa
    2. `mla_kv` — kv_a, rstd_kva
    3. `dsa_select` — — [meta: never recomputed]
    4. `dsa_attn` — lse, attn_out
    5. `resid1_norm2` — h_mid, rstd_ffn
    6. `up_proj` — x1, x3  ← derived recompute boundary
    7. `swiglu` — —
    8. `down_resid` — —
- kernel calls, by stage:
    - `attn_norm`:
        0. `rmsnorm_fwd`
    - `mla_q`:
        1. `mm`
        2. `rmsnorm_fwd`
        3. `mm`
        4. `rope_fwd`
        5. `rmsnorm_apply`
    - `mla_kv`:
        6. `mm`
        7. `rope_fwd`
        8. `mm`
        9. `rope_fwd`
        10. `mm ×2`
        11. `rmsnorm_fwd`
        12. `rope_fwd`
        13. `mm`
    - `dsa_select`:
        14. `dsa_index_scores`
        15. `dsa_topk`
    - `dsa_attn`:
        16. `dsa_sparse_attn_fwd`
    - `resid1_norm2`:
        17. `addmm`
        18. `rmsnorm_fwd`
    - `up_proj`:
        19. `mm ×2`
    - `swiglu`:
        20. `swiglu_fwd_out`
    - `down_resid`:
        21. `addmm`

### `dsamoe_fwd` — `Dsv32MoeBlockFwd`

- example task: `block_fwd_0_0_3`
- inputs: `y_0_0_2` (768.00 MiB), `W_3` (18.40 GiB), `current_round_0_0` (4 B), `Aux_3` (3.00 KiB)
- outputs: `y_0_0_3` (768.00 MiB), `A_0_0_3` (7.62 GiB), `AuxTemp_0_0_3` (517.00 MiB)
- mutates: `Aux_3`
- stages (name — emitted ctx fields):
    0. `attn_norm` — rstd_attn
    1. `mla_q` — q_a, rstd_qa
    2. `mla_kv` — kv_a, rstd_kva
    3. `dsa_select` — — [meta: never recomputed]
    4. `dsa_attn` — lse, attn_out
    5. `resid1_norm2` — h_mid, rstd_ffn
    6. `moe_route` — router_logits
    7. `moe_dispatch` — —
    8. `moe_experts13` — h13
    9. `moe_shared` — s13  ← derived recompute boundary
    10. `moe_experts2_combine` — —
- kernel calls, by stage:
    - `attn_norm`:
        0. `rmsnorm_fwd`
    - `mla_q`:
        1. `mm`
        2. `rmsnorm_fwd`
        3. `mm`
        4. `rope_fwd`
        5. `rmsnorm_apply`
    - `mla_kv`:
        6. `mm`
        7. `rope_fwd`
        8. `mm`
        9. `rope_fwd`
        10. `mm ×2`
        11. `rmsnorm_fwd`
        12. `rope_fwd`
        13. `mm`
    - `dsa_select`:
        14. `dsa_index_scores`
        15. `dsa_topk`
    - `dsa_attn`:
        16. `dsa_sparse_attn_fwd`
    - `resid1_norm2`:
        17. `addmm`
        18. `rmsnorm_fwd`
    - `moe_route`:
        19. `mm`
        20. `moe_topk_sigmoid_noaux`
        21. `scatter_add_ ×2`
    - `moe_dispatch`:
        22. `moe_sort`
        23. `moe_dispatch_fwd`
    - `moe_experts13`:
        24. `moe_grouped_mm_fwd`
    - `moe_shared`:
        25. `mm`
    - `moe_experts2_combine`:
        26. `swiglu_packed_fwd`
        27. `moe_grouped_mm_fwd`
        28. `swiglu_packed_fwd`
        29. `mm`
        30. `moe_combine_fwd`

### `head_loss` — `HeadLoss`

- example task: `head_loss_0_0`
- inputs: `y_0_0_77` (768.00 MiB), `targets_0_0` (256.00 KiB), `W_head` (1.77 GiB)
- outputs: `dy_0_0_77` (768.00 MiB), `loss_0_0` (4 B), `dW_head_0` (1.77 GiB)
- mutates: —
- kernel calls:
    0. `rmsnorm_fwd`
    1. `mm`
    2. `ce_loss_fwd_bwd`
    3. `mm ×2`
    4. `rmsnorm_bwd`

### `optimizer_head` — `AdamWStep`

- example task: `optimizer_head_0`
- inputs: `W_head` (1.77 GiB), `dW_head_0` (1.77 GiB), `O_head` (3.54 GiB)
- outputs: —
- mutates: `W_head`, `O_head`
- kernel calls:
    0. `adamw_step ×2`

### `dsamoe_bwd` — `Dsv32MoeBlockBwd`

- example task: `block_bwd_0_0_77`
- inputs: `dy_0_0_77` (768.00 MiB), `A_0_0_77` (7.62 GiB), `y_0_0_76` (768.00 MiB), `W_77` (18.40 GiB), `AuxTemp_0_0_77` (517.00 MiB), `Aux_77` (3.00 KiB)
- outputs: `dy_0_0_76` (768.00 MiB), `dW_0_77` (18.40 GiB)
- mutates: `W_77`
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
    27. `sort`
    28. `scatter_add_`
    29. `dsa_sparse_attn_bwd`
    30. `rmsnorm_apply`
    31. `mm`
    32. `rope_fwd`
    33. `mm`
    34. `rope_fwd`
    35. `mm`
    36. `dsa_index_scores`
    37. `_softmax`
    38. `dsa_probs_sum`
    39. `dsa_index_bwd`
    40. `rope_bwd`
    41. `mm`
    42. `rope_bwd`
    43. `mm ×3`
    44. `rope_bwd`
    45. `mm ×2`
    46. `rmsnorm_bwd`
    47. `rope_bwd`
    48. `mm ×2`
    49. `rmsnorm_bwd`
    50. `mm ×3`
    51. `rmsnorm_bwd`

### `optimizer_block` — `AdamWStep`

- example task: `optimizer_0_77`
- inputs: `W_77` (18.40 GiB), `dW_0_77` (18.40 GiB), `O_77` (36.80 GiB)
- outputs: —
- mutates: `W_77`, `O_77`
- kernel calls:
    0. `adamw_step ×19`

### `dsadense_bwd` — `Dsv32DenseBlockBwd`

- example task: `block_bwd_0_0_2`
- inputs: `dy_0_0_2` (768.00 MiB), `A_0_0_2` (6.09 GiB), `y_0_0_1` (768.00 MiB), `W_2` (765.03 MiB), `AuxTemp_0_0_2` (512.00 MiB)
- outputs: `dy_0_0_1` (768.00 MiB), `dW_0_2` (765.03 MiB)
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
    13. `sort`
    14. `scatter_add_`
    15. `dsa_sparse_attn_bwd`
    16. `rmsnorm_apply`
    17. `mm`
    18. `rope_fwd`
    19. `mm`
    20. `rope_fwd`
    21. `mm`
    22. `dsa_index_scores`
    23. `_softmax`
    24. `dsa_probs_sum`
    25. `dsa_index_bwd`
    26. `rope_bwd`
    27. `mm`
    28. `rope_bwd`
    29. `mm ×3`
    30. `rope_bwd`
    31. `mm ×2`
    32. `rmsnorm_bwd`
    33. `rope_bwd`
    34. `mm ×2`
    35. `rmsnorm_bwd`
    36. `mm ×3`
    37. `rmsnorm_bwd`

### `embed_bwd` — `EmbedBwd`

- example task: `embed_bwd_0_0`
- inputs: `dy_embed_0_0` (768.00 MiB), `tokens_0_0` (256.00 KiB)
- outputs: `dW_embed_0` (1.77 GiB)
- mutates: —
- kernel calls:
    0. `embed_bwd_accum`

### `optimizer_embed` — `AdamWStep`

- example task: `optimizer_embed_0`
- inputs: `W_embed` (1.77 GiB), `dW_embed_0` (1.77 GiB), `O_embed` (3.54 GiB)
- outputs: —
- mutates: `W_embed`, `O_embed`
- kernel calls:
    0. `adamw_step`

