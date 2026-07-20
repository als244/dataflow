# dsv32 / `dsv32_mini_warmup` @ 16x4K: tasks, objects, kernels

GENERATED from `ShapedDsv32Config.dsv32_mini_warmup()` at run shape microbatch 16 × seq 4096 — regenerate with `python tools/gen_model_docs/gen_model_page.py --preset dsv32_mini_warmup --microbatch 16 --seq-len 4096`. All presets: [builtin_models.md](../../builtin_models.md); task-kind fleet index: [task_kinds.md](../../task_kinds.md).

Layer kinds (18 layers): `dense dense moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe`

**Run shape**: microbatch 16 × seq_len 4096 = **65,536 tokens per round** (× 4 grad-accum round(s) per step). `A_*`/`AuxTemp_*` objects are sized per round; bytes/token figures transfer to any run shape.

## Object summary

At this run shape (65,536 tokens/round). Token-scaled objects show per-token size in parens. Details per kind below.

| object | scope | size |
|---|---|---|
| `W_i (dense)` | layer | 106.45 MiB |
| `dW_i (dense)` | layer/step | 106.45 MiB |
| `O_i (dense)` | layer | 1.63 MiB |
| `A (dense)` | layer × round | 104.75 MiB (1.64 KiB/token) |
| `W_i (moe)` | layer | 1.52 GiB |
| `dW_i (moe)` | layer/step | 1.52 GiB |
| `O_i (moe)` | layer | 1.63 MiB |
| `A (moe)` | layer × round | 104.75 MiB (1.64 KiB/token) |
| `M (moe)` | layer × round | 5.00 MiB (80.0 B/token) |
| `W_head` | run | 505.00 MiB |
| `W_embed` | run | 505.00 MiB |
| `hidden state (y)` | boundary buffer | 256.00 MiB (4.00 KiB/token) |

### Aggregate totals (all layers, this run shape)

| type | objects | total size |
|---|---|---|
| W (all weights, incl. embed/head) | 20 | 25.55 GiB |
| dW (all gradients, per step) | 18 | 14.63 MiB |
| O (all optimizer state) | 18 | 29.27 MiB |
| A (all saved activations, one round) | 72 | 7.37 GiB (117.84 KiB/token) |
| M (all metadata, one round) | 64 | 320.05 MiB (5.00 KiB/token) |

## Dims

| field | value |
|---|---|
| `d_model` | 2048 |
| `n_heads` | 16 |
| `q_lora_rank` | 512 |
| `kv_lora_rank` | 256 |
| `qk_nope_dim` | 64 |
| `qk_rope_dim` | 32 |
| `v_head_dim` | 64 |
| `d_ff` | 8192 |
| `first_k_dense` | 2 |
| `vocab_size` | 129280 |
| `tokens` | 65536 |
| `seq_len` | 4096 |
| `rope_base` | 10000.0 |
| `index_n_heads` | 8 |
| `index_head_dim` | 64 |
| `index_topk` | 1024 |
| `sparse_mode` | False |
| `train_indexer` | True |

## Objects, per layer kind

`dW_i` mirrors `W_i`'s fields at the grad dtypes; `O_i` holds the optimizer policy's state slots per field (adamw default: `m_f`+`v_f` at the opt dtype; sgd fields contribute none — see extending.md §6). `A_i`/`M_i` exist per (step, round).

### kind `dense` (e.g. layer 0)

**`W_0` weights** — 106.45 MiB

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (2048,) | 4.00 KiB |
| `w_q_a` | bf16 | (2048, 512) | 2.00 MiB |
| `q_a_norm_w` | bf16 | (512,) | 1.00 KiB |
| `w_q_b` | bf16 | (512, 1536) | 1.50 MiB |
| `w_kv_a` | bf16 | (2048, 288) | 1.12 MiB |
| `kv_a_norm_w` | bf16 | (256,) | 512 B |
| `w_kv_b` | bf16 | (256, 2048) | 1.00 MiB |
| `wo` | bf16 | (1024, 2048) | 4.00 MiB |
| `w_idx_q` | bf16 | (512, 512) | 512.00 KiB |
| `w_idx_k` | bf16 | (2048, 64) | 256.00 KiB |
| `idx_k_ln_w` | bf16 | (64,) | 128 B |
| `idx_k_ln_b` | bf16 | (64,) | 128 B |
| `w_idx_w` | fp32 | (2048, 8) | 64.00 KiB |
| `ffn_norm_w` | bf16 | (2048,) | 4.00 KiB |
| `w1` | bf16 | (2048, 8192) | 32.00 MiB |
| `w3` | bf16 | (2048, 8192) | 32.00 MiB |
| `w2` | bf16 | (8192, 2048) | 32.00 MiB |

**`A_.._0` saved context** — 104.75 MiB = **1.64 KiB/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 256.00 KiB |
| `q_a` | bf16 | (65536, 512) | 64.00 MiB |
| `rstd_qa` | fp32 | (65536,) | 256.00 KiB |
| `kv_a` | bf16 | (65536, 288) | 36.00 MiB |
| `rstd_kva` | fp32 | (65536,) | 256.00 KiB |
| `lse` | fp32 | (16, 65536) | 4.00 MiB |

### kind `moe` (e.g. layer 2)

**`W_2` weights** — 1.52 GiB

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (2048,) | 4.00 KiB |
| `w_q_a` | bf16 | (2048, 512) | 2.00 MiB |
| `q_a_norm_w` | bf16 | (512,) | 1.00 KiB |
| `w_q_b` | bf16 | (512, 1536) | 1.50 MiB |
| `w_kv_a` | bf16 | (2048, 288) | 1.12 MiB |
| `kv_a_norm_w` | bf16 | (256,) | 512 B |
| `w_kv_b` | bf16 | (256, 2048) | 1.00 MiB |
| `wo` | bf16 | (1024, 2048) | 4.00 MiB |
| `w_idx_q` | bf16 | (512, 512) | 512.00 KiB |
| `w_idx_k` | bf16 | (2048, 64) | 256.00 KiB |
| `idx_k_ln_w` | bf16 | (64,) | 128 B |
| `idx_k_ln_b` | bf16 | (64,) | 128 B |
| `w_idx_w` | fp32 | (2048, 8) | 64.00 KiB |
| `ffn_norm_w` | bf16 | (2048,) | 4.00 KiB |
| `w_router` | bf16 | (2048, 128) | 512.00 KiB |
| `w_router_bias` | fp32 | (128,) | 512 B |
| `w13_experts` | bf16 | (128, 2048, 2048) | 1.00 GiB |
| `w2_experts` | bf16 | (128, 1024, 2048) | 512.00 MiB |
| `w_s13` | bf16 | (2048, 2048) | 8.00 MiB |
| `w_s2` | bf16 | (1024, 2048) | 4.00 MiB |

**`A_.._2` saved context** — 104.75 MiB = **1.64 KiB/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 256.00 KiB |
| `q_a` | bf16 | (65536, 512) | 64.00 MiB |
| `rstd_qa` | fp32 | (65536,) | 256.00 KiB |
| `kv_a` | bf16 | (65536, 288) | 36.00 MiB |
| `rstd_kva` | fp32 | (65536,) | 256.00 KiB |
| `lse` | fp32 | (16, 65536) | 4.00 MiB |

**`M_.._2` metadata** — 5.00 MiB = **80.0 B/token** (never recomputed)

| field | dtype | shape | bytes |
|---|---|---|---|
| `route_w` | bf16 | (65536, 8) | 1.00 MiB |
| `route_ids` | int32 | (65536, 8) | 2.00 MiB |
| `route_order` | int32 | (524288,) | 2.00 MiB |
| `route_offsets` | int32 | (129,) | 516 B |

**`W_head`** — 505.00 MiB

| field | dtype | shape | bytes |
|---|---|---|---|
| `w` | bf16 | (129280, 2048) | 505.00 MiB |
| `final_norm_w` | bf16 | (2048,) | 4.00 KiB |

## Tasks

### `prologue_round` — `RoundPrologue`

- example task: `prologue_round_0_0`
- inputs: `Aux_2` (1.50 KiB), `Aux_3` (1.50 KiB), `Aux_4` (1.50 KiB), `Aux_5` (1.50 KiB), `Aux_6` (1.50 KiB), `Aux_7` (1.50 KiB), `Aux_8` (1.50 KiB), `Aux_9` (1.50 KiB), `Aux_10` (1.50 KiB), `Aux_11` (1.50 KiB), `Aux_12` (1.50 KiB), `Aux_13` (1.50 KiB), `Aux_14` (1.50 KiB), `Aux_15` (1.50 KiB), `Aux_16` (1.50 KiB), `Aux_17` (1.50 KiB)
- outputs: `current_round_0_0` (4 B)
- mutates: `Aux_2`, `Aux_3`, `Aux_4`, `Aux_5`, `Aux_6`, `Aux_7`, `Aux_8`, `Aux_9`, `Aux_10`, `Aux_11`, `Aux_12`, `Aux_13`, `Aux_14`, `Aux_15`, `Aux_16`, `Aux_17`

### `embed_fwd` — `EmbedFwd`

- example task: `embed_fwd_0_0`
- inputs: `tokens_0_0` (256.00 KiB), `W_embed` (505.00 MiB)
- outputs: `y_embed_0_0` (256.00 MiB)
- mutates: —
- kernel calls:
    0. `index_select`

### `dsadense_fwd` — `Dsv32WarmupDenseBlockFwd`

- example task: `block_fwd_0_0_0`
- inputs: `y_embed_0_0` (256.00 MiB), `W_0` (106.45 MiB)
- outputs: `y_0_0_0` (256.00 MiB), `A_0_0_0` (104.75 MiB)
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

### `dsamoe_fwd` — `Dsv32WarmupMoeBlockFwd`

- example task: `block_fwd_0_0_2`
- inputs: `y_0_0_1` (256.00 MiB), `W_2` (1.52 GiB), `current_round_0_0` (4 B), `Aux_2` (1.50 KiB)
- outputs: `y_0_0_2` (256.00 MiB), `A_0_0_2` (104.75 MiB), `AuxTemp_0_0_2` (5.00 MiB)
- mutates: `Aux_2`
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

### `dsamoe_bwd` — `Dsv32WarmupMoeBlockBwd`

- example task: `block_bwd_0_0_17`
- inputs: `A_0_0_17` (104.75 MiB), `y_0_0_16` (256.00 MiB), `W_17` (1.52 GiB), `AuxTemp_0_0_17` (5.00 MiB)
- outputs: `dW_0_17` (832.50 KiB), `loss_0_0` (4 B)
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

### `dsadense_bwd` — `Dsv32WarmupDenseBlockBwd`

- example task: `block_bwd_0_0_1`
- inputs: `A_0_0_1` (104.75 MiB), `y_0_0_0` (256.00 MiB), `W_1` (106.45 MiB), `loss_0_0` (4 B)
- outputs: `dW_0_1` (832.50 KiB)
- mutates: `loss_0_0`
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

### `optimizer_block` — `AdamWStep`

- example task: `optimizer_0_17`
- inputs: `W_17` (1.52 GiB), `dW_0_17` (832.50 KiB), `O_17` (1.63 MiB)
- outputs: —
- mutates: `W_17`, `O_17`
- kernel calls:
    0. `adamw_step ×19`

