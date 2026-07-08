# dsv32 / `dsv32_mini` @ 16x4K: tasks, objects, kernels

GENERATED from `ShapedDsv32Config.dsv32_mini()` at run shape microbatch 16 × seq 4096 — regenerate with `python tools/gen_model_page.py --preset dsv32_mini --microbatch 16 --seq-len 4096`. All presets: [builtin_models.md](../../builtin_models.md); task-kind fleet index: [task_kinds.md](../../task_kinds.md).

Layer kinds (18 layers): `dense dense moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe`

**Run shape**: microbatch 16 × seq_len 4096 = **65,536 tokens per round** (× 1 grad-accum round(s) per step). `A_*`/`M_*` objects are sized per round; bytes/token figures transfer to any run shape.

## Object summary

At this run shape (65,536 tokens/round). Token-scaled objects show bytes/token in parens. Details per kind below.

| object | scope | bytes |
|---|---|---|
| `W_i (dense)` | layer | 111,618,048 |
| `dW_i (dense)` | layer/step | 111,618,048 |
| `O_i (dense)` | layer | 223,236,096 |
| `A (dense)` | layer × round | 2,660,237,312 (40,592.0/token) |
| `M (dense)` | layer × round | 268,435,456 (4,096.0/token) |
| `W_i (moe)` | layer | 1,634,675,200 |
| `dW_i (moe)` | layer/step | 1,634,675,200 |
| `O_i (moe)` | layer | 3,269,350,400 |
| `A (moe)` | layer × round | 2,945,449,984 (44,944.0/token) |
| `M (moe)` | layer × round | 273,679,104 (4,176.0/token) |
| `W_head` | run | 529,534,976 |
| `W_embed` | run | 529,530,880 |
| `O_embed` | run | 1,059,061,760 |
| `O_head` | run | 1,059,069,952 |
| `hidden state (y)` | boundary buffer | 268,435,456 (4,096.0/token) |

### Aggregate totals (all layers, this run shape)

| type | objects | total bytes |
|---|---|---|
| W (all weights, incl. embed/head) | 20 | 27,437,105,152 |
| dW (all gradients, per step) | 20 | 27,437,105,152 |
| O (all optimizer state) | 20 | 54,874,210,304 |
| A (all saved activations, one round) | 18 | 52,447,674,368 (800,288.0/token) |
| M (all metadata, one round) | 18 | 4,915,736,576 (75,008.2/token) |

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
| `opt_policy` | adamw |
| `index_n_heads` | 8 |
| `index_head_dim` | 64 |
| `index_topk` | 1024 |
| `sparse_mode` | True |
| `train_indexer` | True |

## Objects, per layer kind

`dW_i` mirrors `W_i`'s fields at the grad dtypes; `O_i` holds the optimizer policy's state slots per field (adamw default: `m_f`+`v_f` at the opt dtype; sgd fields contribute none — see extending.md §6). `A_i`/`M_i` exist per (step, round).

### kind `dense` (e.g. layer 0)

**`W_0` weights** — 111,618,048 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (2048,) | 4,096 |
| `w_q_a` | bf16 | (2048, 512) | 2,097,152 |
| `q_a_norm_w` | bf16 | (512,) | 1,024 |
| `w_q_b` | bf16 | (512, 1536) | 1,572,864 |
| `w_kv_a` | bf16 | (2048, 288) | 1,179,648 |
| `kv_a_norm_w` | bf16 | (256,) | 512 |
| `w_kv_b` | bf16 | (256, 2048) | 1,048,576 |
| `wo` | bf16 | (1024, 2048) | 4,194,304 |
| `w_idx_q` | bf16 | (512, 512) | 524,288 |
| `w_idx_k` | bf16 | (2048, 64) | 262,144 |
| `idx_k_ln_w` | bf16 | (64,) | 128 |
| `idx_k_ln_b` | bf16 | (64,) | 128 |
| `w_idx_w` | fp32 | (2048, 8) | 65,536 |
| `ffn_norm_w` | bf16 | (2048,) | 4,096 |
| `w1` | bf16 | (2048, 8192) | 33,554,432 |
| `w3` | bf16 | (2048, 8192) | 33,554,432 |
| `w2` | bf16 | (8192, 2048) | 33,554,432 |

**`A_.._0` saved context** — 2,660,237,312 bytes = **40,592.0 bytes/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 262,144 |
| `q_a` | bf16 | (65536, 512) | 67,108,864 |
| `rstd_qa` | fp32 | (65536,) | 262,144 |
| `kv_a` | bf16 | (65536, 288) | 37,748,736 |
| `rstd_kva` | fp32 | (65536,) | 262,144 |
| `lse` | fp32 | (256, 4096) | 4,194,304 |
| `attn_out` | bf16 | (65536, 1024) | 134,217,728 |
| `h_mid` | bf16 | (65536, 2048) | 268,435,456 |
| `rstd_ffn` | fp32 | (65536,) | 262,144 |
| `x1` | bf16 | (65536, 8192) | 1,073,741,824 |
| `x3` | bf16 | (65536, 8192) | 1,073,741,824 |

**`M_.._0` metadata** — 268,435,456 bytes = **4,096.0 bytes/token** (never recomputed)

| field | dtype | shape | bytes |
|---|---|---|---|
| `dsa_idx` | int32 | (65536, 1024) | 268,435,456 |

### kind `moe` (e.g. layer 2)

**`W_2` weights** — 1,634,675,200 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (2048,) | 4,096 |
| `w_q_a` | bf16 | (2048, 512) | 2,097,152 |
| `q_a_norm_w` | bf16 | (512,) | 1,024 |
| `w_q_b` | bf16 | (512, 1536) | 1,572,864 |
| `w_kv_a` | bf16 | (2048, 288) | 1,179,648 |
| `kv_a_norm_w` | bf16 | (256,) | 512 |
| `w_kv_b` | bf16 | (256, 2048) | 1,048,576 |
| `wo` | bf16 | (1024, 2048) | 4,194,304 |
| `w_idx_q` | bf16 | (512, 512) | 524,288 |
| `w_idx_k` | bf16 | (2048, 64) | 262,144 |
| `idx_k_ln_w` | bf16 | (64,) | 128 |
| `idx_k_ln_b` | bf16 | (64,) | 128 |
| `w_idx_w` | fp32 | (2048, 8) | 65,536 |
| `ffn_norm_w` | bf16 | (2048,) | 4,096 |
| `w_router` | bf16 | (2048, 128) | 524,288 |
| `w_router_bias` | fp32 | (128,) | 512 |
| `w13_experts` | bf16 | (128, 2048, 2048) | 1,073,741,824 |
| `w2_experts` | bf16 | (128, 1024, 2048) | 536,870,912 |
| `w_s13` | bf16 | (2048, 2048) | 8,388,608 |
| `w_s2` | bf16 | (1024, 2048) | 4,194,304 |

**`A_.._2` saved context** — 2,945,449,984 bytes = **44,944.0 bytes/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 262,144 |
| `q_a` | bf16 | (65536, 512) | 67,108,864 |
| `rstd_qa` | fp32 | (65536,) | 262,144 |
| `kv_a` | bf16 | (65536, 288) | 37,748,736 |
| `rstd_kva` | fp32 | (65536,) | 262,144 |
| `lse` | fp32 | (256, 4096) | 4,194,304 |
| `attn_out` | bf16 | (65536, 1024) | 134,217,728 |
| `h_mid` | bf16 | (65536, 2048) | 268,435,456 |
| `rstd_ffn` | fp32 | (65536,) | 262,144 |
| `router_logits` | bf16 | (65536, 128) | 16,777,216 |
| `h13` | bf16 | (524288, 2048) | 2,147,483,648 |
| `s13` | bf16 | (65536, 2048) | 268,435,456 |

**`M_.._2` metadata** — 273,679,104 bytes = **4,176.0 bytes/token** (never recomputed)

| field | dtype | shape | bytes |
|---|---|---|---|
| `dsa_idx` | int32 | (65536, 1024) | 268,435,456 |
| `route_w` | bf16 | (65536, 8) | 1,048,576 |
| `route_ids` | int32 | (65536, 8) | 2,097,152 |
| `route_order` | int32 | (524288,) | 2,097,152 |
| `route_offsets` | int32 | (129,) | 516 |

**`W_head`** — 529,534,976 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `w` | bf16 | (129280, 2048) | 529,530,880 |
| `final_norm_w` | bf16 | (2048,) | 4,096 |

## Tasks

### `embed_fwd` — `EmbedFwd`

- example task: `embed_fwd_0_0`
- inputs: `tokens_0_0` (262,144B), `W_embed` (529,530,880B)
- outputs: `y_embed_0_0` (268,435,456B)
- mutates: —
- kernel calls:
    0. `index_select`

### `dsadense_fwd` — `Dsv32DenseBlockFwd`

- example task: `block_fwd_0_0_0`
- inputs: `y_embed_0_0` (268,435,456B), `W_0` (111,618,048B)
- outputs: `y_0_0_0` (268,435,456B), `A_0_0_0` (2,660,237,312B), `M_0_0_0` (268,435,456B)
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

- example task: `block_fwd_0_0_2`
- inputs: `y_0_0_1` (268,435,456B), `W_2` (1,634,675,200B)
- outputs: `y_0_0_2` (268,435,456B), `A_0_0_2` (2,945,449,984B), `M_0_0_2` (273,679,104B)
- mutates: —
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
    - `moe_dispatch`:
        21. `moe_sort`
        22. `moe_dispatch_fwd`
    - `moe_experts13`:
        23. `moe_grouped_mm_fwd`
    - `moe_shared`:
        24. `mm`
    - `moe_experts2_combine`:
        25. `swiglu_packed_fwd`
        26. `moe_grouped_mm_fwd`
        27. `swiglu_packed_fwd`
        28. `mm`
        29. `moe_combine_fwd`

### `head_loss` — `HeadLoss`

- example task: `head_loss_0_0`
- inputs: `y_0_0_17` (268,435,456B), `targets_0_0` (262,144B), `W_head` (529,534,976B)
- outputs: `dy_0_0_17` (268,435,456B), `loss_0_0` (4B), `dW_head_0` (529,534,976B)
- mutates: —
- kernel calls:
    0. `rmsnorm_fwd`
    1. `mm`
    2. `ce_loss_fwd_bwd`
    3. `mm ×2`
    4. `rmsnorm_bwd`

### `optimizer_head` — `AdamWStep`

- example task: `optimizer_head_0`
- inputs: `W_head` (529,534,976B), `dW_head_0` (529,534,976B), `O_head` (1,059,069,952B)
- outputs: —
- mutates: `W_head`, `O_head`
- kernel calls:
    0. `adamw_step ×2`

### `dsamoe_bwd` — `Dsv32MoeBlockBwd`

- example task: `block_bwd_0_0_17`
- inputs: `dy_0_0_17` (268,435,456B), `A_0_0_17` (2,945,449,984B), `y_0_0_16` (268,435,456B), `W_17` (1,634,675,200B), `M_0_0_17` (273,679,104B)
- outputs: `dy_0_0_16` (268,435,456B), `dW_0_17` (1,634,675,200B)
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

### `optimizer_block` — `AdamWStep`

- example task: `optimizer_0_17`
- inputs: `W_17` (1,634,675,200B), `dW_0_17` (1,634,675,200B), `O_17` (3,269,350,400B)
- outputs: —
- mutates: `W_17`, `O_17`
- kernel calls:
    0. `adamw_step ×19`

### `dsadense_bwd` — `Dsv32DenseBlockBwd`

- example task: `block_bwd_0_0_1`
- inputs: `dy_0_0_1` (268,435,456B), `A_0_0_1` (2,660,237,312B), `y_0_0_0` (268,435,456B), `W_1` (111,618,048B), `M_0_0_1` (268,435,456B)
- outputs: `dy_0_0_0` (268,435,456B), `dW_0_1` (111,618,048B)
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
- inputs: `dy_embed_0_0` (268,435,456B), `tokens_0_0` (262,144B)
- outputs: `dW_embed_0` (529,530,880B)
- mutates: —
- kernel calls:
    0. `embed_bwd_accum`

### `optimizer_embed` — `AdamWStep`

- example task: `optimizer_embed_0`
- inputs: `W_embed` (529,530,880B), `dW_embed_0` (529,530,880B), `O_embed` (1,059,061,760B)
- outputs: —
- mutates: `W_embed`, `O_embed`
- kernel calls:
    0. `adamw_step`

