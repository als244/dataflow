# dsv32 / `glm5` @ 16x4K: tasks, objects, kernels

GENERATED from `ShapedDsv32Config.glm5()` at run shape microbatch 16 × seq 4096 — regenerate with `python tools/gen_model_page.py --preset glm5 --microbatch 16 --seq-len 4096`. All presets: [builtin_models.md](../../builtin_models.md); task-kind fleet index: [task_kinds.md](../../task_kinds.md).

Layer kinds (78 layers): `dense dense dense moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe`

**Run shape**: microbatch 16 × seq_len 4096 = **65,536 tokens per round** (× 1 grad-accum round(s) per step). `A_*`/`M_*` objects are sized per round; bytes/token figures transfer to any run shape.

## Object summary

At this run shape (65,536 tokens/round). Token-scaled objects show bytes/token in parens. Details per kind below.

| object | scope | bytes |
|---|---|---|
| `W_i (dense)` | layer | 802,190,848 |
| `dW_i (dense)` | layer/step | 802,190,848 |
| `O_i (dense)` | layer | 1,604,381,696 |
| `A (dense)` | layer × round | 6,535,774,208 (99,728.0/token) |
| `M (dense)` | layer × round | 536,870,912 (8,192.0/token) |
| `W_i (moe)` | layer | 19,755,203,072 |
| `dW_i (moe)` | layer/step | 19,755,203,072 |
| `O_i (moe)` | layer | 39,510,406,144 |
| `A (moe)` | layer × round | 8,179,941,376 (124,816.0/token) |
| `M (moe)` | layer × round | 542,115,072 (8,272.0/token) |
| `W_head` | run | 1,903,177,728 |
| `W_embed` | run | 1,903,165,440 |
| `O_embed` | run | 3,806,330,880 |
| `O_head` | run | 3,806,355,456 |
| `hidden state (y)` | boundary buffer | 805,306,368 (12,288.0/token) |

### Aggregate totals (all layers, this run shape)

| type | objects | total bytes |
|---|---|---|
| W (all weights, incl. embed/head) | 80 | 1,487,853,146,112 |
| dW (all gradients, per step) | 80 | 1,487,853,146,112 |
| O (all optimizer state) | 80 | 2,975,706,292,224 |
| A (all saved activations, one round) | 78 | 633,102,925,824 (9,660,384.0/token) |
| M (all metadata, one round) | 78 | 42,269,243,136 (644,977.5/token) |

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
| `opt_policy` | adamw |
| `index_n_heads` | 32 |
| `index_head_dim` | 128 |
| `index_topk` | 2048 |
| `sparse_mode` | True |
| `train_indexer` | True |

## Objects, per layer kind

`dW_i` mirrors `W_i`'s fields at the grad dtypes; `O_i` holds the optimizer policy's state slots per field (adamw default: `m_f`+`v_f` at the opt dtype; sgd fields contribute none — see extending.md §6). `A_i`/`M_i` exist per (step, round).

### kind `dense` (e.g. layer 0)

**`W_0` weights** — 802,190,848 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (6144,) | 12,288 |
| `w_q_a` | bf16 | (6144, 2048) | 25,165,824 |
| `q_a_norm_w` | bf16 | (2048,) | 4,096 |
| `w_q_b` | bf16 | (2048, 16384) | 67,108,864 |
| `w_kv_a` | bf16 | (6144, 576) | 7,077,888 |
| `kv_a_norm_w` | bf16 | (512,) | 1,024 |
| `w_kv_b` | bf16 | (512, 28672) | 29,360,128 |
| `wo` | bf16 | (16384, 6144) | 201,326,592 |
| `w_idx_q` | bf16 | (2048, 4096) | 16,777,216 |
| `w_idx_k` | bf16 | (6144, 128) | 1,572,864 |
| `idx_k_ln_w` | bf16 | (128,) | 256 |
| `idx_k_ln_b` | bf16 | (128,) | 256 |
| `w_idx_w` | fp32 | (6144, 32) | 786,432 |
| `ffn_norm_w` | bf16 | (6144,) | 12,288 |
| `w1` | bf16 | (6144, 12288) | 150,994,944 |
| `w3` | bf16 | (6144, 12288) | 150,994,944 |
| `w2` | bf16 | (12288, 6144) | 150,994,944 |

**`A_.._0` saved context** — 6,535,774,208 bytes = **99,728.0 bytes/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 262,144 |
| `q_a` | bf16 | (65536, 2048) | 268,435,456 |
| `rstd_qa` | fp32 | (65536,) | 262,144 |
| `kv_a` | bf16 | (65536, 576) | 75,497,472 |
| `rstd_kva` | fp32 | (65536,) | 262,144 |
| `lse` | fp32 | (1024, 4096) | 16,777,216 |
| `attn_out` | bf16 | (65536, 16384) | 2,147,483,648 |
| `h_mid` | bf16 | (65536, 6144) | 805,306,368 |
| `rstd_ffn` | fp32 | (65536,) | 262,144 |
| `x1` | bf16 | (65536, 12288) | 1,610,612,736 |
| `x3` | bf16 | (65536, 12288) | 1,610,612,736 |

**`M_.._0` metadata** — 536,870,912 bytes = **8,192.0 bytes/token** (never recomputed)

| field | dtype | shape | bytes |
|---|---|---|---|
| `dsa_idx` | int32 | (65536, 2048) | 536,870,912 |

### kind `moe` (e.g. layer 3)

**`W_3` weights** — 19,755,203,072 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (6144,) | 12,288 |
| `w_q_a` | bf16 | (6144, 2048) | 25,165,824 |
| `q_a_norm_w` | bf16 | (2048,) | 4,096 |
| `w_q_b` | bf16 | (2048, 16384) | 67,108,864 |
| `w_kv_a` | bf16 | (6144, 576) | 7,077,888 |
| `kv_a_norm_w` | bf16 | (512,) | 1,024 |
| `w_kv_b` | bf16 | (512, 28672) | 29,360,128 |
| `wo` | bf16 | (16384, 6144) | 201,326,592 |
| `w_idx_q` | bf16 | (2048, 4096) | 16,777,216 |
| `w_idx_k` | bf16 | (6144, 128) | 1,572,864 |
| `idx_k_ln_w` | bf16 | (128,) | 256 |
| `idx_k_ln_b` | bf16 | (128,) | 256 |
| `w_idx_w` | fp32 | (6144, 32) | 786,432 |
| `ffn_norm_w` | bf16 | (6144,) | 12,288 |
| `w_router` | bf16 | (6144, 256) | 3,145,728 |
| `w_router_bias` | fp32 | (256,) | 1,024 |
| `w13_experts` | bf16 | (256, 6144, 4096) | 12,884,901,888 |
| `w2_experts` | bf16 | (256, 2048, 6144) | 6,442,450,944 |
| `w_s13` | bf16 | (6144, 4096) | 50,331,648 |
| `w_s2` | bf16 | (2048, 6144) | 25,165,824 |

**`A_.._3` saved context** — 8,179,941,376 bytes = **124,816.0 bytes/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 262,144 |
| `q_a` | bf16 | (65536, 2048) | 268,435,456 |
| `rstd_qa` | fp32 | (65536,) | 262,144 |
| `kv_a` | bf16 | (65536, 576) | 75,497,472 |
| `rstd_kva` | fp32 | (65536,) | 262,144 |
| `lse` | fp32 | (1024, 4096) | 16,777,216 |
| `attn_out` | bf16 | (65536, 16384) | 2,147,483,648 |
| `h_mid` | bf16 | (65536, 6144) | 805,306,368 |
| `rstd_ffn` | fp32 | (65536,) | 262,144 |
| `router_logits` | bf16 | (65536, 256) | 33,554,432 |
| `h13` | bf16 | (524288, 4096) | 4,294,967,296 |
| `s13` | bf16 | (65536, 4096) | 536,870,912 |

**`M_.._3` metadata** — 542,115,072 bytes = **8,272.0 bytes/token** (never recomputed)

| field | dtype | shape | bytes |
|---|---|---|---|
| `dsa_idx` | int32 | (65536, 2048) | 536,870,912 |
| `route_w` | bf16 | (65536, 8) | 1,048,576 |
| `route_ids` | int32 | (65536, 8) | 2,097,152 |
| `route_order` | int32 | (524288,) | 2,097,152 |
| `route_offsets` | int32 | (257,) | 1,028 |

**`W_head`** — 1,903,177,728 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `w` | bf16 | (154880, 6144) | 1,903,165,440 |
| `final_norm_w` | bf16 | (6144,) | 12,288 |

## Tasks

### `embed_fwd` — `EmbedFwd`

- example task: `embed_fwd_0_0`
- inputs: `tokens_0_0` (262,144B), `W_embed` (1,903,165,440B)
- outputs: `y_embed_0_0` (805,306,368B)
- mutates: —
- kernel calls:
    0. `index_select`

### `dsadense_fwd` — `Dsv32DenseBlockFwd`

- example task: `block_fwd_0_0_0`
- inputs: `y_embed_0_0` (805,306,368B), `W_0` (802,190,848B)
- outputs: `y_0_0_0` (805,306,368B), `A_0_0_0` (6,535,774,208B), `M_0_0_0` (536,870,912B)
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
    - `attn_norm`: `rmsnorm_fwd`
    - `mla_q`: `mm`, `rmsnorm_fwd`, `mm`, `rope_fwd`, `rmsnorm_apply`
    - `mla_kv`: `mm`, `rope_fwd`, `mm`, `rope_fwd`, `mm ×2`, `rmsnorm_fwd`, `rope_fwd`, `mm`
    - `dsa_select`: `dsa_index_scores`, `dsa_topk`
    - `dsa_attn`: `dsa_sparse_attn_fwd`
    - `resid1_norm2`: `addmm`, `rmsnorm_fwd`
    - `up_proj`: `mm ×2`
    - `swiglu`: `swiglu_fwd_out`
    - `down_resid`: `addmm`

### `dsamoe_fwd` — `Dsv32MoeBlockFwd`

- example task: `block_fwd_0_0_3`
- inputs: `y_0_0_2` (805,306,368B), `W_3` (19,755,203,072B)
- outputs: `y_0_0_3` (805,306,368B), `A_0_0_3` (8,179,941,376B), `M_0_0_3` (542,115,072B)
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
    - `attn_norm`: `rmsnorm_fwd`
    - `mla_q`: `mm`, `rmsnorm_fwd`, `mm`, `rope_fwd`, `rmsnorm_apply`
    - `mla_kv`: `mm`, `rope_fwd`, `mm`, `rope_fwd`, `mm ×2`, `rmsnorm_fwd`, `rope_fwd`, `mm`
    - `dsa_select`: `dsa_index_scores`, `dsa_topk`
    - `dsa_attn`: `dsa_sparse_attn_fwd`
    - `resid1_norm2`: `addmm`, `rmsnorm_fwd`
    - `moe_route`: `mm`, `moe_topk_sigmoid_noaux`
    - `moe_dispatch`: `moe_sort`, `moe_dispatch_fwd`
    - `moe_experts13`: `moe_grouped_mm_fwd`
    - `moe_shared`: `mm`
    - `moe_experts2_combine`: `swiglu_packed_fwd`, `moe_grouped_mm_fwd`, `swiglu_packed_fwd`, `mm`, `moe_combine_fwd`

### `head_loss` — `HeadLoss`

- example task: `head_loss_0_0`
- inputs: `y_0_0_77` (805,306,368B), `targets_0_0` (262,144B), `W_head` (1,903,177,728B)
- outputs: `dy_0_0_77` (805,306,368B), `loss_0_0` (4B), `dW_head_0` (1,903,177,728B)
- mutates: —
- kernel calls:
    0. `rmsnorm_fwd`
    1. `mm`
    2. `ce_loss_fwd_bwd`
    3. `mm ×2`
    4. `rmsnorm_bwd`

### `optimizer_head` — `AdamWStep`

- example task: `optimizer_head_0`
- inputs: `W_head` (1,903,177,728B), `dW_head_0` (1,903,177,728B), `O_head` (3,806,355,456B)
- outputs: —
- mutates: `W_head`, `O_head`
- kernel calls:
    0. `adamw_step ×2`

### `dsamoe_bwd` — `Dsv32MoeBlockBwd`

- example task: `block_bwd_0_0_77`
- inputs: `dy_0_0_77` (805,306,368B), `A_0_0_77` (8,179,941,376B), `y_0_0_76` (805,306,368B), `W_77` (19,755,203,072B), `M_0_0_77` (542,115,072B)
- outputs: `dy_0_0_76` (805,306,368B), `dW_0_77` (19,755,203,072B)
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

- example task: `optimizer_0_77`
- inputs: `W_77` (19,755,203,072B), `dW_0_77` (19,755,203,072B), `O_77` (39,510,406,144B)
- outputs: —
- mutates: `W_77`, `O_77`
- kernel calls:
    0. `adamw_step ×19`

### `dsadense_bwd` — `Dsv32DenseBlockBwd`

- example task: `block_bwd_0_0_2`
- inputs: `dy_0_0_2` (805,306,368B), `A_0_0_2` (6,535,774,208B), `y_0_0_1` (805,306,368B), `W_2` (802,190,848B), `M_0_0_2` (536,870,912B)
- outputs: `dy_0_0_1` (805,306,368B), `dW_0_2` (802,190,848B)
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
- inputs: `dy_embed_0_0` (805,306,368B), `tokens_0_0` (262,144B)
- outputs: `dW_embed_0` (1,903,165,440B)
- mutates: —
- kernel calls:
    0. `embed_bwd_accum`

### `optimizer_embed` — `AdamWStep`

- example task: `optimizer_embed_0`
- inputs: `W_embed` (1,903,165,440B), `dW_embed_0` (1,903,165,440B), `O_embed` (3,806,330,880B)
- outputs: —
- mutates: `W_embed`, `O_embed`
- kernel calls:
    0. `adamw_step`

