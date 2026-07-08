# dsv32 / `tiny` @ 16x4K: tasks, objects, kernels

GENERATED from `ShapedDsv32Config.tiny()` at run shape microbatch 16 × seq 4096 — regenerate with `python tools/gen_model_page.py --preset tiny --microbatch 16 --seq-len 4096`. All presets: [builtin_models.md](../../builtin_models.md); task-kind fleet index: [task_kinds.md](../../task_kinds.md).

Layer kinds (3 layers): `dense moe moe`

**Run shape**: microbatch 16 × seq_len 4096 = **65,536 tokens per round** (× 1 grad-accum round(s) per step). `A_*`/`M_*` objects are sized per round; bytes/token figures transfer to any run shape.

## Object summary

At this run shape (65,536 tokens/round). Token-scaled objects show bytes/token in parens. Details per kind below.

| object | scope | bytes |
|---|---|---|
| `W_i (dense)` | layer | 306,688 |
| `dW_i (dense)` | layer/step | 306,688 |
| `O_i (dense)` | layer | 613,376 |
| `A (dense)` | layer × round | 108,003,328 (1,648.0/token) |
| `M (dense)` | layer × round | 6,291,456 (96.0/token) |
| `W_i (moe)` | layer | 333,568 |
| `dW_i (moe)` | layer/step | 333,568 |
| `O_i (moe)` | layer | 667,136 |
| `A (moe)` | layer × round | 67,108,864 (1,024.0/token) |
| `M (moe)` | layer × round | 7,602,432 (116.0/token) |
| `W_head` | run | 131,328 |
| `W_embed` | run | 131,072 |
| `O_embed` | run | 262,144 |
| `O_head` | run | 262,656 |
| `hidden state (y)` | boundary buffer | 16,777,216 (256.0/token) |

### Aggregate totals (all layers, this run shape)

| type | objects | total bytes |
|---|---|---|
| W (all weights, incl. embed/head) | 5 | 1,236,224 |
| dW (all gradients, incl. metadata grads, per step) | 5 | 1,236,224 |
| O (all optimizer state) | 5 | 2,472,448 |
| A (all saved contexts, one round) | 3 | 242,221,056 (3,696.0/token) |
| M (all metadata, one round) | 3 | 21,496,320 (328.0/token) |

## Dims

| field | value |
|---|---|
| `d_model` | 128 |
| `n_heads` | 4 |
| `q_lora_rank` | 64 |
| `kv_lora_rank` | 32 |
| `qk_nope_dim` | 16 |
| `qk_rope_dim` | 8 |
| `v_head_dim` | 16 |
| `d_ff` | 256 |
| `first_k_dense` | 1 |
| `vocab_size` | 512 |
| `tokens` | 65536 |
| `seq_len` | 4096 |
| `rope_base` | 10000.0 |
| `opt_policy` | adamw |
| `index_n_heads` | 8 |
| `index_head_dim` | 32 |
| `index_topk` | 24 |
| `sparse_mode` | True |
| `train_indexer` | True |

## Objects, per layer kind

`dW_i` mirrors `W_i`'s fields at the grad dtypes; `O_i` holds the optimizer policy's state slots per field (adamw default: `m_f`+`v_f` at the opt dtype; sgd fields contribute none — see extending.md §6). `A_i`/`M_i` exist per (step, round).

### kind `dense` (e.g. layer 0)

**`W_0` weights** — 306,688 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (128,) | 256 |
| `w_q_a` | bf16 | (128, 64) | 16,384 |
| `q_a_norm_w` | bf16 | (64,) | 128 |
| `w_q_b` | bf16 | (64, 96) | 12,288 |
| `w_kv_a` | bf16 | (128, 40) | 10,240 |
| `kv_a_norm_w` | bf16 | (32,) | 64 |
| `w_kv_b` | bf16 | (32, 128) | 8,192 |
| `wo` | bf16 | (64, 128) | 16,384 |
| `w_idx_q` | bf16 | (64, 256) | 32,768 |
| `w_idx_k` | bf16 | (128, 32) | 8,192 |
| `idx_k_ln_w` | bf16 | (32,) | 64 |
| `idx_k_ln_b` | bf16 | (32,) | 64 |
| `w_idx_w` | fp32 | (128, 8) | 4,096 |
| `ffn_norm_w` | bf16 | (128,) | 256 |
| `w1` | bf16 | (128, 256) | 65,536 |
| `w3` | bf16 | (128, 256) | 65,536 |
| `w2` | bf16 | (256, 128) | 65,536 |

**`A_.._0` saved context** — 108,003,328 bytes = **1,648.0 bytes/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 262,144 |
| `q_a` | bf16 | (65536, 64) | 8,388,608 |
| `rstd_qa` | fp32 | (65536,) | 262,144 |
| `kv_a` | bf16 | (65536, 40) | 5,242,880 |
| `rstd_kva` | fp32 | (65536,) | 262,144 |
| `lse` | fp32 | (64, 4096) | 1,048,576 |
| `attn_out` | bf16 | (65536, 64) | 8,388,608 |
| `h_mid` | bf16 | (65536, 128) | 16,777,216 |
| `rstd_ffn` | fp32 | (65536,) | 262,144 |
| `x1` | bf16 | (65536, 256) | 33,554,432 |
| `x3` | bf16 | (65536, 256) | 33,554,432 |

**`M_.._0` metadata** — 6,291,456 bytes = **96.0 bytes/token** (never recomputed)

| field | dtype | shape | bytes |
|---|---|---|---|
| `dsa_idx` | int32 | (65536, 24) | 6,291,456 |

### kind `moe` (e.g. layer 1)

**`W_1` weights** — 333,568 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (128,) | 256 |
| `w_q_a` | bf16 | (128, 64) | 16,384 |
| `q_a_norm_w` | bf16 | (64,) | 128 |
| `w_q_b` | bf16 | (64, 96) | 12,288 |
| `w_kv_a` | bf16 | (128, 40) | 10,240 |
| `kv_a_norm_w` | bf16 | (32,) | 64 |
| `w_kv_b` | bf16 | (32, 128) | 8,192 |
| `wo` | bf16 | (64, 128) | 16,384 |
| `w_idx_q` | bf16 | (64, 256) | 32,768 |
| `w_idx_k` | bf16 | (128, 32) | 8,192 |
| `idx_k_ln_w` | bf16 | (32,) | 64 |
| `idx_k_ln_b` | bf16 | (32,) | 64 |
| `w_idx_w` | fp32 | (128, 8) | 4,096 |
| `ffn_norm_w` | bf16 | (128,) | 256 |
| `w_router` | bf16 | (128, 8) | 2,048 |
| `w_router_bias` | fp32 | (8,) | 32 |
| `w13_experts` | bf16 | (8, 128, 64) | 131,072 |
| `w2_experts` | bf16 | (8, 32, 128) | 65,536 |
| `w_s13` | bf16 | (128, 64) | 16,384 |
| `w_s2` | bf16 | (32, 128) | 8,192 |

**`A_.._1` saved context** — 67,108,864 bytes = **1,024.0 bytes/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 262,144 |
| `q_a` | bf16 | (65536, 64) | 8,388,608 |
| `rstd_qa` | fp32 | (65536,) | 262,144 |
| `kv_a` | bf16 | (65536, 40) | 5,242,880 |
| `rstd_kva` | fp32 | (65536,) | 262,144 |
| `lse` | fp32 | (64, 4096) | 1,048,576 |
| `attn_out` | bf16 | (65536, 64) | 8,388,608 |
| `h_mid` | bf16 | (65536, 128) | 16,777,216 |
| `rstd_ffn` | fp32 | (65536,) | 262,144 |
| `router_logits` | bf16 | (65536, 8) | 1,048,576 |
| `h13` | bf16 | (131072, 64) | 16,777,216 |
| `s13` | bf16 | (65536, 64) | 8,388,608 |

**`M_.._1` metadata** — 7,602,432 bytes = **116.0 bytes/token** (never recomputed)

| field | dtype | shape | bytes |
|---|---|---|---|
| `dsa_idx` | int32 | (65536, 24) | 6,291,456 |
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

### `dsadense_fwd` — `Dsv32DenseBlockFwd`

- example task: `block_fwd_0_0_0`
- inputs: `y_embed_0_0` (16,777,216B), `W_0` (306,688B)
- outputs: `y_0_0_0` (16,777,216B), `A_0_0_0` (108,003,328B), `M_0_0_0` (6,291,456B)
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
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): rmsnorm_fwd×2 → rope_fwd → rmsnorm_apply → rope_fwd×2 → rmsnorm_fwd → rope_fwd → dsa_index_scores → dsa_topk → dsa_sparse_attn_fwd → rmsnorm_fwd → swiglu_fwd_out

### `dsamoe_fwd` — `Dsv32MoeBlockFwd`

- example task: `block_fwd_0_0_1`
- inputs: `y_0_0_0` (16,777,216B), `W_1` (333,568B)
- outputs: `y_0_0_1` (16,777,216B), `A_0_0_1` (67,108,864B), `M_0_0_1` (7,602,432B)
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
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): rmsnorm_fwd×2 → rope_fwd → rmsnorm_apply → rope_fwd×2 → rmsnorm_fwd → rope_fwd → dsa_index_scores → dsa_topk → dsa_sparse_attn_fwd → rmsnorm_fwd → moe_topk_sigmoid_noaux → moe_sort → moe_dispatch_fwd → moe_grouped_mm_fwd → swiglu_packed_fwd → moe_grouped_mm_fwd → swiglu_packed_fwd → moe_combine_fwd

### `head_loss` — `HeadLoss`

- example task: `head_loss_0_0`
- inputs: `y_0_0_2` (16,777,216B), `targets_0_0` (262,144B), `W_head` (131,328B)
- outputs: `dy_0_0_2` (16,777,216B), `loss_0_0` (4B), `dW_head_0` (131,328B)
- mutates: —
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): rmsnorm_fwd → ce_loss_fwd_bwd → rmsnorm_bwd

### `optimizer_head` — `AdamWStep`

- example task: `optimizer_head_0`
- inputs: `W_head` (131,328B), `dW_head_0` (131,328B), `O_head` (262,656B)
- outputs: —
- mutates: `W_head`, `O_head`
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): adamw_step×2

### `dsamoe_bwd` — `Dsv32MoeBlockBwd`

- example task: `block_bwd_0_0_2`
- inputs: `dy_0_0_2` (16,777,216B), `A_0_0_2` (67,108,864B), `y_0_0_1` (16,777,216B), `W_2` (333,568B), `M_0_0_2` (7,602,432B)
- outputs: `dy_0_0_1` (16,777,216B), `dW_0_2` (333,568B)
- mutates: —
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): rmsnorm_apply → moe_dispatch_fwd×2 → swiglu_packed_fwd → moe_grouped_mm_dgrad → moe_rowdot → moe_scale_rows → moe_grouped_mm_wgrad → moe_scale_rows → swiglu_packed_bwd → moe_grouped_mm_wgrad → moe_grouped_mm_dgrad → moe_dispatch_bwd → moe_router_bwd_sigmoid → moe_seq_aux_grad → swiglu_packed_fwd → swiglu_packed_bwd → rmsnorm_bwd → rmsnorm_apply → rope_fwd → rmsnorm_apply → rope_fwd → dsa_sparse_attn_bwd → rmsnorm_apply → rope_fwd×2 → dsa_index_scores → dsa_probs_sum → dsa_index_bwd → rope_bwd×3 → rmsnorm_bwd → rope_bwd → rmsnorm_bwd×2

### `optimizer_block` — `AdamWStep`

- example task: `optimizer_0_2`
- inputs: `W_2` (333,568B), `dW_0_2` (333,568B), `O_2` (667,136B)
- outputs: —
- mutates: `W_2`, `O_2`
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): adamw_step×19

### `dsadense_bwd` — `Dsv32DenseBlockBwd`

- example task: `block_bwd_0_0_0`
- inputs: `dy_0_0_0` (16,777,216B), `A_0_0_0` (108,003,328B), `y_embed_0_0` (16,777,216B), `W_0` (306,688B), `M_0_0_0` (6,291,456B)
- outputs: `dy_embed_0_0` (16,777,216B), `dW_0_0` (306,688B)
- mutates: —
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): rmsnorm_apply → swiglu_fwd_out → swiglu_bwd → rmsnorm_bwd → rmsnorm_apply → rope_fwd → rmsnorm_apply → rope_fwd → dsa_sparse_attn_bwd → rmsnorm_apply → rope_fwd×2 → dsa_index_scores → dsa_probs_sum → dsa_index_bwd → rope_bwd×3 → rmsnorm_bwd → rope_bwd → rmsnorm_bwd×2

### `embed_bwd` — `EmbedBwd`

- example task: `embed_bwd_0_0`
- inputs: `dy_embed_0_0` (16,777,216B), `tokens_0_0` (262,144B)
- outputs: `dW_embed_0` (131,072B)
- mutates: —
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): embed_bwd_accum

### `optimizer_embed` — `AdamWStep`

- example task: `optimizer_embed_0`
- inputs: `W_embed` (131,072B), `dW_embed_0` (131,072B), `O_embed` (262,144B)
- outputs: —
- mutates: `W_embed`, `O_embed`
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): adamw_step

