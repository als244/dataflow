# glm52 / `glm52_mini` @ 16x4K: tasks, objects, kernels

GENERATED from `ShapedGlm52Config.glm52_mini()` at run shape microbatch 16 × seq 4096 — regenerate with `python tools/gen_model_page.py --preset glm52_mini --microbatch 16 --seq-len 4096`. All presets: [builtin_models.md](../../builtin_models.md); task-kind fleet index: [task_kinds.md](../../task_kinds.md).

Layer kinds (18 layers): `gdl gdl gml gmf gmf gmf gml gmf gmf gmf gml gmf gmf gmf gml gmf gmf gmf`

**Run shape**: microbatch 16 × seq_len 4096 = **65,536 tokens per round** (× 1 grad-accum round(s) per step). `A_*`/`M_*` objects are sized per round; bytes/token figures transfer to any run shape.

## Object summary

At this run shape (65,536 tokens/round). Token-scaled objects show bytes/token in parens. Details per kind below.

| object | scope | bytes |
|---|---|---|
| `W_i (gdl)` | layer | 111,618,048 |
| `dW_i (gdl)` | layer/step | 111,618,048 |
| `O_i (gdl)` | layer | 223,236,096 |
| `A (gdl)` | layer × round | 2,660,237,312 (40,592.0/token) |
| `M (gdl)` | layer × round | 268,435,456 (4,096.0/token) |
| `W_i (gml)` | layer | 1,634,675,200 |
| `dW_i (gml)` | layer/step | 1,634,675,200 |
| `O_i (gml)` | layer | 3,269,350,400 |
| `A (gml)` | layer × round | 2,945,449,984 (44,944.0/token) |
| `M (gml)` | layer × round | 273,679,104 (4,176.0/token) |
| `W_i (gmf)` | layer | 1,633,822,720 |
| `dW_i (gmf)` | layer/step | 1,633,822,720 |
| `O_i (gmf)` | layer | 3,267,645,440 |
| `A (gmf)` | layer × round | 2,945,449,984 (44,944.0/token) |
| `M (gmf)` | layer × round | 5,243,648 (80.0/token) |
| `W_head` | run | 529,534,976 |
| `W_embed` | run | 529,530,880 |
| `O_embed` | run | 1,059,061,760 |
| `O_head` | run | 1,059,069,952 |
| `hidden state (y)` | boundary buffer | 268,435,456 (4,096.0/token) |

### Aggregate totals (all layers, this run shape)

| type | objects | total bytes |
|---|---|---|
| W (all weights, incl. embed/head) | 20 | 27,426,875,392 |
| dW (all gradients, incl. metadata grads, per step) | 24 | 28,500,617,216 |
| O (all optimizer state) | 20 | 54,853,750,784 |
| A (all saved contexts, one round) | 18 | 52,447,674,368 (800,288.0/token) |
| M (all metadata, one round) | 18 | 1,694,511,104 (25,856.2/token) |

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

### kind `gdl` (e.g. layer 0)

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

### kind `gml` (e.g. layer 2)

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

### kind `gmf` (e.g. layer 3)

**`W_3` weights** — 1,633,822,720 bytes

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
| `ffn_norm_w` | bf16 | (2048,) | 4,096 |
| `w_router` | bf16 | (2048, 128) | 524,288 |
| `w_router_bias` | fp32 | (128,) | 512 |
| `w13_experts` | bf16 | (128, 2048, 2048) | 1,073,741,824 |
| `w2_experts` | bf16 | (128, 1024, 2048) | 536,870,912 |
| `w_s13` | bf16 | (2048, 2048) | 8,388,608 |
| `w_s2` | bf16 | (1024, 2048) | 4,194,304 |

**`A_.._3` saved context** — 2,945,449,984 bytes = **44,944.0 bytes/token** (per (step, round))

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

**`M_.._3` metadata** — 5,243,648 bytes = **80.0 bytes/token** (never recomputed)

| field | dtype | shape | bytes |
|---|---|---|---|
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
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): index_select

### `gdl_fwd` — `Glm52DlBlockFwd`

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
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): rmsnorm_fwd → mm → rmsnorm_fwd → mm → rope_fwd → rmsnorm_apply → mm → rope_fwd → mm → rope_fwd → mm×2 → rmsnorm_fwd → rope_fwd → mm → dsa_index_scores → dsa_topk → dsa_sparse_attn_fwd → addmm → rmsnorm_fwd → mm×2 → swiglu_fwd_out → addmm

### `gml_fwd` — `Glm52MlBlockFwd`

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
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): rmsnorm_fwd → mm → rmsnorm_fwd → mm → rope_fwd → rmsnorm_apply → mm → rope_fwd → mm → rope_fwd → mm×2 → rmsnorm_fwd → rope_fwd → mm → dsa_index_scores → dsa_topk → dsa_sparse_attn_fwd → addmm → rmsnorm_fwd → mm → moe_topk_sigmoid_noaux → moe_sort → moe_dispatch_fwd → moe_grouped_mm_fwd → mm → swiglu_packed_fwd → moe_grouped_mm_fwd → swiglu_packed_fwd → mm → moe_combine_fwd

### `gmf_fwd` — `Glm52MfBlockFwd`

- example task: `block_fwd_0_0_3`
- inputs: `y_0_0_2` (268,435,456B), `W_3` (1,633,822,720B), `M_0_0_2` (273,679,104B)
- outputs: `y_0_0_3` (268,435,456B), `A_0_0_3` (2,945,449,984B), `M_0_0_3` (5,243,648B)
- mutates: —
- stages (name — emitted ctx fields):
    0. `attn_norm` — rstd_attn
    1. `mla_q` — q_a, rstd_qa
    2. `mla_kv` — kv_a, rstd_kva
    3. `dsa_attn` — lse, attn_out
    4. `resid1_norm2` — h_mid, rstd_ffn
    5. `moe_route` — router_logits
    6. `moe_dispatch` — —
    7. `moe_experts13` — h13
    8. `moe_shared` — s13  ← derived recompute boundary
    9. `moe_experts2_combine` — —
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): rmsnorm_fwd → mm → rmsnorm_fwd → mm → rope_fwd → mm → rmsnorm_fwd → rope_fwd → mm → dsa_sparse_attn_fwd → addmm → rmsnorm_fwd → mm → moe_topk_sigmoid_noaux → moe_sort → moe_dispatch_fwd → moe_grouped_mm_fwd → mm → swiglu_packed_fwd → moe_grouped_mm_fwd → swiglu_packed_fwd → mm → moe_combine_fwd

### `head_loss` — `HeadLoss`

- example task: `head_loss_0_0`
- inputs: `y_0_0_17` (268,435,456B), `targets_0_0` (262,144B), `W_head` (529,534,976B)
- outputs: `dy_0_0_17` (268,435,456B), `loss_0_0` (4B), `dW_head_0` (529,534,976B)
- mutates: —
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): rmsnorm_fwd → mm → ce_loss_fwd_bwd → mm×2 → rmsnorm_bwd

### `optimizer_head` — `AdamWStep`

- example task: `optimizer_head_0`
- inputs: `W_head` (529,534,976B), `dW_head_0` (529,534,976B), `O_head` (1,059,069,952B)
- outputs: —
- mutates: `W_head`, `O_head`
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): adamw_step×2

### `gmf_bwd` — `Glm52MfBlockBwd`

- example task: `block_bwd_0_0_17`
- inputs: `dy_0_0_17` (268,435,456B), `A_0_0_17` (2,945,449,984B), `y_0_0_16` (268,435,456B), `W_17` (1,633,822,720B), `M_0_0_17` (5,243,648B), `M_0_0_14` (273,679,104B)
- outputs: `dy_0_0_16` (268,435,456B), `dW_0_17` (1,633,822,720B), `dM_0_0_14` (268,435,456B)
- mutates: —
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): rmsnorm_apply → moe_dispatch_fwd×2 → swiglu_packed_fwd → moe_grouped_mm_dgrad → moe_rowdot → moe_scale_rows → moe_grouped_mm_wgrad → moe_scale_rows → swiglu_packed_bwd → moe_grouped_mm_wgrad → moe_grouped_mm_dgrad → moe_dispatch_bwd → moe_router_bwd_sigmoid → moe_seq_aux_grad → mm → swiglu_packed_fwd → mm×2 → swiglu_packed_bwd → mm → rmsnorm_bwd → mm×2 → rmsnorm_apply → mm → rope_fwd → rmsnorm_apply → rope_fwd → mm → sort → scatter_add_ → dsa_sparse_attn_bwd → rmsnorm_apply → dsa_probs_sum → rope_bwd → mm×2 → rmsnorm_bwd → rope_bwd → mm×2 → rmsnorm_bwd → mm×3 → rmsnorm_bwd

### `optimizer_block` — `AdamWStep`

- example task: `optimizer_0_17`
- inputs: `W_17` (1,633,822,720B), `dW_0_17` (1,633,822,720B), `O_17` (3,267,645,440B)
- outputs: —
- mutates: `W_17`, `O_17`
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): adamw_step×14

### `gml_bwd` — `Glm52MlBlockBwd`

- example task: `block_bwd_0_0_14`
- inputs: `dy_0_0_14` (268,435,456B), `A_0_0_14` (2,945,449,984B), `y_0_0_13` (268,435,456B), `W_14` (1,634,675,200B), `M_0_0_14` (273,679,104B), `dM_0_0_14` (268,435,456B)
- outputs: `dy_0_0_13` (268,435,456B), `dW_0_14` (1,634,675,200B)
- mutates: —
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): rmsnorm_apply → moe_dispatch_fwd×2 → swiglu_packed_fwd → moe_grouped_mm_dgrad → moe_rowdot → moe_scale_rows → moe_grouped_mm_wgrad → moe_scale_rows → swiglu_packed_bwd → moe_grouped_mm_wgrad → moe_grouped_mm_dgrad → moe_dispatch_bwd → moe_router_bwd_sigmoid → moe_seq_aux_grad → mm → swiglu_packed_fwd → mm×2 → swiglu_packed_bwd → mm → rmsnorm_bwd → mm×2 → rmsnorm_apply → mm → rope_fwd → rmsnorm_apply → rope_fwd → mm → sort → scatter_add_ → dsa_sparse_attn_bwd → rmsnorm_apply → mm → rope_fwd → mm → rope_fwd → mm → dsa_index_scores → dsa_probs_sum → scatter_add_ → logsumexp → dsa_index_bwd → rope_bwd → mm → rope_bwd → mm×3 → rope_bwd → mm×2 → rmsnorm_bwd → rope_bwd → mm×2 → rmsnorm_bwd → mm×3 → rmsnorm_bwd

### `gdl_bwd` — `Glm52DlBlockBwd`

- example task: `block_bwd_0_0_1`
- inputs: `dy_0_0_1` (268,435,456B), `A_0_0_1` (2,660,237,312B), `y_0_0_0` (268,435,456B), `W_1` (111,618,048B), `M_0_0_1` (268,435,456B)
- outputs: `dy_0_0_0` (268,435,456B), `dW_0_1` (111,618,048B)
- mutates: —
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): rmsnorm_apply → swiglu_fwd_out → mm×2 → swiglu_bwd → mm×3 → rmsnorm_bwd → mm×2 → rmsnorm_apply → mm → rope_fwd → rmsnorm_apply → rope_fwd → mm → sort → scatter_add_ → dsa_sparse_attn_bwd → rmsnorm_apply → mm → rope_fwd → mm → rope_fwd → mm → dsa_index_scores → _softmax → dsa_probs_sum → dsa_index_bwd → rope_bwd → mm → rope_bwd → mm×3 → rope_bwd → mm×2 → rmsnorm_bwd → rope_bwd → mm×2 → rmsnorm_bwd → mm×3 → rmsnorm_bwd

### `embed_bwd` — `EmbedBwd`

- example task: `embed_bwd_0_0`
- inputs: `dy_embed_0_0` (268,435,456B), `tokens_0_0` (262,144B)
- outputs: `dW_embed_0` (529,530,880B)
- mutates: —
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): embed_bwd_accum

### `optimizer_embed` — `AdamWStep`

- example task: `optimizer_embed_0`
- inputs: `W_embed` (529,530,880B), `dW_embed_0` (529,530,880B), `O_embed` (1,059,061,760B)
- outputs: —
- mutates: `W_embed`, `O_embed`
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): adamw_step

