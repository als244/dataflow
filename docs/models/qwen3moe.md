# qwen3moe: tasks, objects, kernels

GENERATED from `ShapedQwen3MoeConfig.tiny()` — regenerate with `python tools/gen_model_docs.py --family qwen3moe`. Presets: [builtin_models.md](../builtin_models.md); task-kind fleet index: [task_kinds.md](../task_kinds.md).

Layer kinds (2 layers): `block block`

**Run shape of this documentation preset**: microbatch 1 × seq_len 128 = **128 tokens per round** (× 1 grad-accum round(s) per step). `A_*`/`M_*` objects are sized per round; their bytes/token figures below transfer to any run shape.

## Dims (documentation preset)

| field | value |
|---|---|
| `d_model` | 128 |
| `n_heads` | 4 |
| `n_kv_heads` | 2 |
| `head_dim` | 32 |
| `d_ff` | 64 |
| `vocab_size` | 512 |
| `tokens` | 128 |
| `seq_len` | 128 |
| `rope_base` | 1000000.0 |
| `opt_policy` | adamw |

## Objects, per layer kind

`dW_i` mirrors `W_i`'s fields at the grad dtypes; `O_i` holds the optimizer policy's state slots per field (adamw default: `m_f`+`v_f` at the opt dtype; sgd fields contribute none — see extending.md §6). `A_i`/`M_i` exist per (step, round).

### kind `block` (e.g. layer 0)

**`W_0` weights** — 494,592 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (128,) | 256 |
| `wq` | bf16 | (128, 128) | 32,768 |
| `wk` | bf16 | (128, 64) | 16,384 |
| `wv` | bf16 | (128, 64) | 16,384 |
| `q_norm_w` | bf16 | (32,) | 64 |
| `k_norm_w` | bf16 | (32,) | 64 |
| `wo` | bf16 | (128, 128) | 32,768 |
| `ffn_norm_w` | bf16 | (128,) | 256 |
| `w_router` | bf16 | (128, 8) | 2,048 |
| `w13_experts` | bf16 | (8, 128, 128) | 262,144 |
| `w2_experts` | bf16 | (8, 64, 128) | 131,072 |

**`A_.._0` saved context** — 204,800 bytes = **1,600.0 bytes/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (128,) | 512 |
| `qm` | bf16 | (128, 128) | 32,768 |
| `km` | bf16 | (128, 64) | 16,384 |
| `rstd_q` | fp32 | (512,) | 2,048 |
| `rstd_k` | fp32 | (256,) | 1,024 |
| `v` | bf16 | (128, 64) | 16,384 |
| `lse` | fp32 | (4, 128) | 2,048 |
| `attn_out` | bf16 | (128, 128) | 32,768 |
| `h_mid` | bf16 | (128, 128) | 32,768 |
| `rstd_ffn` | fp32 | (128,) | 512 |
| `router_logits` | bf16 | (128, 8) | 2,048 |
| `h13` | bf16 | (256, 128) | 65,536 |

**`M_.._0` metadata** — 2,816 bytes = **22.0 bytes/token** (never recomputed)

| field | dtype | shape | bytes |
|---|---|---|---|
| `route_w` | bf16 | (128, 2) | 512 |
| `route_ids` | int32 | (128, 2) | 1,024 |
| `route_order` | int32 | (256,) | 1,024 |
| `route_offsets` | int32 | (9,) | 36 |

**`W_head`** — 131,328 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `w` | bf16 | (512, 128) | 131,072 |
| `final_norm_w` | bf16 | (128,) | 256 |

## Tasks

### `embed_fwd` — `EmbedFwd`

- example task: `embed_fwd_0_0`
- inputs: `tokens_0_0` (512B), `W_embed` (131,072B)
- outputs: `y_embed_0_0` (32,768B)
- mutates: —

### `q3moeattn_fwd` — `Qwen3MoeBlockFwd`

- example task: `block_fwd_0_0_0`
- inputs: `y_embed_0_0` (32,768B), `W_0` (494,592B)
- outputs: `y_0_0_0` (32,768B), `A_0_0_0` (204,800B), `M_0_0_0` (2,816B)
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
- kernel calls (measured, one launch): rmsnorm_fwd×3 → rope_fwd×2 → rmsnorm_fwd → moe_topk_softmax → moe_sort → moe_dispatch_fwd → moe_grouped_mm_fwd → swiglu_packed_fwd → moe_grouped_mm_fwd → moe_combine_fwd

### `head_loss` — `HeadLoss`

- example task: `head_loss_0_0`
- inputs: `y_0_0_1` (32,768B), `targets_0_0` (512B), `W_head` (131,328B)
- outputs: `dy_0_0_1` (32,768B), `loss_0_0` (4B), `dW_head_0` (131,328B)
- mutates: —
- kernel calls (measured, one launch): rmsnorm_fwd → ce_loss_fwd_bwd → rmsnorm_bwd

### `optimizer_head` — `AdamWStep`

- example task: `optimizer_head_0`
- inputs: `W_head` (131,328B), `dW_head_0` (131,328B), `O_head` (262,656B)
- outputs: —
- mutates: `W_head`, `O_head`
- kernel calls (measured, one launch): adamw_step×2

### `q3moeattn_bwd` — `Qwen3MoeBlockBwd`

- example task: `block_bwd_0_0_1`
- inputs: `dy_0_0_1` (32,768B), `A_0_0_1` (204,800B), `y_0_0_0` (32,768B), `W_1` (494,592B), `M_0_0_1` (2,816B)
- outputs: `dy_0_0_0` (32,768B), `dW_0_1` (494,592B)
- mutates: —
- kernel calls (measured, one launch): rmsnorm_apply → moe_dispatch_fwd×2 → swiglu_packed_fwd → moe_grouped_mm_dgrad → moe_rowdot → moe_scale_rows → moe_grouped_mm_wgrad → moe_scale_rows → swiglu_packed_bwd → moe_grouped_mm_wgrad → moe_grouped_mm_dgrad → moe_dispatch_bwd → moe_router_bwd → moe_aux_lb_grad → rmsnorm_bwd → rmsnorm_apply×2 → rope_fwd×2 → rope_bwd×2 → rmsnorm_bwd×2 → rmsnorm_apply → rmsnorm_bwd

### `optimizer_block` — `AdamWStep`

- example task: `optimizer_0_1`
- inputs: `W_1` (494,592B), `dW_0_1` (494,592B), `O_1` (989,184B)
- outputs: —
- mutates: `W_1`, `O_1`
- kernel calls (measured, one launch): adamw_step×11

### `embed_bwd` — `EmbedBwd`

- example task: `embed_bwd_0_0`
- inputs: `dy_embed_0_0` (32,768B), `tokens_0_0` (512B)
- outputs: `dW_embed_0` (131,072B)
- mutates: —
- kernel calls (measured, one launch): embed_bwd_accum

### `optimizer_embed` — `AdamWStep`

- example task: `optimizer_embed_0`
- inputs: `W_embed` (131,072B), `dW_embed_0` (131,072B), `O_embed` (262,144B)
- outputs: —
- mutates: `W_embed`, `O_embed`
- kernel calls (measured, one launch): adamw_step

