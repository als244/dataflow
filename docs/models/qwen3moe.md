# qwen3moe: tasks, objects, kernels

GENERATED from `ShapedQwen3MoeConfig.qwen3moe_30b()` at the standard documentation run shape (seq 4096 × microbatch 16) — regenerate with `python tools/gen_model_docs.py --family qwen3moe`. Presets: [builtin_models.md](../builtin_models.md); task-kind fleet index: [task_kinds.md](../task_kinds.md).

Layer kinds (48 layers): `block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block`

**Run shape of this documentation preset**: microbatch 16 × seq_len 4096 = **65,536 tokens per round** (× 1 grad-accum round(s) per step). `A_*`/`M_*` objects are sized per round; their bytes/token figures below transfer to any run shape.

## Dims (documentation preset)

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

**`W_0` weights** — 1,246,241,280 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (2048,) | 4,096 |
| `wq` | bf16 | (2048, 4096) | 16,777,216 |
| `wk` | bf16 | (2048, 512) | 2,097,152 |
| `wv` | bf16 | (2048, 512) | 2,097,152 |
| `q_norm_w` | bf16 | (128,) | 256 |
| `k_norm_w` | bf16 | (128,) | 256 |
| `wo` | bf16 | (4096, 2048) | 16,777,216 |
| `ffn_norm_w` | bf16 | (2048,) | 4,096 |
| `w_router` | bf16 | (2048, 128) | 524,288 |
| `w13_experts` | bf16 | (128, 2048, 1536) | 805,306,368 |
| `w2_experts` | bf16 | (128, 768, 2048) | 402,653,184 |

**`A_.._0` saved context** — 3,122,135,040 bytes = **47,640.0 bytes/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 262,144 |
| `qm` | bf16 | (65536, 4096) | 536,870,912 |
| `km` | bf16 | (65536, 512) | 67,108,864 |
| `rstd_q` | fp32 | (2097152,) | 8,388,608 |
| `rstd_k` | fp32 | (262144,) | 1,048,576 |
| `v` | bf16 | (65536, 512) | 67,108,864 |
| `lse` | fp32 | (512, 4096) | 8,388,608 |
| `attn_out` | bf16 | (65536, 4096) | 536,870,912 |
| `h_mid` | bf16 | (65536, 2048) | 268,435,456 |
| `rstd_ffn` | fp32 | (65536,) | 262,144 |
| `router_logits` | bf16 | (65536, 128) | 16,777,216 |
| `h13` | bf16 | (524288, 1536) | 1,610,612,736 |

**`M_.._0` metadata** — 5,243,648 bytes = **80.0 bytes/token** (never recomputed)

| field | dtype | shape | bytes |
|---|---|---|---|
| `route_w` | bf16 | (65536, 8) | 1,048,576 |
| `route_ids` | int32 | (65536, 8) | 2,097,152 |
| `route_order` | int32 | (524288,) | 2,097,152 |
| `route_offsets` | int32 | (129,) | 516 |

**`W_head`** — 622,333,952 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `w` | bf16 | (151936, 2048) | 622,329,856 |
| `final_norm_w` | bf16 | (2048,) | 4,096 |

## Tasks

### `embed_fwd` — `EmbedFwd`

- example task: `embed_fwd_0_0`
- inputs: `tokens_0_0` (262,144B), `W_embed` (622,329,856B)
- outputs: `y_embed_0_0` (268,435,456B)
- mutates: —

### `q3moeattn_fwd` — `Qwen3MoeBlockFwd`

- example task: `block_fwd_0_0_0`
- inputs: `y_embed_0_0` (268,435,456B), `W_0` (1,246,241,280B)
- outputs: `y_0_0_0` (268,435,456B), `A_0_0_0` (3,122,135,040B), `M_0_0_0` (5,243,648B)
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
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): rmsnorm_fwd×3 → rope_fwd×2 → rmsnorm_fwd → moe_topk_softmax → moe_sort → moe_dispatch_fwd → moe_grouped_mm_fwd → swiglu_packed_fwd → moe_grouped_mm_fwd → moe_combine_fwd

### `head_loss` — `HeadLoss`

- example task: `head_loss_0_0`
- inputs: `y_0_0_47` (268,435,456B), `targets_0_0` (262,144B), `W_head` (622,333,952B)
- outputs: `dy_0_0_47` (268,435,456B), `loss_0_0` (4B), `dW_head_0` (622,333,952B)
- mutates: —
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): rmsnorm_fwd → ce_loss_fwd_bwd → rmsnorm_bwd

### `optimizer_head` — `AdamWStep`

- example task: `optimizer_head_0`
- inputs: `W_head` (622,333,952B), `dW_head_0` (622,333,952B), `O_head` (1,244,667,904B)
- outputs: —
- mutates: `W_head`, `O_head`
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): adamw_step×2

### `q3moeattn_bwd` — `Qwen3MoeBlockBwd`

- example task: `block_bwd_0_0_47`
- inputs: `dy_0_0_47` (268,435,456B), `A_0_0_47` (3,122,135,040B), `y_0_0_46` (268,435,456B), `W_47` (1,246,241,280B), `M_0_0_47` (5,243,648B)
- outputs: `dy_0_0_46` (268,435,456B), `dW_0_47` (1,246,241,280B)
- mutates: —
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): rmsnorm_apply → moe_dispatch_fwd×2 → swiglu_packed_fwd → moe_grouped_mm_dgrad → moe_rowdot → moe_scale_rows → moe_grouped_mm_wgrad → moe_scale_rows → swiglu_packed_bwd → moe_grouped_mm_wgrad → moe_grouped_mm_dgrad → moe_dispatch_bwd → moe_router_bwd → moe_aux_lb_grad → rmsnorm_bwd → rmsnorm_apply×2 → rope_fwd×2 → rope_bwd×2 → rmsnorm_bwd×2 → rmsnorm_apply → rmsnorm_bwd

### `optimizer_block` — `AdamWStep`

- example task: `optimizer_0_47`
- inputs: `W_47` (1,246,241,280B), `dW_0_47` (1,246,241,280B), `O_47` (2,492,482,560B)
- outputs: —
- mutates: `W_47`, `O_47`
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): adamw_step×11

### `embed_bwd` — `EmbedBwd`

- example task: `embed_bwd_0_0`
- inputs: `dy_embed_0_0` (268,435,456B), `tokens_0_0` (262,144B)
- outputs: `dW_embed_0` (622,329,856B)
- mutates: —
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): embed_bwd_accum

### `optimizer_embed` — `AdamWStep`

- example task: `optimizer_embed_0`
- inputs: `W_embed` (622,329,856B), `dW_embed_0` (622,329,856B), `O_embed` (1,244,659,712B)
- outputs: —
- mutates: `W_embed`, `O_embed`
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): adamw_step

