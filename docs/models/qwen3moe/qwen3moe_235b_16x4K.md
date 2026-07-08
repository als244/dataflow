# qwen3moe / `qwen3moe_235b` @ 16x4K: tasks, objects, kernels

GENERATED from `ShapedQwen3MoeConfig.qwen3moe_235b()` at run shape microbatch 16 × seq 4096 — regenerate with `python tools/gen_model_page.py --preset qwen3moe_235b --microbatch 16 --seq-len 4096`. All presets: [builtin_models.md](../../builtin_models.md); task-kind fleet index: [task_kinds.md](../../task_kinds.md).

Layer kinds (94 layers): `block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block`

**Run shape**: microbatch 16 × seq_len 4096 = **65,536 tokens per round** (× 1 grad-accum round(s) per step). `A_*`/`M_*` objects are sized per round; bytes/token figures transfer to any run shape.

## Object summary

At this run shape (65,536 tokens/round). Token-scaled objects show bytes/token in parens. Details per kind below.

| object | scope | bytes |
|---|---|---|
| `W_i (block)` | layer | 4,975,510,016 |
| `dW_i (block)` | layer/step | 4,975,510,016 |
| `O_i (block)` | layer | 9,951,020,032 |
| `A (block)` | layer × round | 6,091,702,272 (92,952.0/token) |
| `M (block)` | layer × round | 5,243,648 (80.0/token) |
| `W_head` | run | 1,244,667,904 |
| `W_embed` | run | 1,244,659,712 |
| `O_embed` | run | 2,489,319,424 |
| `O_head` | run | 2,489,335,808 |
| `hidden state (y)` | boundary buffer | 536,870,912 (8,192.0/token) |

### Aggregate totals (all layers, this run shape)

| type | objects | total bytes |
|---|---|---|
| W (all weights, incl. embed/head) | 96 | 470,187,269,120 |
| dW (all gradients, incl. metadata grads, per step) | 96 | 470,187,269,120 |
| O (all optimizer state) | 96 | 940,374,538,240 |
| A (all saved contexts, one round) | 94 | 572,620,013,568 (8,737,488.0/token) |
| M (all metadata, one round) | 94 | 492,902,912 (7,521.1/token) |

## Dims

| field | value |
|---|---|
| `d_model` | 4096 |
| `n_heads` | 64 |
| `n_kv_heads` | 4 |
| `head_dim` | 128 |
| `d_ff` | 1536 |
| `vocab_size` | 151936 |
| `tokens` | 65536 |
| `seq_len` | 4096 |
| `rope_base` | 1000000.0 |
| `opt_policy` | adamw |

## Objects, per layer kind

`dW_i` mirrors `W_i`'s fields at the grad dtypes; `O_i` holds the optimizer policy's state slots per field (adamw default: `m_f`+`v_f` at the opt dtype; sgd fields contribute none — see extending.md §6). `A_i`/`M_i` exist per (step, round).

### kind `block` (e.g. layer 0)

**`W_0` weights** — 4,975,510,016 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (4096,) | 8,192 |
| `wq` | bf16 | (4096, 8192) | 67,108,864 |
| `wk` | bf16 | (4096, 512) | 4,194,304 |
| `wv` | bf16 | (4096, 512) | 4,194,304 |
| `q_norm_w` | bf16 | (128,) | 256 |
| `k_norm_w` | bf16 | (128,) | 256 |
| `wo` | bf16 | (8192, 4096) | 67,108,864 |
| `ffn_norm_w` | bf16 | (4096,) | 8,192 |
| `w_router` | bf16 | (4096, 128) | 1,048,576 |
| `w13_experts` | bf16 | (128, 4096, 3072) | 3,221,225,472 |
| `w2_experts` | bf16 | (128, 1536, 4096) | 1,610,612,736 |

**`A_.._0` saved context** — 6,091,702,272 bytes = **92,952.0 bytes/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 262,144 |
| `qm` | bf16 | (65536, 8192) | 1,073,741,824 |
| `km` | bf16 | (65536, 512) | 67,108,864 |
| `rstd_q` | fp32 | (4194304,) | 16,777,216 |
| `rstd_k` | fp32 | (262144,) | 1,048,576 |
| `v` | bf16 | (65536, 512) | 67,108,864 |
| `lse` | fp32 | (1024, 4096) | 16,777,216 |
| `attn_out` | bf16 | (65536, 8192) | 1,073,741,824 |
| `h_mid` | bf16 | (65536, 4096) | 536,870,912 |
| `rstd_ffn` | fp32 | (65536,) | 262,144 |
| `router_logits` | bf16 | (65536, 128) | 16,777,216 |
| `h13` | bf16 | (524288, 3072) | 3,221,225,472 |

**`M_.._0` metadata** — 5,243,648 bytes = **80.0 bytes/token** (never recomputed)

| field | dtype | shape | bytes |
|---|---|---|---|
| `route_w` | bf16 | (65536, 8) | 1,048,576 |
| `route_ids` | int32 | (65536, 8) | 2,097,152 |
| `route_order` | int32 | (524288,) | 2,097,152 |
| `route_offsets` | int32 | (129,) | 516 |

**`W_head`** — 1,244,667,904 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `w` | bf16 | (151936, 4096) | 1,244,659,712 |
| `final_norm_w` | bf16 | (4096,) | 8,192 |

## Tasks

### `embed_fwd` — `EmbedFwd`

- example task: `embed_fwd_0_0`
- inputs: `tokens_0_0` (262,144B), `W_embed` (1,244,659,712B)
- outputs: `y_embed_0_0` (536,870,912B)
- mutates: —

### `q3moeattn_fwd` — `Qwen3MoeBlockFwd`

- example task: `block_fwd_0_0_0`
- inputs: `y_embed_0_0` (536,870,912B), `W_0` (4,975,510,016B)
- outputs: `y_0_0_0` (536,870,912B), `A_0_0_0` (6,091,702,272B), `M_0_0_0` (5,243,648B)
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
- inputs: `y_0_0_93` (536,870,912B), `targets_0_0` (262,144B), `W_head` (1,244,667,904B)
- outputs: `dy_0_0_93` (536,870,912B), `loss_0_0` (4B), `dW_head_0` (1,244,667,904B)
- mutates: —
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): rmsnorm_fwd → ce_loss_fwd_bwd → rmsnorm_bwd

### `optimizer_head` — `AdamWStep`

- example task: `optimizer_head_0`
- inputs: `W_head` (1,244,667,904B), `dW_head_0` (1,244,667,904B), `O_head` (2,489,335,808B)
- outputs: —
- mutates: `W_head`, `O_head`
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): adamw_step×2

### `q3moeattn_bwd` — `Qwen3MoeBlockBwd`

- example task: `block_bwd_0_0_93`
- inputs: `dy_0_0_93` (536,870,912B), `A_0_0_93` (6,091,702,272B), `y_0_0_92` (536,870,912B), `W_93` (4,975,510,016B), `M_0_0_93` (5,243,648B)
- outputs: `dy_0_0_92` (536,870,912B), `dW_0_93` (4,975,510,016B)
- mutates: —
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): rmsnorm_apply → moe_dispatch_fwd×2 → swiglu_packed_fwd → moe_grouped_mm_dgrad → moe_rowdot → moe_scale_rows → moe_grouped_mm_wgrad → moe_scale_rows → swiglu_packed_bwd → moe_grouped_mm_wgrad → moe_grouped_mm_dgrad → moe_dispatch_bwd → moe_router_bwd → moe_aux_lb_grad → rmsnorm_bwd → rmsnorm_apply×2 → rope_fwd×2 → rope_bwd×2 → rmsnorm_bwd×2 → rmsnorm_apply → rmsnorm_bwd

### `optimizer_block` — `AdamWStep`

- example task: `optimizer_0_93`
- inputs: `W_93` (4,975,510,016B), `dW_0_93` (4,975,510,016B), `O_93` (9,951,020,032B)
- outputs: —
- mutates: `W_93`, `O_93`
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): adamw_step×11

### `embed_bwd` — `EmbedBwd`

- example task: `embed_bwd_0_0`
- inputs: `dy_embed_0_0` (536,870,912B), `tokens_0_0` (262,144B)
- outputs: `dW_embed_0` (1,244,659,712B)
- mutates: —
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): embed_bwd_accum

### `optimizer_embed` — `AdamWStep`

- example task: `optimizer_embed_0`
- inputs: `W_embed` (1,244,659,712B), `dW_embed_0` (1,244,659,712B), `O_embed` (2,489,319,424B)
- outputs: —
- mutates: `W_embed`, `O_embed`
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): adamw_step

