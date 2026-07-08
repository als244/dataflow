# qwen35moe / `tiny` @ 16x4K: tasks, objects, kernels

GENERATED from `ShapedQwen35MoeConfig.tiny()` at run shape microbatch 16 × seq 4096 — regenerate with `python tools/gen_model_page.py --preset tiny --microbatch 16 --seq-len 4096`. All presets: [builtin_models.md](../../builtin_models.md); task-kind fleet index: [task_kinds.md](../../task_kinds.md).

Layer kinds (4 layers): `lin lin lin full`

**Run shape**: microbatch 16 × seq_len 4096 = **65,536 tokens per round** (× 1 grad-accum round(s) per step). `A_*`/`M_*` objects are sized per round; bytes/token figures transfer to any run shape.

## Object summary

At this run shape (65,536 tokens/round). Token-scaled objects show bytes/token in parens. Details per kind below.

| object | scope | bytes |
|---|---|---|
| `W_i (lin)` | layer | 2,044,160 |
| `dW_i (lin)` | layer/step | 2,044,160 |
| `O_i (lin)` | layer | 4,088,320 |
| `A (lin)` | layer × round | 239,730,688 (3,658.0/token) |
| `W_i (full)` | layer | 2,299,904 |
| `dW_i (full)` | layer/step | 2,299,904 |
| `O_i (full)` | layer | 4,599,808 |
| `A (full)` | layer × round | 272,760,832 (4,162.0/token) |
| `W_head` | run | 262,656 |
| `W_embed` | run | 262,144 |
| `O_embed` | run | 524,288 |
| `O_head` | run | 525,312 |
| `hidden state (y)` | boundary buffer | 33,554,432 (512.0/token) |

### Aggregate totals (all layers, this run shape)

| type | objects | total bytes |
|---|---|---|
| W (all weights, incl. embed/head) | 6 | 8,957,184 |
| dW (all gradients, incl. metadata grads, per step) | 6 | 8,957,184 |
| O (all optimizer state) | 6 | 17,914,368 |
| A (all saved contexts, one round) | 4 | 991,952,896 (15,136.0/token) |
| M (all metadata, one round) | 4 | 5,243,904 (80.0/token) |

## Dims

| field | value |
|---|---|
| `d_model` | 256 |
| `n_layers` | 4 |
| `full_attention_interval` | 4 |
| `n_heads` | 4 |
| `n_kv_heads` | 2 |
| `head_dim` | 64 |
| `partial_rotary_factor` | 0.25 |
| `lin_k_heads` | 2 |
| `lin_v_heads` | 4 |
| `lin_k_head_dim` | 32 |
| `lin_v_head_dim` | 32 |
| `lin_conv_kernel` | 4 |
| `d_ff` | 128 |
| `vocab_size` | 512 |
| `tokens` | 65536 |
| `seq_len` | 4096 |
| `rope_base` | 10000000.0 |
| `opt_policy` | adamw |

## Objects, per layer kind

`dW_i` mirrors `W_i`'s fields at the grad dtypes; `O_i` holds the optimizer policy's state slots per field (adamw default: `m_f`+`v_f` at the opt dtype; sgd fields contribute none — see extending.md §6). `A_i`/`M_i` exist per (step, round).

### kind `lin` (e.g. layer 0)

**`W_0` weights** — 2,044,160 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (256,) | 512 |
| `w_qkvz` | bf16 | (256, 384) | 196,608 |
| `w_ba` | bf16 | (256, 8) | 4,096 |
| `w_conv` | bf16 | (256, 4) | 2,048 |
| `A_log` | bf16 | (4,) | 8 |
| `dt_bias` | bf16 | (4,) | 8 |
| `lin_norm_w` | bf16 | (32,) | 64 |
| `w_out` | bf16 | (128, 256) | 65,536 |
| `ffn_norm_w` | bf16 | (256,) | 512 |
| `w_router` | bf16 | (256, 8) | 4,096 |
| `w13_experts` | bf16 | (8, 256, 256) | 1,048,576 |
| `w2_experts` | bf16 | (8, 128, 256) | 524,288 |
| `w_shared_gate` | bf16 | (256, 1) | 512 |
| `w_s13` | bf16 | (256, 256) | 131,072 |
| `w_s2` | bf16 | (128, 256) | 65,536 |

**`A_.._0` saved context** — 239,730,688 bytes = **3,658.0 bytes/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 262,144 |
| `qkvz` | bf16 | (65536, 384) | 50,331,648 |
| `ba` | bf16 | (65536, 8) | 1,048,576 |
| `g_post` | fp32 | (65536, 4) | 1,048,576 |
| `A_int` | bf16 | (65536, 4, 64) | 33,554,432 |
| `core_out` | bf16 | (65536, 4, 32) | 16,777,216 |
| `rstd_gate` | fp32 | (262144,) | 1,048,576 |
| `xo` | bf16 | (65536, 256) | 33,554,432 |
| `rstd_ffn` | fp32 | (65536,) | 262,144 |
| `router_logits` | bf16 | (65536, 8) | 1,048,576 |
| `h13` | bf16 | (131072, 256) | 67,108,864 |
| `gate_pre` | bf16 | (65536, 1) | 131,072 |
| `s13` | bf16 | (65536, 256) | 33,554,432 |

### kind `full` (e.g. layer 3)

**`W_3` weights** — 2,299,904 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (256,) | 512 |
| `wq` | bf16 | (256, 512) | 262,144 |
| `wk` | bf16 | (256, 128) | 65,536 |
| `wv` | bf16 | (256, 128) | 65,536 |
| `q_norm_w` | bf16 | (64,) | 128 |
| `k_norm_w` | bf16 | (64,) | 128 |
| `wo` | bf16 | (256, 256) | 131,072 |
| `ffn_norm_w` | bf16 | (256,) | 512 |
| `w_router` | bf16 | (256, 8) | 4,096 |
| `w13_experts` | bf16 | (8, 256, 256) | 1,048,576 |
| `w2_experts` | bf16 | (8, 128, 256) | 524,288 |
| `w_shared_gate` | bf16 | (256, 1) | 512 |
| `w_s13` | bf16 | (256, 256) | 131,072 |
| `w_s2` | bf16 | (128, 256) | 65,536 |

**`A_.._3` saved context** — 272,760,832 bytes = **4,162.0 bytes/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 262,144 |
| `qm` | bf16 | (65536, 256) | 33,554,432 |
| `km` | bf16 | (65536, 128) | 16,777,216 |
| `rstd_q` | fp32 | (262144,) | 1,048,576 |
| `rstd_k` | fp32 | (131072,) | 524,288 |
| `gate` | bf16 | (65536, 256) | 33,554,432 |
| `v` | bf16 | (65536, 128) | 16,777,216 |
| `lse` | fp32 | (64, 4096) | 1,048,576 |
| `attn_out` | bf16 | (65536, 256) | 33,554,432 |
| `xo` | bf16 | (65536, 256) | 33,554,432 |
| `rstd_ffn` | fp32 | (65536,) | 262,144 |
| `router_logits` | bf16 | (65536, 8) | 1,048,576 |
| `h13` | bf16 | (131072, 256) | 67,108,864 |
| `gate_pre` | bf16 | (65536, 1) | 131,072 |
| `s13` | bf16 | (65536, 256) | 33,554,432 |

**`W_head`** — 262,656 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `w` | bf16 | (512, 256) | 262,144 |
| `final_norm_w` | bf16 | (256,) | 512 |

## Tasks

### `embed_fwd` — `EmbedFwd`

- example task: `embed_fwd_0_0`
- inputs: `tokens_0_0` (262,144B), `W_embed` (262,144B)
- outputs: `y_embed_0_0` (33,554,432B)
- mutates: —

### `linmoe_fwd` — `Qwen35MoeLinBlockFwd`

- example task: `block_fwd_0_0_0`
- inputs: `y_embed_0_0` (33,554,432B), `W_0` (2,044,160B)
- outputs: `y_0_0_0` (33,554,432B), `A_0_0_0` (239,730,688B), `M_0_0_0` (1,310,976B)
- mutates: —
- stages (name — emitted ctx fields):
    0. `attn_norm` — rstd_attn
    1. `proj` — qkvz, ba
    2. `conv` — —
    3. `heads_l2norm` — —
    4. `fla` — g_post, A_int, core_out
    5. `norm_out` — rstd_gate, xo
    6. `ffn_norm` — rstd_ffn
    7. `moe_route` — router_logits
    8. `moe_dispatch` — —
    9. `moe_experts13` — h13
    10. `moe_shared` — s13, gate_pre  ← derived recompute boundary
    11. `moe_experts2_combine` — —
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): rmsnorm_fwd → causal_conv1d_silu_fwd → gated_rmsnorm_fwd → rmsnorm_fwd → moe_topk_softmax → moe_sort → moe_dispatch_fwd → moe_grouped_mm_fwd → swiglu_packed_fwd → moe_grouped_mm_fwd → swiglu_packed_fwd → moe_scale_rows → moe_combine_fwd

### `gattnmoe_fwd` — `Qwen35MoeAttnBlockFwd`

- example task: `block_fwd_0_0_3`
- inputs: `y_0_0_2` (33,554,432B), `W_3` (2,299,904B)
- outputs: `y_0_0_3` (33,554,432B), `A_0_0_3` (272,760,832B), `M_0_0_3` (1,310,976B)
- mutates: —
- stages (name — emitted ctx fields):
    0. `attn_norm` — rstd_attn
    1. `qkv_gate` — qm, km, gate, v
    2. `qknorm_rope` — rstd_q, rstd_k
    3. `attn` — lse, attn_out
    4. `gate_o` — xo
    5. `ffn_norm` — rstd_ffn
    6. `moe_route` — router_logits
    7. `moe_dispatch` — —
    8. `moe_experts13` — h13
    9. `moe_shared` — s13, gate_pre  ← derived recompute boundary
    10. `moe_experts2_combine` — —
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): rmsnorm_fwd×3 → rope_fwd×2 → rmsnorm_fwd → moe_topk_softmax → moe_sort → moe_dispatch_fwd → moe_grouped_mm_fwd → swiglu_packed_fwd → moe_grouped_mm_fwd → swiglu_packed_fwd → moe_scale_rows → moe_combine_fwd

### `head_loss` — `HeadLoss`

- example task: `head_loss_0_0`
- inputs: `y_0_0_3` (33,554,432B), `targets_0_0` (262,144B), `W_head` (262,656B)
- outputs: `dy_0_0_3` (33,554,432B), `loss_0_0` (4B), `dW_head_0` (262,656B)
- mutates: —
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): rmsnorm_fwd → ce_loss_fwd_bwd → rmsnorm_bwd

### `optimizer_head` — `AdamWStep`

- example task: `optimizer_head_0`
- inputs: `W_head` (262,656B), `dW_head_0` (262,656B), `O_head` (525,312B)
- outputs: —
- mutates: `W_head`, `O_head`
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): adamw_step×2

### `gattnmoe_bwd` — `Qwen35MoeAttnBlockBwd`

- example task: `block_bwd_0_0_3`
- inputs: `dy_0_0_3` (33,554,432B), `A_0_0_3` (272,760,832B), `y_0_0_2` (33,554,432B), `W_3` (2,299,904B), `M_0_0_3` (1,310,976B)
- outputs: `dy_0_0_2` (33,554,432B), `dW_0_3` (2,299,904B)
- mutates: —
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): rmsnorm_apply → moe_dispatch_fwd×2 → swiglu_packed_fwd → moe_grouped_mm_dgrad → moe_rowdot → moe_scale_rows → moe_grouped_mm_wgrad → moe_scale_rows → swiglu_packed_bwd → moe_grouped_mm_wgrad → moe_grouped_mm_dgrad → moe_dispatch_bwd → moe_router_bwd → moe_aux_lb_grad → swiglu_packed_fwd → moe_rowdot → moe_scale_rows → swiglu_packed_bwd → rmsnorm_bwd → rmsnorm_apply×2 → rope_fwd×2 → rope_bwd×2 → rmsnorm_bwd×2 → rmsnorm_apply → rmsnorm_bwd

### `optimizer_block` — `AdamWStep`

- example task: `optimizer_0_3`
- inputs: `W_3` (2,299,904B), `dW_0_3` (2,299,904B), `O_3` (4,599,808B)
- outputs: —
- mutates: `W_3`, `O_3`
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): adamw_step×14

### `linmoe_bwd` — `Qwen35MoeLinBlockBwd`

- example task: `block_bwd_0_0_2`
- inputs: `dy_0_0_2` (33,554,432B), `A_0_0_2` (239,730,688B), `y_0_0_1` (33,554,432B), `W_2` (2,044,160B), `M_0_0_2` (1,310,976B)
- outputs: `dy_0_0_1` (33,554,432B), `dW_0_2` (2,044,160B)
- mutates: —
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): rmsnorm_apply → moe_dispatch_fwd×2 → swiglu_packed_fwd → moe_grouped_mm_dgrad → moe_rowdot → moe_scale_rows → moe_grouped_mm_wgrad → moe_scale_rows → swiglu_packed_bwd → moe_grouped_mm_wgrad → moe_grouped_mm_dgrad → moe_dispatch_bwd → moe_router_bwd → moe_aux_lb_grad → swiglu_packed_fwd → moe_rowdot → moe_scale_rows → swiglu_packed_bwd → rmsnorm_bwd → gated_rmsnorm_bwd → causal_conv1d_silu_fwd → causal_conv1d_silu_bwd → rmsnorm_apply → rmsnorm_bwd

### `embed_bwd` — `EmbedBwd`

- example task: `embed_bwd_0_0`
- inputs: `dy_embed_0_0` (33,554,432B), `tokens_0_0` (262,144B)
- outputs: `dW_embed_0` (262,144B)
- mutates: —
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): embed_bwd_accum

### `optimizer_embed` — `AdamWStep`

- example task: `optimizer_embed_0`
- inputs: `W_embed` (262,144B), `dW_embed_0` (262,144B), `O_embed` (524,288B)
- outputs: —
- mutates: `W_embed`, `O_embed`
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): adamw_step

