# qwen35moe: tasks, objects, kernels

GENERATED from `ShapedQwen35MoeConfig.tiny()` — regenerate with `python tools/gen_model_docs.py --family qwen35moe`. Presets: [builtin_models.md](../builtin_models.md); task-kind fleet index: [task_kinds.md](../task_kinds.md).

Layer kinds (4 layers): `lin lin lin full`

## Dims (documentation preset)

| field | value |
|---|---|
| `d_model` | 256 |
| `n_layers` | 4 |
| `full_attention_interval` | 4 |
| `n_heads` | 4 |
| `n_kv_heads` | 2 |
| `head_dim` | 64 |
| `partial_rotary_factor` | 0.25 |
| `num_k_heads` | 2 |
| `num_v_heads` | 4 |
| `head_k_dim` | 32 |
| `head_v_dim` | 32 |
| `conv_kernel` | 4 |
| `d_ff` | 128 |
| `vocab_size` | 512 |
| `tokens` | 128 |
| `seq_len` | 128 |
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

**`A_.._0` saved context** — 468,224 bytes (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (128,) | 512 |
| `qkvz` | bf16 | (128, 384) | 98,304 |
| `ba` | bf16 | (128, 8) | 2,048 |
| `g_post` | fp32 | (128, 4) | 2,048 |
| `A_int` | bf16 | (128, 4, 64) | 65,536 |
| `core_out` | bf16 | (128, 4, 32) | 32,768 |
| `rstd_gate` | fp32 | (512,) | 2,048 |
| `xo` | bf16 | (128, 256) | 65,536 |
| `rstd_ffn` | fp32 | (128,) | 512 |
| `router_logits` | bf16 | (128, 8) | 2,048 |
| `h13` | bf16 | (256, 256) | 131,072 |
| `gate_pre` | bf16 | (128, 1) | 256 |
| `s13` | bf16 | (128, 256) | 65,536 |

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

**`A_.._3` saved context** — 532,736 bytes (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (128,) | 512 |
| `qm` | bf16 | (128, 256) | 65,536 |
| `km` | bf16 | (128, 128) | 32,768 |
| `rstd_q` | fp32 | (512,) | 2,048 |
| `rstd_k` | fp32 | (256,) | 1,024 |
| `gate` | bf16 | (128, 256) | 65,536 |
| `v` | bf16 | (128, 128) | 32,768 |
| `lse` | fp32 | (4, 128) | 2,048 |
| `attn_out` | bf16 | (128, 256) | 65,536 |
| `xo` | bf16 | (128, 256) | 65,536 |
| `rstd_ffn` | fp32 | (128,) | 512 |
| `router_logits` | bf16 | (128, 8) | 2,048 |
| `h13` | bf16 | (256, 256) | 131,072 |
| `gate_pre` | bf16 | (128, 1) | 256 |
| `s13` | bf16 | (128, 256) | 65,536 |

**`W_head`** — 262,656 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `w` | bf16 | (512, 256) | 262,144 |
| `final_norm_w` | bf16 | (256,) | 512 |

## Tasks

### `embed_fwd` — `EmbedFwd`

- example task: `embed_fwd_0_0`
- inputs: `tokens_0_0` (512B), `W_embed` (262,144B)
- outputs: `y_embed_0_0` (65,536B)
- mutates: —

### `linmoe_fwd` — `Qwen35MoeLinBlockFwd`

- example task: `block_fwd_0_0_0`
- inputs: `y_embed_0_0` (65,536B), `W_0` (2,044,160B)
- outputs: `y_0_0_0` (65,536B), `A_0_0_0` (468,224B), `M_0_0_0` (2,816B)
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
- kernel calls (measured, one launch): rmsnorm_fwd → causal_conv1d_silu_fwd → gated_rmsnorm_fwd → rmsnorm_fwd → moe_topk_softmax → moe_sort → moe_dispatch_fwd → moe_grouped_mm_fwd → swiglu_packed_fwd → moe_grouped_mm_fwd → swiglu_packed_fwd → moe_scale_rows → moe_combine_fwd

### `gattnmoe_fwd` — `Qwen35MoeAttnBlockFwd`

- example task: `block_fwd_0_0_3`
- inputs: `y_0_0_2` (65,536B), `W_3` (2,299,904B)
- outputs: `y_0_0_3` (65,536B), `A_0_0_3` (532,736B), `M_0_0_3` (2,816B)
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
- kernel calls (measured, one launch): rmsnorm_fwd×3 → rope_fwd×2 → rmsnorm_fwd → moe_topk_softmax → moe_sort → moe_dispatch_fwd → moe_grouped_mm_fwd → swiglu_packed_fwd → moe_grouped_mm_fwd → swiglu_packed_fwd → moe_scale_rows → moe_combine_fwd

### `head_loss` — `HeadLoss`

- example task: `head_loss_0_0`
- inputs: `y_0_0_3` (65,536B), `targets_0_0` (512B), `W_head` (262,656B)
- outputs: `dy_0_0_3` (65,536B), `loss_0_0` (4B), `dW_head_0` (262,656B)
- mutates: —
- kernel calls (measured, one launch): rmsnorm_fwd → ce_loss_fwd_bwd → rmsnorm_bwd

### `optimizer_head` — `AdamWStep`

- example task: `optimizer_head_0`
- inputs: `W_head` (262,656B), `dW_head_0` (262,656B), `O_head` (525,312B)
- outputs: —
- mutates: `W_head`, `O_head`
- kernel calls (measured, one launch): adamw_step×2

### `gattnmoe_bwd` — `Qwen35MoeAttnBlockBwd`

- example task: `block_bwd_0_0_3`
- inputs: `dy_0_0_3` (65,536B), `A_0_0_3` (532,736B), `y_0_0_2` (65,536B), `W_3` (2,299,904B), `M_0_0_3` (2,816B)
- outputs: `dy_0_0_2` (65,536B), `dW_0_3` (2,299,904B)
- mutates: —
- kernel calls (measured, one launch): rmsnorm_apply → moe_dispatch_fwd×2 → swiglu_packed_fwd → moe_grouped_mm_dgrad → moe_rowdot → moe_scale_rows → moe_grouped_mm_wgrad → moe_scale_rows → swiglu_packed_bwd → moe_grouped_mm_wgrad → moe_grouped_mm_dgrad → moe_dispatch_bwd → moe_router_bwd → moe_aux_lb_grad → swiglu_packed_fwd → moe_rowdot → moe_scale_rows → swiglu_packed_bwd → rmsnorm_bwd → rmsnorm_apply×2 → rope_fwd×2 → rope_bwd×2 → rmsnorm_bwd×2 → rmsnorm_apply → rmsnorm_bwd

### `optimizer_block` — `AdamWStep`

- example task: `optimizer_0_3`
- inputs: `W_3` (2,299,904B), `dW_0_3` (2,299,904B), `O_3` (4,599,808B)
- outputs: —
- mutates: `W_3`, `O_3`
- kernel calls (measured, one launch): adamw_step×14

### `linmoe_bwd` — `Qwen35MoeLinBlockBwd`

- example task: `block_bwd_0_0_2`
- inputs: `dy_0_0_2` (65,536B), `A_0_0_2` (468,224B), `y_0_0_1` (65,536B), `W_2` (2,044,160B), `M_0_0_2` (2,816B)
- outputs: `dy_0_0_1` (65,536B), `dW_0_2` (2,044,160B)
- mutates: —
- kernel calls (measured, one launch): rmsnorm_apply → moe_dispatch_fwd×2 → swiglu_packed_fwd → moe_grouped_mm_dgrad → moe_rowdot → moe_scale_rows → moe_grouped_mm_wgrad → moe_scale_rows → swiglu_packed_bwd → moe_grouped_mm_wgrad → moe_grouped_mm_dgrad → moe_dispatch_bwd → moe_router_bwd → moe_aux_lb_grad → swiglu_packed_fwd → moe_rowdot → moe_scale_rows → swiglu_packed_bwd → rmsnorm_bwd → gated_rmsnorm_bwd → causal_conv1d_silu_fwd → causal_conv1d_silu_bwd → rmsnorm_apply → rmsnorm_bwd

### `embed_bwd` — `EmbedBwd`

- example task: `embed_bwd_0_0`
- inputs: `dy_embed_0_0` (65,536B), `tokens_0_0` (512B)
- outputs: `dW_embed_0` (262,144B)
- mutates: —
- kernel calls (measured, one launch): embed_bwd_accum

### `optimizer_embed` — `AdamWStep`

- example task: `optimizer_embed_0`
- inputs: `W_embed` (262,144B), `dW_embed_0` (262,144B), `O_embed` (524,288B)
- outputs: —
- mutates: `W_embed`, `O_embed`
- kernel calls (measured, one launch): adamw_step

