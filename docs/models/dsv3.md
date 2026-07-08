# dsv3: tasks, objects, kernels

GENERATED from `ShapedDsv3Config.dsv3_mini()` — regenerate with `python tools/gen_model_docs.py --family dsv3`. Presets: [builtin_models.md](../builtin_models.md); task-kind fleet index: [task_kinds.md](../task_kinds.md).

Layer kinds (18 layers): `dense dense moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe`

**Run shape of this documentation preset**: microbatch 1 × seq_len 4096 = **4,096 tokens per round** (× 1 grad-accum round(s) per step). `A_*`/`M_*` objects are sized per round; their bytes/token figures below transfer to any run shape.

## Dims (documentation preset)

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
| `tokens` | 4096 |
| `seq_len` | 4096 |
| `rope_base` | 10000.0 |
| `opt_policy` | adamw |

## Objects, per layer kind

`dW_i` mirrors `W_i`'s fields at the grad dtypes; `O_i` holds the optimizer policy's state slots per field (adamw default: `m_f`+`v_f` at the opt dtype; sgd fields contribute none — see extending.md §6). `A_i`/`M_i` exist per (step, round).

### kind `dense` (e.g. layer 0)

**`W_0` weights** — 110,765,568 bytes

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
| `w1` | bf16 | (2048, 8192) | 33,554,432 |
| `w3` | bf16 | (2048, 8192) | 33,554,432 |
| `w2` | bf16 | (8192, 2048) | 33,554,432 |

**`A_.._0` saved context** — 166,264,832 bytes = **40,592.0 bytes/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (4096,) | 16,384 |
| `q_a` | bf16 | (4096, 512) | 4,194,304 |
| `rstd_qa` | fp32 | (4096,) | 16,384 |
| `kv_a` | bf16 | (4096, 288) | 2,359,296 |
| `rstd_kva` | fp32 | (4096,) | 16,384 |
| `lse` | fp32 | (16, 4096) | 262,144 |
| `attn_out` | bf16 | (4096, 1024) | 8,388,608 |
| `h_mid` | bf16 | (4096, 2048) | 16,777,216 |
| `rstd_ffn` | fp32 | (4096,) | 16,384 |
| `x1` | bf16 | (4096, 8192) | 67,108,864 |
| `x3` | bf16 | (4096, 8192) | 67,108,864 |

### kind `moe` (e.g. layer 2)

**`W_2` weights** — 1,633,822,720 bytes

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

**`A_.._2` saved context** — 184,090,624 bytes = **44,944.0 bytes/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (4096,) | 16,384 |
| `q_a` | bf16 | (4096, 512) | 4,194,304 |
| `rstd_qa` | fp32 | (4096,) | 16,384 |
| `kv_a` | bf16 | (4096, 288) | 2,359,296 |
| `rstd_kva` | fp32 | (4096,) | 16,384 |
| `lse` | fp32 | (16, 4096) | 262,144 |
| `attn_out` | bf16 | (4096, 1024) | 8,388,608 |
| `h_mid` | bf16 | (4096, 2048) | 16,777,216 |
| `rstd_ffn` | fp32 | (4096,) | 16,384 |
| `router_logits` | bf16 | (4096, 128) | 1,048,576 |
| `h13` | bf16 | (32768, 2048) | 134,217,728 |
| `s13` | bf16 | (4096, 2048) | 16,777,216 |

**`M_.._2` metadata** — 328,448 bytes = **80.2 bytes/token** (never recomputed)

| field | dtype | shape | bytes |
|---|---|---|---|
| `route_w` | bf16 | (4096, 8) | 65,536 |
| `route_ids` | int32 | (4096, 8) | 131,072 |
| `route_order` | int32 | (32768,) | 131,072 |
| `route_offsets` | int32 | (129,) | 516 |

**`W_head`** — 529,534,976 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `w` | bf16 | (129280, 2048) | 529,530,880 |
| `final_norm_w` | bf16 | (2048,) | 4,096 |

## Tasks

### `embed_fwd` — `EmbedFwd`

- example task: `embed_fwd_0_0`
- inputs: `tokens_0_0` (16,384B), `W_embed` (529,530,880B)
- outputs: `y_embed_0_0` (16,777,216B)
- mutates: —

### `mladense_fwd` — `Dsv3DenseBlockFwd`

- example task: `block_fwd_0_0_0`
- inputs: `y_embed_0_0` (16,777,216B), `W_0` (110,765,568B)
- outputs: `y_0_0_0` (16,777,216B), `A_0_0_0` (166,264,832B)
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
- kernel calls (measured, one launch): rmsnorm_fwd×2 → rope_fwd → rmsnorm_fwd → rope_fwd → rmsnorm_fwd → swiglu_fwd_out

### `mlamoe_fwd` — `Dsv3MoeBlockFwd`

- example task: `block_fwd_0_0_2`
- inputs: `y_0_0_1` (16,777,216B), `W_2` (1,633,822,720B)
- outputs: `y_0_0_2` (16,777,216B), `A_0_0_2` (184,090,624B), `M_0_0_2` (328,448B)
- mutates: —
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
- kernel calls (measured, one launch): rmsnorm_fwd×2 → rope_fwd → rmsnorm_fwd → rope_fwd → rmsnorm_fwd → moe_topk_sigmoid_noaux → moe_sort → moe_dispatch_fwd → moe_grouped_mm_fwd → swiglu_packed_fwd → moe_grouped_mm_fwd → swiglu_packed_fwd → moe_combine_fwd

### `head_loss` — `HeadLoss`

- example task: `head_loss_0_0`
- inputs: `y_0_0_17` (16,777,216B), `targets_0_0` (16,384B), `W_head` (529,534,976B)
- outputs: `dy_0_0_17` (16,777,216B), `loss_0_0` (4B), `dW_head_0` (529,534,976B)
- mutates: —
- kernel calls (measured, one launch): rmsnorm_fwd → ce_loss_fwd_bwd → rmsnorm_bwd → rmsnorm_fwd → ce_loss_fwd_bwd → rmsnorm_bwd

### `optimizer_head` — `AdamWStep`

- example task: `optimizer_head_0`
- inputs: `W_head` (529,534,976B), `dW_head_0` (529,534,976B), `O_head` (1,059,069,952B)
- outputs: —
- mutates: `W_head`, `O_head`
- kernel calls (measured, one launch): adamw_step×2

### `mlamoe_bwd` — `Dsv3MoeBlockBwd`

- example task: `block_bwd_0_0_17`
- inputs: `dy_0_0_17` (16,777,216B), `A_0_0_17` (184,090,624B), `y_0_0_16` (16,777,216B), `W_17` (1,633,822,720B), `M_0_0_17` (328,448B)
- outputs: `dy_0_0_16` (16,777,216B), `dW_0_17` (1,633,822,720B)
- mutates: —
- kernel calls (measured, one launch): rmsnorm_apply → moe_dispatch_fwd×2 → swiglu_packed_fwd → moe_grouped_mm_dgrad → moe_rowdot → moe_scale_rows → moe_grouped_mm_wgrad → moe_scale_rows → swiglu_packed_bwd → moe_grouped_mm_wgrad → moe_grouped_mm_dgrad → moe_dispatch_bwd → moe_router_bwd_sigmoid → moe_seq_aux_grad → swiglu_packed_fwd → swiglu_packed_bwd → rmsnorm_bwd → rmsnorm_apply → rope_fwd → rmsnorm_apply → rope_fwd → rope_bwd → rmsnorm_bwd → rope_bwd → rmsnorm_bwd → rmsnorm_apply → rmsnorm_bwd

### `optimizer_block` — `AdamWStep`

- example task: `optimizer_0_17`
- inputs: `W_17` (1,633,822,720B), `dW_0_17` (1,633,822,720B), `O_17` (3,267,645,440B)
- outputs: —
- mutates: `W_17`, `O_17`
- kernel calls (measured, one launch): adamw_step×14

### `mladense_bwd` — `Dsv3DenseBlockBwd`

- example task: `block_bwd_0_0_1`
- inputs: `dy_0_0_1` (16,777,216B), `A_0_0_1` (166,264,832B), `y_0_0_0` (16,777,216B), `W_1` (110,765,568B)
- outputs: `dy_0_0_0` (16,777,216B), `dW_0_1` (110,765,568B)
- mutates: —
- kernel calls (measured, one launch): rmsnorm_apply → swiglu_fwd_out → swiglu_bwd → rmsnorm_bwd → rmsnorm_apply → rope_fwd → rmsnorm_apply → rope_fwd → rope_bwd → rmsnorm_bwd → rope_bwd → rmsnorm_bwd → rmsnorm_apply → rmsnorm_bwd

### `embed_bwd` — `EmbedBwd`

- example task: `embed_bwd_0_0`
- inputs: `dy_embed_0_0` (16,777,216B), `tokens_0_0` (16,384B)
- outputs: `dW_embed_0` (529,530,880B)
- mutates: —
- kernel calls (measured, one launch): embed_bwd_accum

### `optimizer_embed` — `AdamWStep`

- example task: `optimizer_embed_0`
- inputs: `W_embed` (529,530,880B), `dW_embed_0` (529,530,880B), `O_embed` (1,059,061,760B)
- outputs: —
- mutates: `W_embed`, `O_embed`
- kernel calls (measured, one launch): adamw_step

