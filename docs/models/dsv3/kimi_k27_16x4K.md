# dsv3 / `kimi_k27` @ 16x4K: tasks, objects, kernels

GENERATED from `ShapedDsv3Config.kimi_k27()` at run shape microbatch 16 × seq 4096 — regenerate with `python tools/gen_model_page.py --preset kimi_k27 --microbatch 16 --seq-len 4096`. All presets: [builtin_models.md](../../builtin_models.md); task-kind fleet index: [task_kinds.md](../../task_kinds.md).

Layer kinds (61 layers): `dense moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe moe`

**Run shape**: microbatch 16 × seq_len 4096 = **65,536 tokens per round** (× 1 grad-accum round(s) per step). `A_*`/`M_*` objects are sized per round; bytes/token figures transfer to any run shape.

## Object summary

At this run shape (65,536 tokens/round). Token-scaled objects show bytes/token in parens. Details per kind below.

| object | scope | bytes |
|---|---|---|
| `W_i (dense)` | layer | 995,000,320 |
| `dW_i (dense)` | layer/step | 995,000,320 |
| `O_i (dense)` | layer | 1,990,000,640 |
| `A (dense)` | layer × round | 7,139,753,984 (108,944.0/token) |
| `W_i (moe)` | layer | 34,118,731,264 |
| `dW_i (moe)` | layer/step | 34,118,731,264 |
| `O_i (moe)` | layer | 68,237,462,528 |
| `A (moe)` | layer × round | 7,190,085,632 (109,712.0/token) |
| `M (moe)` | layer × round | 5,244,672 (80.0/token) |
| `W_head` | run | 2,348,824,576 |
| `W_embed` | run | 2,348,810,240 |
| `O_embed` | run | 4,697,620,480 |
| `O_head` | run | 4,697,649,152 |
| `hidden state (y)` | boundary buffer | 939,524,096 (14,336.0/token) |

### Aggregate totals (all layers, this run shape)

| type | objects | total bytes |
|---|---|---|
| W (all weights, incl. embed/head) | 63 | 2,052,816,510,976 |
| dW (all gradients, incl. metadata grads, per step) | 63 | 2,052,816,510,976 |
| O (all optimizer state) | 63 | 4,105,633,021,952 |
| A (all saved contexts, one round) | 61 | 438,544,891,904 (6,691,664.0/token) |
| M (all metadata, one round) | 60 | 314,680,320 (4,801.6/token) |

## Dims

| field | value |
|---|---|
| `d_model` | 7168 |
| `n_heads` | 64 |
| `q_lora_rank` | 1536 |
| `kv_lora_rank` | 512 |
| `qk_nope_dim` | 128 |
| `qk_rope_dim` | 64 |
| `v_head_dim` | 128 |
| `d_ff` | 18432 |
| `first_k_dense` | 1 |
| `vocab_size` | 163840 |
| `tokens` | 65536 |
| `seq_len` | 4096 |
| `rope_base` | 50000.0 |
| `opt_policy` | adamw |

## Objects, per layer kind

`dW_i` mirrors `W_i`'s fields at the grad dtypes; `O_i` holds the optimizer policy's state slots per field (adamw default: `m_f`+`v_f` at the opt dtype; sgd fields contribute none — see extending.md §6). `A_i`/`M_i` exist per (step, round).

### kind `dense` (e.g. layer 0)

**`W_0` weights** — 995,000,320 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (7168,) | 14,336 |
| `w_q_a` | bf16 | (7168, 1536) | 22,020,096 |
| `q_a_norm_w` | bf16 | (1536,) | 3,072 |
| `w_q_b` | bf16 | (1536, 12288) | 37,748,736 |
| `w_kv_a` | bf16 | (7168, 576) | 8,257,536 |
| `kv_a_norm_w` | bf16 | (512,) | 1,024 |
| `w_kv_b` | bf16 | (512, 16384) | 16,777,216 |
| `wo` | bf16 | (8192, 7168) | 117,440,512 |
| `ffn_norm_w` | bf16 | (7168,) | 14,336 |
| `w1` | bf16 | (7168, 18432) | 264,241,152 |
| `w3` | bf16 | (7168, 18432) | 264,241,152 |
| `w2` | bf16 | (18432, 7168) | 264,241,152 |

**`A_.._0` saved context** — 7,139,753,984 bytes = **108,944.0 bytes/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 262,144 |
| `q_a` | bf16 | (65536, 1536) | 201,326,592 |
| `rstd_qa` | fp32 | (65536,) | 262,144 |
| `kv_a` | bf16 | (65536, 576) | 75,497,472 |
| `rstd_kva` | fp32 | (65536,) | 262,144 |
| `lse` | fp32 | (1024, 4096) | 16,777,216 |
| `attn_out` | bf16 | (65536, 8192) | 1,073,741,824 |
| `h_mid` | bf16 | (65536, 7168) | 939,524,096 |
| `rstd_ffn` | fp32 | (65536,) | 262,144 |
| `x1` | bf16 | (65536, 18432) | 2,415,919,104 |
| `x3` | bf16 | (65536, 18432) | 2,415,919,104 |

### kind `moe` (e.g. layer 1)

**`W_1` weights** — 34,118,731,264 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (7168,) | 14,336 |
| `w_q_a` | bf16 | (7168, 1536) | 22,020,096 |
| `q_a_norm_w` | bf16 | (1536,) | 3,072 |
| `w_q_b` | bf16 | (1536, 12288) | 37,748,736 |
| `w_kv_a` | bf16 | (7168, 576) | 8,257,536 |
| `kv_a_norm_w` | bf16 | (512,) | 1,024 |
| `w_kv_b` | bf16 | (512, 16384) | 16,777,216 |
| `wo` | bf16 | (8192, 7168) | 117,440,512 |
| `ffn_norm_w` | bf16 | (7168,) | 14,336 |
| `w_router` | bf16 | (7168, 384) | 5,505,024 |
| `w_router_bias` | fp32 | (384,) | 1,536 |
| `w13_experts` | bf16 | (384, 7168, 4096) | 22,548,578,304 |
| `w2_experts` | bf16 | (384, 2048, 7168) | 11,274,289,152 |
| `w_s13` | bf16 | (7168, 4096) | 58,720,256 |
| `w_s2` | bf16 | (2048, 7168) | 29,360,128 |

**`A_.._1` saved context** — 7,190,085,632 bytes = **109,712.0 bytes/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 262,144 |
| `q_a` | bf16 | (65536, 1536) | 201,326,592 |
| `rstd_qa` | fp32 | (65536,) | 262,144 |
| `kv_a` | bf16 | (65536, 576) | 75,497,472 |
| `rstd_kva` | fp32 | (65536,) | 262,144 |
| `lse` | fp32 | (1024, 4096) | 16,777,216 |
| `attn_out` | bf16 | (65536, 8192) | 1,073,741,824 |
| `h_mid` | bf16 | (65536, 7168) | 939,524,096 |
| `rstd_ffn` | fp32 | (65536,) | 262,144 |
| `router_logits` | bf16 | (65536, 384) | 50,331,648 |
| `h13` | bf16 | (524288, 4096) | 4,294,967,296 |
| `s13` | bf16 | (65536, 4096) | 536,870,912 |

**`M_.._1` metadata** — 5,244,672 bytes = **80.0 bytes/token** (never recomputed)

| field | dtype | shape | bytes |
|---|---|---|---|
| `route_w` | bf16 | (65536, 8) | 1,048,576 |
| `route_ids` | int32 | (65536, 8) | 2,097,152 |
| `route_order` | int32 | (524288,) | 2,097,152 |
| `route_offsets` | int32 | (385,) | 1,540 |

**`W_head`** — 2,348,824,576 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `w` | bf16 | (163840, 7168) | 2,348,810,240 |
| `final_norm_w` | bf16 | (7168,) | 14,336 |

## Tasks

### `embed_fwd` — `EmbedFwd`

- example task: `embed_fwd_0_0`
- inputs: `tokens_0_0` (262,144B), `W_embed` (2,348,810,240B)
- outputs: `y_embed_0_0` (939,524,096B)
- mutates: —
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): index_select

### `mladense_fwd` — `Dsv3DenseBlockFwd`

- example task: `block_fwd_0_0_0`
- inputs: `y_embed_0_0` (939,524,096B), `W_0` (995,000,320B)
- outputs: `y_0_0_0` (939,524,096B), `A_0_0_0` (7,139,753,984B)
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
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): rmsnorm_fwd → mm → rmsnorm_fwd → mm → rope_fwd → mm → rmsnorm_fwd → rope_fwd → mm → _scaled_dot_product_flash_attention → addmm → rmsnorm_fwd → mm×2 → swiglu_fwd_out → addmm

### `mlamoe_fwd` — `Dsv3MoeBlockFwd`

- example task: `block_fwd_0_0_1`
- inputs: `y_0_0_0` (939,524,096B), `W_1` (34,118,731,264B)
- outputs: `y_0_0_1` (939,524,096B), `A_0_0_1` (7,190,085,632B), `M_0_0_1` (5,244,672B)
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
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): rmsnorm_fwd → mm → rmsnorm_fwd → mm → rope_fwd → mm → rmsnorm_fwd → rope_fwd → mm → _scaled_dot_product_flash_attention → addmm → rmsnorm_fwd → mm → moe_topk_sigmoid_noaux → moe_sort → moe_dispatch_fwd → moe_grouped_mm_fwd → mm → swiglu_packed_fwd → moe_grouped_mm_fwd → swiglu_packed_fwd → mm → moe_combine_fwd

### `head_loss` — `HeadLoss`

- example task: `head_loss_0_0`
- inputs: `y_0_0_60` (939,524,096B), `targets_0_0` (262,144B), `W_head` (2,348,824,576B)
- outputs: `dy_0_0_60` (939,524,096B), `loss_0_0` (4B), `dW_head_0` (2,348,824,576B)
- mutates: —
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): rmsnorm_fwd → mm → ce_loss_fwd_bwd → mm×2 → rmsnorm_bwd

### `optimizer_head` — `AdamWStep`

- example task: `optimizer_head_0`
- inputs: `W_head` (2,348,824,576B), `dW_head_0` (2,348,824,576B), `O_head` (4,697,649,152B)
- outputs: —
- mutates: `W_head`, `O_head`
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): adamw_step×2

### `mlamoe_bwd` — `Dsv3MoeBlockBwd`

- example task: `block_bwd_0_0_60`
- inputs: `dy_0_0_60` (939,524,096B), `A_0_0_60` (7,190,085,632B), `y_0_0_59` (939,524,096B), `W_60` (34,118,731,264B), `M_0_0_60` (5,244,672B)
- outputs: `dy_0_0_59` (939,524,096B), `dW_0_60` (34,118,731,264B)
- mutates: —
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): rmsnorm_apply → moe_dispatch_fwd×2 → swiglu_packed_fwd → moe_grouped_mm_dgrad → moe_rowdot → moe_scale_rows → moe_grouped_mm_wgrad → moe_scale_rows → swiglu_packed_bwd → moe_grouped_mm_wgrad → moe_grouped_mm_dgrad → moe_dispatch_bwd → moe_router_bwd_sigmoid → moe_seq_aux_grad → mm → swiglu_packed_fwd → mm×2 → swiglu_packed_bwd → mm → rmsnorm_bwd → mm×2 → rmsnorm_apply → mm → rope_fwd → rmsnorm_apply → rope_fwd → mm → _scaled_dot_product_flash_attention_backward → rope_bwd → mm×2 → rmsnorm_bwd → rope_bwd → mm×2 → rmsnorm_bwd → rmsnorm_apply → mm×3 → rmsnorm_bwd

### `optimizer_block` — `AdamWStep`

- example task: `optimizer_0_60`
- inputs: `W_60` (34,118,731,264B), `dW_0_60` (34,118,731,264B), `O_60` (68,237,462,528B)
- outputs: —
- mutates: `W_60`, `O_60`
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): adamw_step×14

### `mladense_bwd` — `Dsv3DenseBlockBwd`

- example task: `block_bwd_0_0_0`
- inputs: `dy_0_0_0` (939,524,096B), `A_0_0_0` (7,139,753,984B), `y_embed_0_0` (939,524,096B), `W_0` (995,000,320B)
- outputs: `dy_embed_0_0` (939,524,096B), `dW_0_0` (995,000,320B)
- mutates: —
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): rmsnorm_apply → swiglu_fwd_out → mm×2 → swiglu_bwd → mm×3 → rmsnorm_bwd → mm×2 → rmsnorm_apply → mm → rope_fwd → rmsnorm_apply → rope_fwd → mm → _scaled_dot_product_flash_attention_backward → rope_bwd → mm×2 → rmsnorm_bwd → rope_bwd → mm×2 → rmsnorm_bwd → rmsnorm_apply → mm×3 → rmsnorm_bwd

### `embed_bwd` — `EmbedBwd`

- example task: `embed_bwd_0_0`
- inputs: `dy_embed_0_0` (939,524,096B), `tokens_0_0` (262,144B)
- outputs: `dW_embed_0` (2,348,810,240B)
- mutates: —
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): embed_bwd_accum

### `optimizer_embed` — `AdamWStep`

- example task: `optimizer_embed_0`
- inputs: `W_embed` (2,348,810,240B), `dW_embed_0` (2,348,810,240B), `O_embed` (4,697,620,480B)
- outputs: —
- mutates: `W_embed`, `O_embed`
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): adamw_step

