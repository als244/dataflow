# dsv3 / `tiny` @ 16x4K: tasks, objects, kernels

GENERATED from `ShapedDsv3Config.tiny()` at run shape microbatch 16 × seq 4096 — regenerate with `python tools/gen_model_page.py --preset tiny --microbatch 16 --seq-len 4096`. All presets: [builtin_models.md](../../builtin_models.md); task-kind fleet index: [task_kinds.md](../../task_kinds.md).

Layer kinds (3 layers): `dense moe moe`

**Run shape**: microbatch 16 × seq_len 4096 = **65,536 tokens per round** (× 1 grad-accum round(s) per step). `A_*`/`AuxTemp_*` objects are sized per round; bytes/token figures transfer to any run shape.

## Object summary

At this run shape (65,536 tokens/round). Token-scaled objects show per-token size in parens. Details per kind below.

| object | scope | size |
|---|---|---|
| `W_i (dense)` | layer | 255.00 KiB |
| `dW_i (dense)` | layer/step | 255.00 KiB |
| `O_i (dense)` | layer | 510.00 KiB |
| `A (dense)` | layer × round | 103.00 MiB (1.61 KiB/token) |
| `W_i (moe)` | layer | 281.25 KiB |
| `dW_i (moe)` | layer/step | 281.25 KiB |
| `O_i (moe)` | layer | 562.00 KiB |
| `A (moe)` | layer × round | 64.00 MiB (1.00 KiB/token) |
| `M (moe)` | layer × round | 1.25 MiB (20.0 B/token) |
| `W_head` | run | 128.25 KiB |
| `W_embed` | run | 128.00 KiB |
| `O_embed` | run | 256.00 KiB |
| `O_head` | run | 256.50 KiB |
| `hidden state (y)` | boundary buffer | 16.00 MiB (256.0 B/token) |

### Aggregate totals (all layers, this run shape)

| type | objects | total size |
|---|---|---|
| W (all weights, incl. embed/head) | 5 | 1.05 MiB |
| dW (all gradients, per step) | 5 | 1.05 MiB |
| O (all optimizer state) | 5 | 2.10 MiB |
| A (all saved activations, one round) | 3 | 231.00 MiB (3.61 KiB/token) |
| M (all metadata, one round) | 2 | 2.50 MiB (40.0 B/token) |

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
| `kinds` | ('dense', 'moe', 'moe') |

## Objects, per layer kind

`dW_i` mirrors `W_i`'s fields at the grad dtypes; `O_i` holds the optimizer policy's state slots per field (adamw default: `m_f`+`v_f` at the opt dtype; sgd fields contribute none — see extending.md §6). `A_i`/`M_i` exist per (step, round).

### kind `dense` (e.g. layer 0)

**`W_0` weights** — 255.00 KiB

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (128,) | 256 B |
| `w_q_a` | bf16 | (128, 64) | 16.00 KiB |
| `q_a_norm_w` | bf16 | (64,) | 128 B |
| `w_q_b` | bf16 | (64, 96) | 12.00 KiB |
| `w_kv_a` | bf16 | (128, 40) | 10.00 KiB |
| `kv_a_norm_w` | bf16 | (32,) | 64 B |
| `w_kv_b` | bf16 | (32, 128) | 8.00 KiB |
| `wo` | bf16 | (64, 128) | 16.00 KiB |
| `ffn_norm_w` | bf16 | (128,) | 256 B |
| `w1` | bf16 | (128, 256) | 64.00 KiB |
| `w3` | bf16 | (128, 256) | 64.00 KiB |
| `w2` | bf16 | (256, 128) | 64.00 KiB |

**`A_.._0` saved context** — 103.00 MiB = **1.61 KiB/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 256.00 KiB |
| `q_a` | bf16 | (65536, 64) | 8.00 MiB |
| `rstd_qa` | fp32 | (65536,) | 256.00 KiB |
| `kv_a` | bf16 | (65536, 40) | 5.00 MiB |
| `rstd_kva` | fp32 | (65536,) | 256.00 KiB |
| `lse` | fp32 | (4, 65536) | 1.00 MiB |
| `attn_out` | bf16 | (65536, 64) | 8.00 MiB |
| `h_mid` | bf16 | (65536, 128) | 16.00 MiB |
| `rstd_ffn` | fp32 | (65536,) | 256.00 KiB |
| `x1` | bf16 | (65536, 256) | 32.00 MiB |
| `x3` | bf16 | (65536, 256) | 32.00 MiB |

### kind `moe` (e.g. layer 1)

**`W_1` weights** — 281.25 KiB

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (128,) | 256 B |
| `w_q_a` | bf16 | (128, 64) | 16.00 KiB |
| `q_a_norm_w` | bf16 | (64,) | 128 B |
| `w_q_b` | bf16 | (64, 96) | 12.00 KiB |
| `w_kv_a` | bf16 | (128, 40) | 10.00 KiB |
| `kv_a_norm_w` | bf16 | (32,) | 64 B |
| `w_kv_b` | bf16 | (32, 128) | 8.00 KiB |
| `wo` | bf16 | (64, 128) | 16.00 KiB |
| `ffn_norm_w` | bf16 | (128,) | 256 B |
| `w_router` | bf16 | (128, 8) | 2.00 KiB |
| `w_router_bias` | fp32 | (8,) | 32 B |
| `w13_experts` | bf16 | (8, 128, 64) | 128.00 KiB |
| `w2_experts` | bf16 | (8, 32, 128) | 64.00 KiB |
| `w_s13` | bf16 | (128, 64) | 16.00 KiB |
| `w_s2` | bf16 | (32, 128) | 8.00 KiB |

**`A_.._1` saved context** — 64.00 MiB = **1.00 KiB/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 256.00 KiB |
| `q_a` | bf16 | (65536, 64) | 8.00 MiB |
| `rstd_qa` | fp32 | (65536,) | 256.00 KiB |
| `kv_a` | bf16 | (65536, 40) | 5.00 MiB |
| `rstd_kva` | fp32 | (65536,) | 256.00 KiB |
| `lse` | fp32 | (4, 65536) | 1.00 MiB |
| `attn_out` | bf16 | (65536, 64) | 8.00 MiB |
| `h_mid` | bf16 | (65536, 128) | 16.00 MiB |
| `rstd_ffn` | fp32 | (65536,) | 256.00 KiB |
| `router_logits` | bf16 | (65536, 8) | 1.00 MiB |
| `h13` | bf16 | (131072, 64) | 16.00 MiB |
| `s13` | bf16 | (65536, 64) | 8.00 MiB |

**`M_.._1` metadata** — 1.25 MiB = **20.0 B/token** (never recomputed)

| field | dtype | shape | bytes |
|---|---|---|---|
| `route_w` | bf16 | (65536, 2) | 256.00 KiB |
| `route_ids` | int32 | (65536, 2) | 512.00 KiB |
| `route_order` | int32 | (131072,) | 512.00 KiB |
| `route_offsets` | int32 | (9,) | 36 B |

**`W_head`** — 128.25 KiB

| field | dtype | shape | bytes |
|---|---|---|---|
| `w` | bf16 | (512, 128) | 128.00 KiB |
| `final_norm_w` | bf16 | (128,) | 256 B |

## Tasks

### `prologue_round` — `RoundPrologue`

- example task: `prologue_round_0_0`
- inputs: `Aux_1` (512 B), `Aux_2` (512 B)
- outputs: `current_round_0_0` (4 B)
- mutates: `Aux_1`, `Aux_2`

### `embed_fwd` — `EmbedFwd`

- example task: `embed_fwd_0_0`
- inputs: `tokens_0_0` (256.00 KiB), `W_embed` (128.00 KiB)
- outputs: `y_embed_0_0` (16.00 MiB)
- mutates: —
- kernel calls:
    0. `index_select`

### `mladense_fwd` — `Dsv3DenseBlockFwd`

- example task: `block_fwd_0_0_0`
- inputs: `y_embed_0_0` (16.00 MiB), `W_0` (255.00 KiB)
- outputs: `y_0_0_0` (16.00 MiB), `A_0_0_0` (103.00 MiB)
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
- kernel calls, by stage:
    - `attn_norm`:
        0. `rmsnorm_fwd`
    - `mla_q`:
        1. `mm`
        2. `rmsnorm_fwd`
        3. `mm`
        4. `rope_fwd`
    - `mla_kv`:
        5. `mm`
        6. `rmsnorm_fwd`
        7. `rope_fwd`
        8. `mm`
    - `resid1_norm2`:
        9. `addmm`
        10. `rmsnorm_fwd`
    - `up_proj`:
        11. `mm ×2`
    - `swiglu`:
        12. `swiglu_fwd_out`
    - `down_resid`:
        13. `addmm`

### `mlamoe_fwd` — `Dsv3MoeBlockFwd`

- example task: `block_fwd_0_0_1`
- inputs: `y_0_0_0` (16.00 MiB), `W_1` (281.25 KiB), `current_round_0_0` (4 B), `Aux_1` (512 B)
- outputs: `y_0_0_1` (16.00 MiB), `A_0_0_1` (64.00 MiB), `AuxTemp_0_0_1` (1.25 MiB)
- mutates: `Aux_1`
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
- kernel calls, by stage:
    - `attn_norm`:
        0. `rmsnorm_fwd`
    - `mla_q`:
        1. `mm`
        2. `rmsnorm_fwd`
        3. `mm`
        4. `rope_fwd`
    - `mla_kv`:
        5. `mm`
        6. `rmsnorm_fwd`
        7. `rope_fwd`
        8. `mm`
    - `resid1_norm2`:
        9. `addmm`
        10. `rmsnorm_fwd`
    - `moe_route`:
        11. `mm`
        12. `moe_topk_sigmoid_noaux`
        13. `scatter_add_ ×2`
    - `moe_dispatch`:
        14. `moe_sort`
        15. `moe_dispatch_fwd`
    - `moe_experts13`:
        16. `moe_grouped_mm_fwd`
    - `moe_shared`:
        17. `mm`
    - `moe_experts2_combine`:
        18. `swiglu_packed_fwd`
        19. `moe_grouped_mm_fwd`
        20. `swiglu_packed_fwd`
        21. `mm`
        22. `moe_combine_fwd`

### `head_loss` — `HeadLoss`

- example task: `head_loss_0_0`
- inputs: `y_0_0_2` (16.00 MiB), `targets_0_0` (256.00 KiB), `W_head` (128.25 KiB)
- outputs: `dy_0_0_2` (16.00 MiB), `loss_0_0` (4 B), `dW_head_0` (128.25 KiB)
- mutates: —
- kernel calls:
    0. `rmsnorm_fwd`
    1. `mm`
    2. `ce_loss_fwd_bwd`
    3. `mm ×2`
    4. `rmsnorm_bwd`

### `optimizer_head` — `AdamWStep`

- example task: `optimizer_head_0`
- inputs: `W_head` (128.25 KiB), `dW_head_0` (128.25 KiB), `O_head` (256.50 KiB)
- outputs: —
- mutates: `W_head`, `O_head`
- kernel calls:
    0. `adamw_step ×2`

### `mlamoe_bwd` — `Dsv3MoeBlockBwd`

- example task: `block_bwd_0_0_2`
- inputs: `dy_0_0_2` (16.00 MiB), `A_0_0_2` (64.00 MiB), `y_0_0_1` (16.00 MiB), `W_2` (281.25 KiB), `AuxTemp_0_0_2` (1.25 MiB), `Aux_2` (512 B)
- outputs: `dy_0_0_1` (16.00 MiB), `dW_0_2` (281.00 KiB)
- mutates: `W_2`
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
    27. `_flash_attention_backward`
    28. `rope_bwd`
    29. `mm ×2`
    30. `rmsnorm_bwd`
    31. `rope_bwd`
    32. `mm ×2`
    33. `rmsnorm_bwd`
    34. `rmsnorm_apply`
    35. `mm ×3`
    36. `rmsnorm_bwd`

### `optimizer_block` — `AdamWStep`

- example task: `optimizer_0_2`
- inputs: `W_2` (281.25 KiB), `dW_0_2` (281.00 KiB), `O_2` (562.00 KiB)
- outputs: —
- mutates: `W_2`, `O_2`
- kernel calls:
    0. `adamw_step ×14`

### `mladense_bwd` — `Dsv3DenseBlockBwd`

- example task: `block_bwd_0_0_0`
- inputs: `dy_0_0_0` (16.00 MiB), `A_0_0_0` (103.00 MiB), `y_embed_0_0` (16.00 MiB), `W_0` (255.00 KiB)
- outputs: `dy_embed_0_0` (16.00 MiB), `dW_0_0` (255.00 KiB)
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
    13. `_flash_attention_backward`
    14. `rope_bwd`
    15. `mm ×2`
    16. `rmsnorm_bwd`
    17. `rope_bwd`
    18. `mm ×2`
    19. `rmsnorm_bwd`
    20. `rmsnorm_apply`
    21. `mm ×3`
    22. `rmsnorm_bwd`

### `embed_bwd` — `EmbedBwd`

- example task: `embed_bwd_0_0`
- inputs: `dy_embed_0_0` (16.00 MiB), `tokens_0_0` (256.00 KiB)
- outputs: `dW_embed_0` (128.00 KiB)
- mutates: —
- kernel calls:
    0. `embed_bwd_accum`

### `optimizer_embed` — `AdamWStep`

- example task: `optimizer_embed_0`
- inputs: `W_embed` (128.00 KiB), `dW_embed_0` (128.00 KiB), `O_embed` (256.00 KiB)
- outputs: —
- mutates: `W_embed`, `O_embed`
- kernel calls:
    0. `adamw_step`

