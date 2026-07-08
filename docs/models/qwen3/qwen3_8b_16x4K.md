# qwen3 / `qwen3_8b` @ 16x4K: tasks, objects, kernels

GENERATED from `ShapedQwen3Config.qwen3_8b()` at run shape microbatch 16 × seq 4096 — regenerate with `python tools/gen_model_page.py --preset qwen3_8b --microbatch 16 --seq-len 4096`. All presets: [builtin_models.md](../../builtin_models.md); task-kind fleet index: [task_kinds.md](../../task_kinds.md).

Layer kinds (36 layers): `block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block`

**Run shape**: microbatch 16 × seq_len 4096 = **65,536 tokens per round** (× 1 grad-accum round(s) per step). `A_*`/`M_*` objects are sized per round; bytes/token figures transfer to any run shape.

## Object summary

At this run shape (65,536 tokens/round). Token-scaled objects show bytes/token in parens. Details per kind below.

| object | scope | bytes |
|---|---|---|
| `W_i (block)` | layer | 385,892,864 |
| `dW_i (block)` | layer/step | 385,892,864 |
| `O_i (block)` | layer | 771,785,728 |
| `A (block)` | layer × round | 5,119,672,320 (78,120.0/token) |
| `W_head` | run | 1,244,667,904 |
| `W_embed` | run | 1,244,659,712 |
| `O_embed` | run | 2,489,319,424 |
| `O_head` | run | 2,489,335,808 |
| `hidden state (y)` | boundary buffer | 536,870,912 (8,192.0/token) |

### Aggregate totals (all layers, this run shape)

| type | objects | total bytes |
|---|---|---|
| W (all weights, incl. embed/head) | 38 | 16,381,470,720 |
| dW (all gradients, incl. metadata grads, per step) | 38 | 16,381,470,720 |
| O (all optimizer state) | 38 | 32,762,941,440 |
| A (all saved contexts, one round) | 36 | 184,308,203,520 (2,812,320.0/token) |

## Dims

| field | value |
|---|---|
| `d_model` | 4096 |
| `n_heads` | 32 |
| `n_kv_heads` | 8 |
| `head_dim` | 128 |
| `d_ff` | 12288 |
| `vocab_size` | 151936 |
| `tokens` | 65536 |
| `seq_len` | 4096 |
| `rope_base` | 1000000.0 |
| `opt_policy` | adamw |

## Objects, per layer kind

`dW_i` mirrors `W_i`'s fields at the grad dtypes; `O_i` holds the optimizer policy's state slots per field (adamw default: `m_f`+`v_f` at the opt dtype; sgd fields contribute none — see extending.md §6). `A_i`/`M_i` exist per (step, round).

### kind `block` (e.g. layer 0)

**`W_0` weights** — 385,892,864 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (4096,) | 8,192 |
| `wq` | bf16 | (4096, 4096) | 33,554,432 |
| `wk` | bf16 | (4096, 1024) | 8,388,608 |
| `wv` | bf16 | (4096, 1024) | 8,388,608 |
| `q_norm_w` | bf16 | (128,) | 256 |
| `k_norm_w` | bf16 | (128,) | 256 |
| `wo` | bf16 | (4096, 4096) | 33,554,432 |
| `ffn_norm_w` | bf16 | (4096,) | 8,192 |
| `w1` | bf16 | (4096, 12288) | 100,663,296 |
| `w3` | bf16 | (4096, 12288) | 100,663,296 |
| `w2` | bf16 | (12288, 4096) | 100,663,296 |

**`A_.._0` saved context** — 5,119,672,320 bytes = **78,120.0 bytes/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 262,144 |
| `qm` | bf16 | (65536, 4096) | 536,870,912 |
| `km` | bf16 | (65536, 1024) | 134,217,728 |
| `rstd_q` | fp32 | (2097152,) | 8,388,608 |
| `rstd_k` | fp32 | (524288,) | 2,097,152 |
| `v` | bf16 | (65536, 1024) | 134,217,728 |
| `lse` | fp32 | (512, 4096) | 8,388,608 |
| `attn_out` | bf16 | (65536, 4096) | 536,870,912 |
| `h_mid` | bf16 | (65536, 4096) | 536,870,912 |
| `rstd_ffn` | fp32 | (65536,) | 262,144 |
| `x1` | bf16 | (65536, 12288) | 1,610,612,736 |
| `x3` | bf16 | (65536, 12288) | 1,610,612,736 |

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
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): index_select

### `block_fwd` — `Qwen3BlockFwd`

- example task: `block_fwd_0_0_0`
- inputs: `y_embed_0_0` (536,870,912B), `W_0` (385,892,864B)
- outputs: `y_0_0_0` (536,870,912B), `A_0_0_0` (5,119,672,320B)
- mutates: —
- stages (name — emitted ctx fields):
    0. `attn_norm` — rstd_attn
    1. `qkv_qknorm` — qm, km, rstd_q, rstd_k, v
    2. `rope` — —
    3. `attn` — lse, attn_out
    4. `resid1_norm2` — h_mid, rstd_ffn
    5. `up_proj` — x1, x3  ← derived recompute boundary
    6. `swiglu` — —
    7. `down_resid` — —
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): rmsnorm_fwd → mm×3 → rmsnorm_fwd×2 → rope_fwd×2 → _scaled_dot_product_flash_attention → addmm → rmsnorm_fwd → mm×2 → swiglu_fwd_out → addmm

### `head_loss` — `HeadLoss`

- example task: `head_loss_0_0`
- inputs: `y_0_0_35` (536,870,912B), `targets_0_0` (262,144B), `W_head` (1,244,667,904B)
- outputs: `dy_0_0_35` (536,870,912B), `loss_0_0` (4B), `dW_head_0` (1,244,667,904B)
- mutates: —
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): rmsnorm_fwd → mm → ce_loss_fwd_bwd → mm×2 → rmsnorm_bwd

### `optimizer_head` — `AdamWStep`

- example task: `optimizer_head_0`
- inputs: `W_head` (1,244,667,904B), `dW_head_0` (1,244,667,904B), `O_head` (2,489,335,808B)
- outputs: —
- mutates: `W_head`, `O_head`
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): adamw_step×2

### `block_bwd` — `Qwen3BlockBwd`

- example task: `block_bwd_0_0_35`
- inputs: `dy_0_0_35` (536,870,912B), `A_0_0_35` (5,119,672,320B), `y_0_0_34` (536,870,912B), `W_35` (385,892,864B)
- outputs: `dy_0_0_34` (536,870,912B), `dW_0_35` (385,892,864B)
- mutates: —
- kernel calls (traced once at tiny dims; per-sequence op counts scale with microbatch): rmsnorm_apply → swiglu_fwd_out → mm×2 → swiglu_bwd → mm×3 → rmsnorm_bwd → mm×2 → rmsnorm_apply×2 → rope_fwd×2 → _scaled_dot_product_flash_attention_backward → rope_bwd×2 → rmsnorm_bwd×2 → rmsnorm_apply → mm×4 → rmsnorm_bwd

### `optimizer_block` — `AdamWStep`

- example task: `optimizer_0_35`
- inputs: `W_35` (385,892,864B), `dW_0_35` (385,892,864B), `O_35` (771,785,728B)
- outputs: —
- mutates: `W_35`, `O_35`
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

