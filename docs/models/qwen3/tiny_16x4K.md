# qwen3 / `tiny` @ 16x4K: tasks, objects, kernels

GENERATED from `ShapedQwen3Config.tiny()` at run shape microbatch 16 × seq 4096 — regenerate with `python tools/gen_model_page.py --preset tiny --microbatch 16 --seq-len 4096`. All presets: [builtin_models.md](../../builtin_models.md); task-kind fleet index: [task_kinds.md](../../task_kinds.md).

Layer kinds (2 layers): `block block`

**Run shape**: microbatch 16 × seq_len 4096 = **65,536 tokens per round** (× 1 grad-accum round(s) per step). `A_*`/`M_*` objects are sized per round; bytes/token figures transfer to any run shape.

## Object summary

At this run shape (65,536 tokens/round). Token-scaled objects show per-token size in parens. Details per kind below.

| object | scope | size |
|---|---|---|
| `W_i (block)` | layer | 1.13 MiB |
| `dW_i (block)` | layer/step | 1.13 MiB |
| `O_i (block)` | layer | 2.25 MiB |
| `A (block)` | layer × round | 259.00 MiB (4.05 KiB/token) |
| `W_head` | run | 256.50 KiB |
| `W_embed` | run | 256.00 KiB |
| `O_embed` | run | 512.00 KiB |
| `O_head` | run | 513.00 KiB |
| `hidden state (y)` | boundary buffer | 32.00 MiB (512.0 B/token) |

### Aggregate totals (all layers, this run shape)

| type | objects | total size |
|---|---|---|
| W (all weights, incl. embed/head) | 4 | 2.75 MiB |
| dW (all gradients, per step) | 4 | 2.75 MiB |
| O (all optimizer state) | 4 | 5.51 MiB |
| A (all saved activations, one round) | 2 | 518.00 MiB (8.09 KiB/token) |

## Dims

| field | value |
|---|---|
| `d_model` | 256 |
| `n_heads` | 4 |
| `n_kv_heads` | 2 |
| `head_dim` | 64 |
| `d_ff` | 512 |
| `vocab_size` | 512 |
| `tokens` | 65536 |
| `seq_len` | 4096 |
| `rope_base` | 1000000.0 |
| `opt_policy` | adamw |

## Objects, per layer kind

`dW_i` mirrors `W_i`'s fields at the grad dtypes; `O_i` holds the optimizer policy's state slots per field (adamw default: `m_f`+`v_f` at the opt dtype; sgd fields contribute none — see extending.md §6). `A_i`/`M_i` exist per (step, round).

### kind `block` (e.g. layer 0)

**`W_0` weights** — 1.13 MiB

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (256,) | 512 B |
| `wq` | bf16 | (256, 256) | 128.00 KiB |
| `wk` | bf16 | (256, 128) | 64.00 KiB |
| `wv` | bf16 | (256, 128) | 64.00 KiB |
| `q_norm_w` | bf16 | (64,) | 128 B |
| `k_norm_w` | bf16 | (64,) | 128 B |
| `wo` | bf16 | (256, 256) | 128.00 KiB |
| `ffn_norm_w` | bf16 | (256,) | 512 B |
| `w1` | bf16 | (256, 512) | 256.00 KiB |
| `w3` | bf16 | (256, 512) | 256.00 KiB |
| `w2` | bf16 | (512, 256) | 256.00 KiB |

**`A_.._0` saved context** — 259.00 MiB = **4.05 KiB/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 256.00 KiB |
| `qm` | bf16 | (65536, 256) | 32.00 MiB |
| `km` | bf16 | (65536, 128) | 16.00 MiB |
| `rstd_q` | fp32 | (262144,) | 1.00 MiB |
| `rstd_k` | fp32 | (131072,) | 512.00 KiB |
| `v` | bf16 | (65536, 128) | 16.00 MiB |
| `lse` | fp32 | (4, 65536) | 1.00 MiB |
| `attn_out` | bf16 | (65536, 256) | 32.00 MiB |
| `h_mid` | bf16 | (65536, 256) | 32.00 MiB |
| `rstd_ffn` | fp32 | (65536,) | 256.00 KiB |
| `x1` | bf16 | (65536, 512) | 64.00 MiB |
| `x3` | bf16 | (65536, 512) | 64.00 MiB |

**`W_head`** — 256.50 KiB

| field | dtype | shape | bytes |
|---|---|---|---|
| `w` | bf16 | (512, 256) | 256.00 KiB |
| `final_norm_w` | bf16 | (256,) | 512 B |

## Tasks

### `embed_fwd` — `EmbedFwd`

- example task: `embed_fwd_0_0`
- inputs: `tokens_0_0` (256.00 KiB), `W_embed` (256.00 KiB)
- outputs: `y_embed_0_0` (32.00 MiB)
- mutates: —
- kernel calls:
    0. `index_select`

### `block_fwd` — `Qwen3BlockFwd`

- example task: `block_fwd_0_0_0`
- inputs: `y_embed_0_0` (32.00 MiB), `W_0` (1.13 MiB)
- outputs: `y_0_0_0` (32.00 MiB), `A_0_0_0` (259.00 MiB)
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
- kernel calls, by stage:
    - `attn_norm`:
        0. `rmsnorm_fwd`
    - `qkv_qknorm`:
        1. `mm ×3`
        2. `rmsnorm_fwd ×2`
    - `rope`:
        3. `rope_fwd ×2`
    - `resid1_norm2`:
        4. `addmm`
        5. `rmsnorm_fwd`
    - `up_proj`:
        6. `mm ×2`
    - `swiglu`:
        7. `swiglu_fwd_out`
    - `down_resid`:
        8. `addmm`

### `head_loss` — `HeadLoss`

- example task: `head_loss_0_0`
- inputs: `y_0_0_1` (32.00 MiB), `targets_0_0` (256.00 KiB), `W_head` (256.50 KiB)
- outputs: `dy_0_0_1` (32.00 MiB), `loss_0_0` (4 B), `dW_head_0` (256.50 KiB)
- mutates: —
- kernel calls:
    0. `rmsnorm_fwd`
    1. `mm`
    2. `ce_loss_fwd_bwd`
    3. `mm ×2`
    4. `rmsnorm_bwd`

### `optimizer_head` — `AdamWStep`

- example task: `optimizer_head_0`
- inputs: `W_head` (256.50 KiB), `dW_head_0` (256.50 KiB), `O_head` (513.00 KiB)
- outputs: —
- mutates: `W_head`, `O_head`
- kernel calls:
    0. `adamw_step ×2`

### `block_bwd` — `Qwen3BlockBwd`

- example task: `block_bwd_0_0_1`
- inputs: `dy_0_0_1` (32.00 MiB), `A_0_0_1` (259.00 MiB), `y_0_0_0` (32.00 MiB), `W_1` (1.13 MiB)
- outputs: `dy_0_0_0` (32.00 MiB), `dW_0_1` (1.13 MiB)
- mutates: —
- kernel calls:
    0. `rmsnorm_apply`
    1. `swiglu_fwd_out`
    2. `mm ×2`
    3. `swiglu_bwd`
    4. `mm ×3`
    5. `rmsnorm_bwd`
    6. `mm ×2`
    7. `rmsnorm_apply ×2`
    8. `rope_fwd ×2`
    9. `_flash_attention_backward`
    10. `rope_bwd ×2`
    11. `rmsnorm_bwd ×2`
    12. `rmsnorm_apply`
    13. `mm ×4`
    14. `rmsnorm_bwd`

### `optimizer_block` — `AdamWStep`

- example task: `optimizer_0_1`
- inputs: `W_1` (1.13 MiB), `dW_0_1` (1.13 MiB), `O_1` (2.25 MiB)
- outputs: —
- mutates: `W_1`, `O_1`
- kernel calls:
    0. `adamw_step ×11`

### `embed_bwd` — `EmbedBwd`

- example task: `embed_bwd_0_0`
- inputs: `dy_embed_0_0` (32.00 MiB), `tokens_0_0` (256.00 KiB)
- outputs: `dW_embed_0` (256.00 KiB)
- mutates: —
- kernel calls:
    0. `embed_bwd_accum`

### `optimizer_embed` — `AdamWStep`

- example task: `optimizer_embed_0`
- inputs: `W_embed` (256.00 KiB), `dW_embed_0` (256.00 KiB), `O_embed` (512.00 KiB)
- outputs: —
- mutates: `W_embed`, `O_embed`
- kernel calls:
    0. `adamw_step`

