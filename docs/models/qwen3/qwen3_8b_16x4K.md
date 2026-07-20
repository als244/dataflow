# qwen3 / `qwen3_8b` @ 16x4K: tasks, objects, kernels

GENERATED from `ShapedQwen3Config.qwen3_8b()` at run shape microbatch 16 × seq 4096 — regenerate with `python tools/gen_model_page.py --preset qwen3_8b --microbatch 16 --seq-len 4096`. All presets: [builtin_models.md](../../builtin_models.md); task-kind fleet index: [task_kinds.md](../../task_kinds.md).

Layer kinds (36 layers): `block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block block`

**Run shape**: microbatch 16 × seq_len 4096 = **65,536 tokens per round** (× 1 grad-accum round(s) per step). `A_*`/`AuxTemp_*` objects are sized per round; bytes/token figures transfer to any run shape.

## Object summary

At this run shape (65,536 tokens/round). Token-scaled objects show per-token size in parens. Details per kind below.

| object | scope | size |
|---|---|---|
| `W_i (block)` | layer | 368.02 MiB |
| `dW_i (block)` | layer/step | 368.02 MiB |
| `O_i (block)` | layer | 736.03 MiB |
| `A (block)` | layer × round | 4.77 GiB (76.29 KiB/token) |
| `W_head` | run | 1.16 GiB |
| `W_embed` | run | 1.16 GiB |
| `O_embed` | run | 2.32 GiB |
| `O_head` | run | 2.32 GiB |
| `hidden state (y)` | boundary buffer | 512.00 MiB (8.00 KiB/token) |

### Aggregate totals (all layers, this run shape)

| type | objects | total size |
|---|---|---|
| W (all weights, incl. embed/head) | 38 | 15.26 GiB |
| dW (all gradients, per step) | 38 | 15.26 GiB |
| O (all optimizer state) | 38 | 30.51 GiB |
| A (all saved activations, one round) | 36 | 171.65 GiB (2.68 MiB/token) |

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

**`W_0` weights** — 368.02 MiB

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (4096,) | 8.00 KiB |
| `wq` | bf16 | (4096, 4096) | 32.00 MiB |
| `wk` | bf16 | (4096, 1024) | 8.00 MiB |
| `wv` | bf16 | (4096, 1024) | 8.00 MiB |
| `q_norm_w` | bf16 | (128,) | 256 B |
| `k_norm_w` | bf16 | (128,) | 256 B |
| `wo` | bf16 | (4096, 4096) | 32.00 MiB |
| `ffn_norm_w` | bf16 | (4096,) | 8.00 KiB |
| `w1` | bf16 | (4096, 12288) | 96.00 MiB |
| `w3` | bf16 | (4096, 12288) | 96.00 MiB |
| `w2` | bf16 | (12288, 4096) | 96.00 MiB |

**`A_.._0` saved context** — 4.77 GiB = **76.29 KiB/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (65536,) | 256.00 KiB |
| `qm` | bf16 | (65536, 4096) | 512.00 MiB |
| `km` | bf16 | (65536, 1024) | 128.00 MiB |
| `rstd_q` | fp32 | (2097152,) | 8.00 MiB |
| `rstd_k` | fp32 | (524288,) | 2.00 MiB |
| `v` | bf16 | (65536, 1024) | 128.00 MiB |
| `lse` | fp32 | (32, 65536) | 8.00 MiB |
| `attn_out` | bf16 | (65536, 4096) | 512.00 MiB |
| `h_mid` | bf16 | (65536, 4096) | 512.00 MiB |
| `rstd_ffn` | fp32 | (65536,) | 256.00 KiB |
| `x1` | bf16 | (65536, 12288) | 1.50 GiB |
| `x3` | bf16 | (65536, 12288) | 1.50 GiB |

**`W_head`** — 1.16 GiB

| field | dtype | shape | bytes |
|---|---|---|---|
| `w` | bf16 | (151936, 4096) | 1.16 GiB |
| `final_norm_w` | bf16 | (4096,) | 8.00 KiB |

## Tasks

### `embed_fwd` — `EmbedFwd`

- example task: `embed_fwd_0_0`
- inputs: `tokens_0_0` (256.00 KiB), `W_embed` (1.16 GiB)
- outputs: `y_embed_0_0` (512.00 MiB)
- mutates: —
- kernel calls:
    0. `index_select`

### `block_fwd` — `Qwen3BlockFwd`

- example task: `block_fwd_0_0_0`
- inputs: `y_embed_0_0` (512.00 MiB), `W_0` (368.02 MiB)
- outputs: `y_0_0_0` (512.00 MiB), `A_0_0_0` (4.77 GiB)
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
- inputs: `y_0_0_35` (512.00 MiB), `targets_0_0` (256.00 KiB), `W_head` (1.16 GiB)
- outputs: `dy_0_0_35` (512.00 MiB), `loss_0_0` (4 B), `dW_head_0` (1.16 GiB)
- mutates: —
- kernel calls:
    0. `rmsnorm_fwd`
    1. `mm`
    2. `ce_loss_fwd_bwd`
    3. `mm ×2`
    4. `rmsnorm_bwd`

### `optimizer_head` — `AdamWStep`

- example task: `optimizer_head_0`
- inputs: `W_head` (1.16 GiB), `dW_head_0` (1.16 GiB), `O_head` (2.32 GiB)
- outputs: —
- mutates: `W_head`, `O_head`
- kernel calls:
    0. `adamw_step ×2`

### `block_bwd` — `Qwen3BlockBwd`

- example task: `block_bwd_0_0_35`
- inputs: `dy_0_0_35` (512.00 MiB), `A_0_0_35` (4.77 GiB), `y_0_0_34` (512.00 MiB), `W_35` (368.02 MiB)
- outputs: `dy_0_0_34` (512.00 MiB), `dW_0_35` (368.02 MiB)
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

- example task: `optimizer_0_35`
- inputs: `W_35` (368.02 MiB), `dW_0_35` (368.02 MiB), `O_35` (736.03 MiB)
- outputs: —
- mutates: `W_35`, `O_35`
- kernel calls:
    0. `adamw_step ×11`

### `embed_bwd` — `EmbedBwd`

- example task: `embed_bwd_0_0`
- inputs: `dy_embed_0_0` (512.00 MiB), `tokens_0_0` (256.00 KiB)
- outputs: `dW_embed_0` (1.16 GiB)
- mutates: —
- kernel calls:
    0. `embed_bwd_accum`

### `optimizer_embed` — `AdamWStep`

- example task: `optimizer_embed_0`
- inputs: `W_embed` (1.16 GiB), `dW_embed_0` (1.16 GiB), `O_embed` (2.32 GiB)
- outputs: —
- mutates: `W_embed`, `O_embed`
- kernel calls:
    0. `adamw_step`

