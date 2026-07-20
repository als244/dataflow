# gpt2 / `tiny` @ 16x4K: tasks, objects, kernels

GENERATED from `ShapedGpt2Config.tiny()` at run shape microbatch 16 × seq 4096 — regenerate with `python tools/gen_model_docs/gen_model_page.py --preset tiny --microbatch 16 --seq-len 4096`. All presets: [builtin_models.md](../../builtin_models.md); task-kind fleet index: [task_kinds.md](../../task_kinds.md).

Layer kinds (2 layers): `block block`

**Run shape**: microbatch 16 × seq_len 4096 = **65,536 tokens per round** (× 1 grad-accum round(s) per step). `A_*`/`AuxTemp_*` objects are sized per round; bytes/token figures transfer to any run shape.

## Object summary

At this run shape (65,536 tokens/round). Token-scaled objects show per-token size in parens. Details per kind below.

| object | scope | size |
|---|---|---|
| `W_i (block)` | layer | 74.50 KiB |
| `dW_i (block)` | layer/step | 74.50 KiB |
| `O_i (block)` | layer | 149.00 KiB |
| `A (block)` | layer × round | 62.00 MiB (992.0 B/token) |
| `W_head` | run | 64.25 KiB |
| `W_embed` | run | 576.00 KiB |
| `O_embed` | run | 1.12 MiB |
| `O_head` | run | 129.00 KiB |
| `hidden state (y)` | boundary buffer | 8.00 MiB (128.0 B/token) |

### Aggregate totals (all layers, this run shape)

| type | objects | total size |
|---|---|---|
| W (all weights, incl. embed/head) | 4 | 789.50 KiB |
| dW (all gradients, per step) | 4 | 789.50 KiB |
| O (all optimizer state) | 4 | 1.54 MiB |
| A (all saved activations, one round) | 2 | 124.00 MiB (1.94 KiB/token) |

## Dims

| field | value |
|---|---|
| `d_model` | 64 |
| `n_heads` | 4 |
| `d_ff` | 160 |
| `vocab_size` | 512 |
| `tokens` | 65536 |
| `seq_len` | 4096 |
| `n_ctx` | 4096 |
| `tied` | False |
| `use_bias` | True |
| `opt_policy` | adamw |

## Objects, per layer kind

`dW_i` mirrors `W_i`'s fields at the grad dtypes; `O_i` holds the optimizer policy's state slots per field (adamw default: `m_f`+`v_f` at the opt dtype; sgd fields contribute none — see extending.md §6). `A_i`/`M_i` exist per (step, round).

### kind `block` (e.g. layer 0)

**`W_0` weights** — 74.50 KiB

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (64,) | 128 B |
| `attn_norm_b` | bf16 | (64,) | 128 B |
| `w_qkv` | bf16 | (64, 192) | 24.00 KiB |
| `b_qkv` | bf16 | (192,) | 384 B |
| `wo` | bf16 | (64, 64) | 8.00 KiB |
| `b_o` | bf16 | (64,) | 128 B |
| `ffn_norm_w` | bf16 | (64,) | 128 B |
| `ffn_norm_b` | bf16 | (64,) | 128 B |
| `w_fc` | bf16 | (64, 160) | 20.00 KiB |
| `b_fc` | bf16 | (160,) | 320 B |
| `w_proj` | bf16 | (160, 64) | 20.00 KiB |
| `b_proj` | bf16 | (64,) | 128 B |

**`A_.._0` saved context** — 62.00 MiB = **992.0 B/token** (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `mean_attn` | fp32 | (65536,) | 256.00 KiB |
| `rstd_attn` | fp32 | (65536,) | 256.00 KiB |
| `q` | bf16 | (65536, 64) | 8.00 MiB |
| `k` | bf16 | (65536, 64) | 8.00 MiB |
| `v` | bf16 | (65536, 64) | 8.00 MiB |
| `lse` | fp32 | (4, 65536) | 1.00 MiB |
| `attn_out` | bf16 | (65536, 64) | 8.00 MiB |
| `h_mid` | bf16 | (65536, 64) | 8.00 MiB |
| `mean_ffn` | fp32 | (65536,) | 256.00 KiB |
| `rstd_ffn` | fp32 | (65536,) | 256.00 KiB |
| `x_fc` | bf16 | (65536, 160) | 20.00 MiB |

**`W_head`** — 64.25 KiB

| field | dtype | shape | bytes |
|---|---|---|---|
| `w` | bf16 | (512, 64) | 64.00 KiB |
| `final_norm_w` | bf16 | (64,) | 128 B |

## Tasks

### `embed_fwd` — `Gpt2EmbedFwd`

- example task: `embed_fwd_0_0`
- inputs: `tokens_0_0` (256.00 KiB), `W_embed` (576.00 KiB)
- outputs: `y_embed_0_0` (8.00 MiB)
- mutates: —
- kernel calls:
    0. `index_select ×2`

### `block_fwd` — `Gpt2BlockFwd`

- example task: `block_fwd_0_0_0`
- inputs: `y_embed_0_0` (8.00 MiB), `W_0` (74.50 KiB)
- outputs: `y_0_0_0` (8.00 MiB), `A_0_0_0` (62.00 MiB)
- mutates: —
- stages (name — emitted ctx fields):
    0. `ln1` — mean_attn, rstd_attn
    1. `qkv` — q, k, v
    2. `attn` — lse, attn_out
    3. `resid1_ln2` — h_mid, mean_ffn, rstd_ffn
    4. `fc_gelu` — x_fc  ← derived recompute boundary
    5. `proj_resid` — —
- kernel calls, by stage:
    - `ln1`:
        0. `layernorm_fwd`
    - `qkv`:
        1. `addmm`
    - `resid1_ln2`:
        2. `addmm`
        3. `layernorm_fwd`
    - `fc_gelu`:
        4. `addmm`
        5. `gelu_fwd_out`
    - `proj_resid`:
        6. `addmm`

### `head_loss` — `Gpt2HeadLoss`

- example task: `head_loss_0_0`
- inputs: `y_0_0_1` (8.00 MiB), `targets_0_0` (256.00 KiB), `W_head` (64.50 KiB)
- outputs: `dy_0_0_1` (8.00 MiB), `loss_0_0` (4 B), `dW_head_0` (64.50 KiB)
- mutates: —
- kernel calls:
    0. `layernorm_fwd`
    1. `mm`
    2. `ce_loss_fwd_bwd`
    3. `mm ×2`
    4. `layernorm_bwd`

### `optimizer_head` — `AdamWStep`

- example task: `optimizer_head_0`
- inputs: `W_head` (64.50 KiB), `dW_head_0` (64.50 KiB), `O_head` (129.00 KiB)
- outputs: —
- mutates: `W_head`, `O_head`
- kernel calls:
    0. `adamw_step ×3`

### `block_bwd` — `Gpt2BlockBwd`

- example task: `block_bwd_0_0_1`
- inputs: `dy_0_0_1` (8.00 MiB), `A_0_0_1` (62.00 MiB), `y_0_0_0` (8.00 MiB), `W_1` (74.50 KiB)
- outputs: `dy_0_0_0` (8.00 MiB), `dW_0_1` (74.50 KiB)
- mutates: —
- kernel calls:
    0. `layernorm_apply`
    1. `gelu_fwd_out`
    2. `mm ×2`
    3. `gelu_bwd`
    4. `mm ×2`
    5. `layernorm_bwd`
    6. `mm ×2`
    7. `_flash_attention_backward`
    8. `layernorm_apply`
    9. `mm ×2`
    10. `layernorm_bwd`

### `optimizer_block` — `AdamWStep`

- example task: `optimizer_0_1`
- inputs: `W_1` (74.50 KiB), `dW_0_1` (74.50 KiB), `O_1` (149.00 KiB)
- outputs: —
- mutates: `W_1`, `O_1`
- kernel calls:
    0. `adamw_step ×12`

### `embed_bwd` — `Gpt2EmbedBwd`

- example task: `embed_bwd_0_0`
- inputs: `dy_embed_0_0` (8.00 MiB), `tokens_0_0` (256.00 KiB)
- outputs: `dW_embed_0` (576.00 KiB)
- mutates: —
- kernel calls:
    0. `embed_bwd_accum ×2`

### `optimizer_embed` — `AdamWStep`

- example task: `optimizer_embed_0`
- inputs: `W_embed` (576.00 KiB), `dW_embed_0` (576.00 KiB), `O_embed` (1.12 MiB)
- outputs: —
- mutates: `W_embed`, `O_embed`
- kernel calls:
    0. `adamw_step ×2`

