# qwen3: tasks, objects, kernels

GENERATED from `ShapedQwen3Config.tiny()` — regenerate with `python tools/gen_model_docs.py --family qwen3`. Presets: [builtin_models.md](../builtin_models.md); task-kind fleet index: [task_kinds.md](../task_kinds.md).

Layer kinds (2 layers): `block block`

## Dims (documentation preset)

| field | value |
|---|---|
| `d_model` | 256 |
| `n_heads` | 4 |
| `n_kv_heads` | 2 |
| `head_dim` | 64 |
| `d_ff` | 512 |
| `vocab_size` | 512 |
| `tokens` | 64 |
| `seq_len` | 64 |
| `rope_base` | 1000000.0 |
| `opt_policy` | adamw |

## Objects, per layer kind

`dW_i` mirrors `W_i`'s fields at the grad dtypes; `O_i` holds the optimizer policy's state slots per field (adamw default: `m_f`+`v_f` at the opt dtype; sgd fields contribute none — see extending.md §6). `A_i`/`M_i` exist per (step, round).

### kind `block` (e.g. layer 0)

**`W_0` weights** — 1,181,184 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `attn_norm_w` | bf16 | (256,) | 512 |
| `wq` | bf16 | (256, 256) | 131,072 |
| `wk` | bf16 | (256, 128) | 65,536 |
| `wv` | bf16 | (256, 128) | 65,536 |
| `q_norm_w` | bf16 | (64,) | 128 |
| `k_norm_w` | bf16 | (64,) | 128 |
| `wo` | bf16 | (256, 256) | 131,072 |
| `ffn_norm_w` | bf16 | (256,) | 512 |
| `w1` | bf16 | (256, 512) | 262,144 |
| `w3` | bf16 | (256, 512) | 262,144 |
| `w2` | bf16 | (512, 256) | 262,144 |

**`A_.._0` saved context** — 265,216 bytes (per (step, round))

| field | dtype | shape | bytes |
|---|---|---|---|
| `rstd_attn` | fp32 | (64,) | 256 |
| `qm` | bf16 | (64, 256) | 32,768 |
| `km` | bf16 | (64, 128) | 16,384 |
| `rstd_q` | fp32 | (256,) | 1,024 |
| `rstd_k` | fp32 | (128,) | 512 |
| `v` | bf16 | (64, 128) | 16,384 |
| `lse` | fp32 | (4, 64) | 1,024 |
| `attn_out` | bf16 | (64, 256) | 32,768 |
| `h_mid` | bf16 | (64, 256) | 32,768 |
| `rstd_ffn` | fp32 | (64,) | 256 |
| `x1` | bf16 | (64, 512) | 65,536 |
| `x3` | bf16 | (64, 512) | 65,536 |

**`W_head`** — 262,656 bytes

| field | dtype | shape | bytes |
|---|---|---|---|
| `w` | bf16 | (512, 256) | 262,144 |
| `final_norm_w` | bf16 | (256,) | 512 |

## Tasks

### `embed_fwd` — `EmbedFwd`

- example task: `embed_fwd_0_0`
- inputs: `tokens_0_0` (256B), `W_embed` (262,144B)
- outputs: `y_embed_0_0` (32,768B)
- mutates: —

### `block_fwd` — `Qwen3BlockFwd`

- example task: `block_fwd_0_0_0`
- inputs: `y_embed_0_0` (32,768B), `W_0` (1,181,184B)
- outputs: `y_0_0_0` (32,768B), `A_0_0_0` (265,216B)
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
- kernel calls (measured, one launch): rmsnorm_fwd×3 → rope_fwd×2 → rmsnorm_fwd → swiglu_fwd_out

### `head_loss` — `HeadLoss`

- example task: `head_loss_0_0`
- inputs: `y_0_0_1` (32,768B), `targets_0_0` (256B), `W_head` (262,656B)
- outputs: `dy_0_0_1` (32,768B), `loss_0_0` (4B), `dW_head_0` (262,656B)
- mutates: —
- kernel calls (measured, one launch): rmsnorm_fwd → ce_loss_fwd_bwd → rmsnorm_bwd

### `optimizer_head` — `AdamWStep`

- example task: `optimizer_head_0`
- inputs: `W_head` (262,656B), `dW_head_0` (262,656B), `O_head` (525,312B)
- outputs: —
- mutates: `W_head`, `O_head`
- kernel calls (measured, one launch): adamw_step×2

### `block_bwd` — `Qwen3BlockBwd`

- example task: `block_bwd_0_0_1`
- inputs: `dy_0_0_1` (32,768B), `A_0_0_1` (265,216B), `y_0_0_0` (32,768B), `W_1` (1,181,184B)
- outputs: `dy_0_0_0` (32,768B), `dW_0_1` (1,181,184B)
- mutates: —
- kernel calls (measured, one launch): rmsnorm_apply → swiglu_fwd_out → swiglu_bwd → rmsnorm_bwd → rmsnorm_apply×2 → rope_fwd×2 → rope_bwd×2 → rmsnorm_bwd×2 → rmsnorm_apply → rmsnorm_bwd

### `optimizer_block` — `AdamWStep`

- example task: `optimizer_0_1`
- inputs: `W_1` (1,181,184B), `dW_0_1` (1,181,184B), `O_1` (2,362,368B)
- outputs: —
- mutates: `W_1`, `O_1`
- kernel calls (measured, one launch): adamw_step×11

### `embed_bwd` — `EmbedBwd`

- example task: `embed_bwd_0_0`
- inputs: `dy_embed_0_0` (32,768B), `tokens_0_0` (256B)
- outputs: `dW_embed_0` (262,144B)
- mutates: —
- kernel calls (measured, one launch): embed_bwd_accum

### `optimizer_embed` — `AdamWStep`

- example task: `optimizer_embed_0`
- inputs: `W_embed` (262,144B), `dW_embed_0` (262,144B), `O_embed` (524,288B)
- outputs: —
- mutates: `W_embed`, `O_embed`
- kernel calls (measured, one launch): adamw_step

