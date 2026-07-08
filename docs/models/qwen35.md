# qwen35: tasks, objects, kernels

GENERATED from `ShapedQwen35Config.tiny()` — regenerate with `python tools/gen_model_docs.py --family qwen35`. Presets: [builtin_models.md](../builtin_models.md); task-kind fleet index: [task_kinds.md](../task_kinds.md).

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
| `d_ff` | 512 |
| `vocab_size` | 512 |
| `tokens` | 128 |
| `seq_len` | 128 |
| `rope_base` | 10000000.0 |
| `opt_policy` | adamw |

## Objects, per layer kind

`dW_i` mirrors `W_i`'s fields at the grad dtypes; `O_i` holds the optimizer policy's state slots per field (adamw default: `m_f`+`v_f` at the opt dtype; sgd fields contribute none — see extending.md §6). `A_i`/`M_i` exist per (step, round).

### kind `lin` (e.g. layer 0)

**`W_0` weights** — 1,056,512 bytes

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
| `w1` | bf16 | (256, 512) | 262,144 |
| `w3` | bf16 | (256, 512) | 262,144 |
| `w2` | bf16 | (512, 256) | 262,144 |

**`A_.._0` saved context** — 531,456 bytes (per (step, round))

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
| `x1` | bf16 | (128, 512) | 131,072 |
| `x3` | bf16 | (128, 512) | 131,072 |

### kind `full` (e.g. layer 3)

**`W_3` weights** — 1,312,256 bytes

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
| `w1` | bf16 | (256, 512) | 262,144 |
| `w3` | bf16 | (256, 512) | 262,144 |
| `w2` | bf16 | (512, 256) | 262,144 |

**`A_.._3` saved context** — 595,968 bytes (per (step, round))

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
| `x1` | bf16 | (128, 512) | 131,072 |
| `x3` | bf16 | (128, 512) | 131,072 |

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

### `linattn_fwd` — `Qwen35LinBlockFwd`

- example task: `block_fwd_0_0_0`
- inputs: `y_embed_0_0` (65,536B), `W_0` (1,056,512B)
- outputs: `y_0_0_0` (65,536B), `A_0_0_0` (531,456B)
- mutates: —
- stages (name — emitted ctx fields):
    0. `attn_norm` — rstd_attn
    1. `proj` — qkvz, ba
    2. `conv` — —
    3. `heads_l2norm` — —
    4. `fla` — g_post, A_int, core_out
    5. `norm_out` — rstd_gate, xo
    6. `ffn_norm` — rstd_ffn
    7. `up_proj` — x1, x3  ← derived recompute boundary
    8. `swiglu` — —
    9. `down_resid` — —
- kernel calls (measured, one launch): rmsnorm_fwd → causal_conv1d_silu_fwd → gated_rmsnorm_fwd → rmsnorm_fwd → swiglu_fwd_out

### `gattn_fwd` — `Qwen35AttnBlockFwd`

- example task: `block_fwd_0_0_3`
- inputs: `y_0_0_2` (65,536B), `W_3` (1,312,256B)
- outputs: `y_0_0_3` (65,536B), `A_0_0_3` (595,968B)
- mutates: —
- stages (name — emitted ctx fields):
    0. `attn_norm` — rstd_attn
    1. `qkv_gate` — qm, km, gate, v
    2. `qknorm_rope` — rstd_q, rstd_k
    3. `attn` — lse, attn_out
    4. `gate_o` — xo
    5. `ffn_norm` — rstd_ffn
    6. `up_proj` — x1, x3  ← derived recompute boundary
    7. `swiglu` — —
    8. `down_resid` — —
- kernel calls (measured, one launch): rmsnorm_fwd×3 → rope_fwd×2 → rmsnorm_fwd → swiglu_fwd_out

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

### `gattn_bwd` — `Qwen35AttnBlockBwd`

- example task: `block_bwd_0_0_3`
- inputs: `dy_0_0_3` (65,536B), `A_0_0_3` (595,968B), `y_0_0_2` (65,536B), `W_3` (1,312,256B)
- outputs: `dy_0_0_2` (65,536B), `dW_0_3` (1,312,256B)
- mutates: —
- kernel calls (measured, one launch): rmsnorm_apply → swiglu_fwd_out → swiglu_bwd → rmsnorm_bwd → rmsnorm_apply×2 → rope_fwd×2 → rope_bwd×2 → rmsnorm_bwd×2 → rmsnorm_apply → rmsnorm_bwd

### `optimizer_block` — `AdamWStep`

- example task: `optimizer_0_3`
- inputs: `W_3` (1,312,256B), `dW_0_3` (1,312,256B), `O_3` (2,624,512B)
- outputs: —
- mutates: `W_3`, `O_3`
- kernel calls (measured, one launch): adamw_step×11

### `linattn_bwd` — `Qwen35LinBlockBwd`

- example task: `block_bwd_0_0_2`
- inputs: `dy_0_0_2` (65,536B), `A_0_0_2` (531,456B), `y_0_0_1` (65,536B), `W_2` (1,056,512B)
- outputs: `dy_0_0_1` (65,536B), `dW_0_2` (1,056,512B)
- mutates: —
- kernel calls (measured, one launch): rmsnorm_apply → swiglu_fwd_out → swiglu_bwd → rmsnorm_bwd → gated_rmsnorm_bwd → causal_conv1d_silu_fwd → causal_conv1d_silu_bwd → rmsnorm_apply → rmsnorm_bwd

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

