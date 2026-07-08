# Builtin model families and presets

GENERATED — regenerate with `python tools/list_models.py >
docs/builtin_models.md` after adding a family or preset. Families
register in `training/families.py`; presets are classmethods on each
family's Shaped config (external families: docs/extending_external.md).

Params are computed from the lowered weight layouts at each preset's
dtype policy (bf16 default). `tiny` presets are the correctness-ladder
scale (docs/extending.md); `mini` presets are single-GPU bench scale;
full-scale presets match the published architectures (dims verified
against the HF configs; totals match announced parameter counts).
One section per family; the extra columns in each table are that
family's OWN configuration axes (fields no other family shares).
Each preset name links to its generated deep reference (objects,
stages, kernels) at the standard 16×4K run shape; other run shapes:
`tools/gen_model_page.py`. Index: [models/](models/README.md).

## llama3 — `ShapedLlamaConfig`

| preset | layers | d_model | vocab | seq default | `d_ff` | `n_kv_heads` | params |
|---|---|---|---|---|---|---|---|
| [`llama3_8b`](models/llama3/llama3_8b_16x4K.md) | 32 | 4096 | 128256 | 4096 | 14336 | 8 | 8.03B |
| [`tiny`](models/llama3/tiny_16x4K.md) | 2 | 64 | 512 | 64 | 160 | 2 | 152K |

## qwen3 — `ShapedQwen3Config`

| preset | layers | d_model | vocab | seq default | `d_ff` | `head_dim` | `n_kv_heads` | params |
|---|---|---|---|---|---|---|---|---|
| [`qwen3_8b`](models/qwen3/qwen3_8b_16x4K.md) | 36 | 4096 | 151936 | 4096 | 12288 | 128 | 8 | 8.19B |
| [`tiny`](models/qwen3/tiny_16x4K.md) | 2 | 256 | 512 | 64 | 512 | 64 | 2 | 1.4M |

## qwen35 — `ShapedQwen35Config`

| preset | layers | d_model | vocab | seq default | `d_ff` | `full_attention_interval` | `head_dim` | `lin_conv_kernel` | `lin_k_head_dim` | `lin_k_heads` | `lin_v_head_dim` | `lin_v_heads` | `n_kv_heads` | `partial_rotary_factor` | `rope_base` | `tied_embeddings` | params |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [`qwen35_9b`](models/qwen35/qwen35_9b_16x4K.md) | 32 | 4096 | 248320 | 4096 | 12288 | 4 | 256 | 4 | 128 | 16 | 128 | 32 | 4 | 0.25 | 10000000.0 | no | 8.95B |
| [`tiny`](models/qwen35/tiny_16x4K.md) | 4 | 256 | 512 | 128 | 512 | 4 | 64 | 4 | 32 | 2 | 32 | 4 | 2 | 0.25 | 10000000.0 | no | 2.5M |
| [`tiny_tied`](models/qwen35/tiny_tied_16x4K.md) | 4 | 256 | 512 | 128 | 512 | 4 | 64 | 4 | 32 | 2 | 32 | 4 | 2 | 0.25 | 10000000.0 | yes | 2.4M |

## olmoe — `ShapedOlmoeConfig`

| preset | layers | d_model | vocab | seq default | `aux_coef` | `d_ff_expert` | `head_dim` | `n_experts` | `n_kv_heads` | `rope_base` | `routing_mode` | `top_k` | params |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [`olmoe_7b`](models/olmoe/olmoe_7b_16x4K.md) | 16 | 2048 | 50304 | 4096 | 0.01 | 1024 | 128 | 64 | 16 | 10000.0 | softmax_the... | 8 | 6.92B |
| [`tiny`](models/olmoe/tiny_16x4K.md) | 2 | 128 | 512 | 128 | 0.01 | 128 | 32 | 8 | 4 | 10000.0 | softmax_the... | 2 | 1.1M |

## qwen35moe — `ShapedQwen35MoeConfig`

| preset | layers | d_model | vocab | seq default | `aux_coef` | `d_ff_expert` | `d_ff_shared` | `full_attention_interval` | `head_dim` | `lin_conv_kernel` | `lin_k_head_dim` | `lin_k_heads` | `lin_v_head_dim` | `lin_v_heads` | `n_experts` | `n_kv_heads` | `n_shared_experts` | `partial_rotary_factor` | `rope_base` | `routing_mode` | `tied_embeddings` | `top_k` | params |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [`qwen35moe_20l`](models/qwen35moe/qwen35moe_20l_16x4K.md) | 20 | 2048 | 248320 | 4096 | 0.001 | 512 | 512 | 4 | 256 | 4 | 128 | 16 | 128 | 32 | 256 | 2 | 1 | 0.25 | 10000000.0 | topk_then_s... | no | 8 | 17.84B |
| [`qwen35moe_35b`](models/qwen35moe/qwen35moe_35b_16x4K.md) | 40 | 2048 | 248320 | 4096 | 0.001 | 512 | 512 | 4 | 256 | 4 | 128 | 16 | 128 | 32 | 256 | 2 | 1 | 0.25 | 10000000.0 | topk_then_s... | no | 8 | 34.66B |
| [`tiny`](models/qwen35moe/tiny_16x4K.md) | 4 | 256 | 512 | 128 | 0.001 | 128 | 128 | 4 | 64 | 4 | 32 | 2 | 32 | 4 | 8 | 2 | 1 | 0.25 | 10000000.0 | topk_then_s... | no | 2 | 4.5M |

## qwen3moe — `ShapedQwen3MoeConfig`

| preset | layers | d_model | vocab | seq default | `aux_coef` | `d_ff_expert` | `head_dim` | `n_experts` | `n_kv_heads` | `rope_base` | `routing_mode` | `top_k` | params |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [`qwen3moe_235b`](models/qwen3moe/qwen3moe_235b_16x4K.md) | 94 | 4096 | 151936 | 4096 | 0.001 | 1536 | 128 | 128 | 4 | 1000000.0 | topk_then_s... | 8 | 235.09B |
| [`qwen3moe_30b`](models/qwen3moe/qwen3moe_30b_16x4K.md) | 48 | 2048 | 151936 | 4096 | 0.001 | 768 | 128 | 128 | 4 | 1000000.0 | topk_then_s... | 8 | 30.53B |
| [`qwen3moe_30b_24l`](models/qwen3moe/qwen3moe_30b_24l_16x4K.md) | 24 | 2048 | 151936 | 4096 | 0.001 | 768 | 128 | 128 | 4 | 1000000.0 | topk_then_s... | 8 | 15.58B |
| [`tiny`](models/qwen3moe/tiny_16x4K.md) | 2 | 128 | 512 | 128 | 0.001 | 64 | 32 | 8 | 2 | 1000000.0 | topk_then_s... | 2 | 626K |

## dsv3 — `ShapedDsv3Config`

| preset | layers | d_model | vocab | seq default | `aux_coef` | `bias_update_speed` | `d_ff_dense` | `d_ff_expert` | `d_ff_shared` | `first_k_dense` | `kv_lora_rank` | `n_experts` | `n_group` | `n_shared_experts` | `q_lora_rank` | `qk_nope_dim` | `qk_rope_dim` | `rope_base` | `routed_scaling` | `top_k` | `topk_group` | `v_head_dim` | params |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [`dsv3_671b`](models/dsv3/dsv3_671b_16x4K.md) | 61 | 7168 | 129280 | 4096 | 0.0001 | 0.001 | 18432 | 2048 | 2048 | 3 | 512 | 256 | 8 | 1 | 1536 | 128 | 64 | 10000.0 | 2.5 | 8 | 4 | 128 | 671.03B |
| [`dsv3_mini`](models/dsv3/dsv3_mini_16x4K.md) | 18 | 2048 | 129280 | 4096 | 0.0001 | 0.001 | 8192 | 1024 | 1024 | 2 | 256 | 128 | 8 | 1 | 512 | 64 | 32 | 10000.0 | 2.5 | 8 | 4 | 64 | 13.71B |
| [`kimi_k2`](models/dsv3/kimi_k2_16x4K.md) | 61 | 7168 | 163840 | 4096 | 0.0001 | 0.001 | 18432 | 2048 | 2048 | 1 | 512 | 384 | 1 | 1 | 1536 | 128 | 64 | 50000.0 | 2.827 | 8 | 1 | 128 | 1.026T |
| [`kimi_k25`](models/dsv3/kimi_k25_16x4K.md) (alias of `kimi_k2`) | 61 | 7168 | 163840 | 4096 | 0.0001 | 0.001 | 18432 | 2048 | 2048 | 1 | 512 | 384 | 1 | 1 | 1536 | 128 | 64 | 50000.0 | 2.827 | 8 | 1 | 128 | 1.026T |
| [`kimi_k26`](models/dsv3/kimi_k26_16x4K.md) (alias of `kimi_k2`) | 61 | 7168 | 163840 | 4096 | 0.0001 | 0.001 | 18432 | 2048 | 2048 | 1 | 512 | 384 | 1 | 1 | 1536 | 128 | 64 | 50000.0 | 2.827 | 8 | 1 | 128 | 1.026T |
| [`kimi_k27`](models/dsv3/kimi_k27_16x4K.md) (alias of `kimi_k2`) | 61 | 7168 | 163840 | 4096 | 0.0001 | 0.001 | 18432 | 2048 | 2048 | 1 | 512 | 384 | 1 | 1 | 1536 | 128 | 64 | 50000.0 | 2.827 | 8 | 1 | 128 | 1.026T |
| [`tiny`](models/dsv3/tiny_16x4K.md) | 3 | 128 | 512 | 128 | 0.0001 | 0.001 | 256 | 32 | 32 | 1 | 32 | 8 | 4 | 1 | 64 | 16 | 8 | 10000.0 | 2.5 | 2 | 2 | 16 | 550K |

## dsv32 — `ShapedDsv32Config`

| preset | layers | d_model | vocab | seq default | `aux_coef` | `bias_update_speed` | `d_ff_dense` | `d_ff_expert` | `d_ff_shared` | `first_k_dense` | `index_head_dim` | `index_n_heads` | `index_topk` | `kv_lora_rank` | `n_experts` | `n_group` | `n_shared_experts` | `q_lora_rank` | `qk_nope_dim` | `qk_rope_dim` | `rope_base` | `routed_scaling` | `sparse_mode` | `top_k` | `topk_group` | `train_indexer` | `v_head_dim` | params |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [`dsv32_671b`](models/dsv32/dsv32_671b_16x4K.md) | 61 | 7168 | 129280 | 4096 | 0.0001 | 0.001 | 18432 | 2048 | 2048 | 3 | 128 | 64 | 2048 | 512 | 256 | 8 | 1 | 1536 | 128 | 64 | 10000.0 | 2.5 | yes | 8 | 4 | yes | 128 | 671.91B |
| [`dsv32_mini`](models/dsv32/dsv32_mini_16x4K.md) | 18 | 2048 | 129280 | 4096 | 0.0001 | 0.001 | 8192 | 1024 | 1024 | 2 | 64 | 8 | 1024 | 256 | 128 | 8 | 1 | 512 | 64 | 32 | 10000.0 | 2.5 | yes | 8 | 4 | yes | 64 | 13.72B |
| [`dsv32_mini_warmup`](models/dsv32/dsv32_mini_warmup_16x4K.md) (alias of `dsv32_mini`) | 18 | 2048 | 129280 | 4096 | 0.0001 | 0.001 | 8192 | 1024 | 1024 | 2 | 64 | 8 | 1024 | 256 | 128 | 8 | 1 | 512 | 64 | 32 | 10000.0 | 2.5 | no | 8 | 4 | yes | 64 | 13.72B |
| [`glm5`](models/dsv32/glm5_16x4K.md) | 78 | 6144 | 154880 | 4096 | 0.0001 | 0.001 | 12288 | 2048 | 2048 | 3 | 128 | 32 | 2048 | 512 | 256 | 1 | 1 | 2048 | 192 | 64 | 1000000.0 | 2.5 | yes | 8 | 1 | yes | 256 | 743.93B |
| [`glm51`](models/dsv32/glm51_16x4K.md) (alias of `glm5`) | 78 | 6144 | 154880 | 4096 | 0.0001 | 0.001 | 12288 | 2048 | 2048 | 3 | 128 | 32 | 2048 | 512 | 256 | 1 | 1 | 2048 | 192 | 64 | 1000000.0 | 2.5 | yes | 8 | 1 | yes | 256 | 743.93B |
| [`tiny`](models/dsv32/tiny_16x4K.md) | 3 | 128 | 512 | 128 | 0.0001 | 0.001 | 256 | 32 | 32 | 1 | 32 | 8 | 24 | 32 | 8 | 4 | 1 | 64 | 16 | 8 | 10000.0 | 2.5 | yes | 2 | 2 | yes | 16 | 618K |

## glm52 — `ShapedGlm52Config`

| preset | layers | d_model | vocab | seq default | `aux_coef` | `bias_update_speed` | `d_ff_dense` | `d_ff_expert` | `d_ff_shared` | `first_k_dense` | `index_head_dim` | `index_n_heads` | `index_topk` | `indexer_types` | `kv_lora_rank` | `n_experts` | `n_group` | `n_shared_experts` | `q_lora_rank` | `qk_nope_dim` | `qk_rope_dim` | `rope_base` | `routed_scaling` | `sparse_mode` | `top_k` | `topk_group` | `train_indexer` | `v_head_dim` | params |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [`glm52`](models/glm52/glm52_16x4K.md) | 78 | 6144 | 154880 | 4096 | 0.0001 | 0.001 | 12288 | 2048 | 2048 | 3 | 128 | 32 | 2048 | 57xshared/21xfull | 512 | 256 | 1 | 1 | 2048 | 192 | 64 | 8000000.0 | 2.5 | yes | 8 | 1 | yes | 256 | 743.38B |
| [`glm52_mini`](models/glm52/glm52_mini_16x4K.md) | 18 | 2048 | 129280 | 4096 | 0.0001 | 0.001 | 8192 | 1024 | 1024 | 2 | 64 | 8 | 1024 | 12xshared/6xfull | 256 | 128 | 8 | 1 | 512 | 64 | 32 | 10000.0 | 2.5 | yes | 8 | 4 | yes | 64 | 13.71B |
| [`glm52_mini_warmup`](models/glm52/glm52_mini_warmup_16x4K.md) (alias of `glm52_mini`) | 18 | 2048 | 129280 | 4096 | 0.0001 | 0.001 | 8192 | 1024 | 1024 | 2 | 64 | 8 | 1024 | 12xshared/6xfull | 256 | 128 | 8 | 1 | 512 | 64 | 32 | 10000.0 | 2.5 | no | 8 | 4 | yes | 64 | 13.71B |
| [`tiny`](models/glm52/tiny_16x4K.md) | 6 | 128 | 512 | 128 | 0.0001 | 0.001 | 256 | 32 | 32 | 1 | 32 | 8 | 24 | 3xfull/3xshared | 32 | 8 | 4 | 1 | 64 | 16 | 8 | 8000000.0 | 2.5 | yes | 2 | 2 | yes | 16 | 1.1M |

Notes:
- Aliases share the exact architecture shape of an earlier preset
  (e.g. Kimi K2.5/2.6/2.7 are shape-identical to K2; GLM 5.1 to 5).
- `bench_train`/`bench_frontier` config names compose as
  `{preset-prefix}-s{seq}k-bs{B}ga{G}` — see docs/benchmarking.md.
- Correctness: `python tools/verify_family.py --family <name>`.

