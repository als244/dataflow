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

| family | preset | layers | d_model | vocab | seq default | params |
|---|---|---|---|---|---|---|
| llama3 | `llama3_8b` | 32 | 4096 | 128256 | 4096 | 8.03B |
| llama3 | `tiny` | 2 | 64 | 512 | 64 | 152K |
| qwen3 | `qwen3_8b` | 36 | 4096 | 151936 | 4096 | 8.19B |
| qwen3 | `tiny` | 2 | 256 | 512 | 64 | 1.4M |
| qwen35 | `qwen35_9b` | 32 | 4096 | 248320 | 4096 | 8.95B |
| qwen35 | `tiny` | 4 | 256 | 512 | 128 | 2.5M |
| qwen35 | `tiny_tied` | 4 | 256 | 512 | 128 | 2.4M |
| olmoe | `olmoe_7b` | 16 | 2048 | 50304 | 4096 | 6.92B |
| olmoe | `tiny` | 2 | 128 | 512 | 128 | 1.1M |
| qwen35moe | `qwen35moe_20l` | 20 | 2048 | 248320 | 4096 | 17.84B |
| qwen35moe | `qwen35moe_35b` | 40 | 2048 | 248320 | 4096 | 34.66B |
| qwen35moe | `tiny` | 4 | 256 | 512 | 128 | 4.5M |
| qwen3moe | `qwen3moe_235b` | 94 | 4096 | 151936 | 4096 | 235.09B |
| qwen3moe | `qwen3moe_30b` | 48 | 2048 | 151936 | 4096 | 30.53B |
| qwen3moe | `qwen3moe_30b_24l` | 24 | 2048 | 151936 | 4096 | 15.58B |
| qwen3moe | `tiny` | 2 | 128 | 512 | 128 | 626K |
| dsv3 | `dsv3_671b` | 61 | 7168 | 129280 | 4096 | 671.03B |
| dsv3 | `dsv3_mini` | 18 | 2048 | 129280 | 4096 | 13.71B |
| dsv3 | `kimi_k2` | 61 | 7168 | 163840 | 4096 | 1.026T |
| dsv3 | `kimi_k25` (alias of `kimi_k2`) | 61 | 7168 | 163840 | 4096 | 1.026T |
| dsv3 | `kimi_k26` (alias of `kimi_k2`) | 61 | 7168 | 163840 | 4096 | 1.026T |
| dsv3 | `kimi_k27` (alias of `kimi_k2`) | 61 | 7168 | 163840 | 4096 | 1.026T |
| dsv3 | `tiny` | 3 | 128 | 512 | 128 | 550K |
| dsv32 | `dsv32_671b` | 61 | 7168 | 129280 | 4096 | 671.91B |
| dsv32 | `dsv32_mini` | 18 | 2048 | 129280 | 4096 | 13.72B |
| dsv32 | `glm5` | 78 | 6144 | 154880 | 4096 | 743.93B |
| dsv32 | `glm51` (alias of `glm5`) | 78 | 6144 | 154880 | 4096 | 743.93B |
| dsv32 | `tiny` | 3 | 128 | 512 | 128 | 618K |
| glm52 | `glm52` | 78 | 6144 | 154880 | 4096 | 743.38B |
| glm52 | `glm52_mini` | 18 | 2048 | 129280 | 4096 | 13.71B |
| glm52 | `tiny` | 6 | 128 | 512 | 128 | 1.1M |

Notes:
- Aliases share the exact architecture shape of an earlier preset
  (e.g. Kimi K2.5/2.6/2.7 are shape-identical to K2; GLM 5.1 to 5).
- `bench_train`/`bench_frontier` config names compose as
  `{preset-prefix}-s{seq}k-bs{B}ga{G}` — see docs/benchmarking.md.
- Correctness: `python tools/verify_family.py --family <name>`.

