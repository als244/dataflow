# Model references — one page per (family, preset)

GENERATED — `python tools/gen_model_docs.py` regenerates everything (`--family X [--preset P]` narrows; `--no-record` skips kernel tracing on CPU-only machines). Default pages use the standard documentation run shape (microbatch 16 × seq 4096 — the `_16x4K` filename suffix); pages at other shapes: `tools/gen_model_page.py`. New families — builtin or plugin — appear automatically.

- **llama3**: [llama3_8b](llama3/llama3_8b_16x4K.md) · [tiny](llama3/tiny_16x4K.md)
- **qwen3**: [qwen3_8b](qwen3/qwen3_8b_16x4K.md) · [tiny](qwen3/tiny_16x4K.md)
- **qwen35**: [qwen35_9b](qwen35/qwen35_9b_16x4K.md) · [tiny](qwen35/tiny_16x4K.md) · [tiny_tied](qwen35/tiny_tied_16x4K.md)
- **olmoe**: [olmoe_7b](olmoe/olmoe_7b_16x4K.md) · [tiny](olmoe/tiny_16x4K.md)
- **qwen35moe**: [qwen35moe_20l](qwen35moe/qwen35moe_20l_16x4K.md) · [qwen35moe_35b](qwen35moe/qwen35moe_35b_16x4K.md) · [tiny](qwen35moe/tiny_16x4K.md)
- **qwen3moe**: [qwen3moe_235b](qwen3moe/qwen3moe_235b_16x4K.md) · [qwen3moe_30b](qwen3moe/qwen3moe_30b_16x4K.md) · [qwen3moe_30b_24l](qwen3moe/qwen3moe_30b_24l_16x4K.md) · [tiny](qwen3moe/tiny_16x4K.md)
- **dsv3**: [dsv3_671b](dsv3/dsv3_671b_16x4K.md) · [dsv3_mini](dsv3/dsv3_mini_16x4K.md) · [kimi_k2](dsv3/kimi_k2_16x4K.md) · [kimi_k25](dsv3/kimi_k25_16x4K.md) · [kimi_k26](dsv3/kimi_k26_16x4K.md) · [kimi_k27](dsv3/kimi_k27_16x4K.md) · [tiny](dsv3/tiny_16x4K.md)
- **dsv32**: [dsv32_671b](dsv32/dsv32_671b_16x4K.md) · [dsv32_mini](dsv32/dsv32_mini_16x4K.md) · [dsv32_mini_warmup](dsv32/dsv32_mini_warmup_16x4K.md) · [glm5](dsv32/glm5_16x4K.md) · [glm51](dsv32/glm51_16x4K.md) · [tiny](dsv32/tiny_16x4K.md)
- **glm52**: [glm52](glm52/glm52_16x4K.md) · [glm52_mini](glm52/glm52_mini_16x4K.md) · [glm52_mini_warmup](glm52/glm52_mini_warmup_16x4K.md) · [tiny](glm52/tiny_16x4K.md)
