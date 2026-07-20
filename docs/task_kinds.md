# Task kinds: compute keys and executables

GENERATED — regenerate with `python tools/gen_model_docs/list_tasks.py >
docs/task_kinds.md` after adding a family or layer kind. Dispatch is
the resolver ABI (`resolver(task) -> executable.launch(ctx)`, keyed on
`task.compute_block_key`) — family-scoped by design; this table is the
fleet inventory. Buffer contracts (positional input/output order per
key) are documented next to each executable class; recompute keys are
derived from the forward stage lists, never hand-written
(docs/extending.md §2).

| family | compute key | group | executable | description |
|---|---|---|---|---|
| gpt2 | `block_bwd` | backward | `Gpt2BlockBwd` | MLP tail (GELU) then attention backward; the two LayerNorms |
| gpt2 | `block_fwd` | forward | `Gpt2BlockFwd` | Transformer-block forward: runs the STAGES list, writing saved |
| gpt2 | `block_recompute` | recompute (planner/derived) | `Gpt2BlockRecompute` | Derived recompute: replays the forward stages through the last |
| gpt2 | `embed_bwd` | backward | `Gpt2EmbedBwd` | Deterministic scatter of dy into BOTH tables: token rows (wte) and |
| gpt2 | `embed_fwd` | forward | `Gpt2EmbedFwd` | Two-table embedding: wte gather by token id PLUS wpe gather by |
| gpt2 | `head_loss` | backward | `Gpt2HeadLoss` | The fused final-norm + head + CE task with the final norm as |
| gpt2 | `optimizer_block` | optimizer | `AdamWStep` | Per-FIELD optimizer step over one packed weight object — |
| gpt2 | `optimizer_embed` | optimizer | `AdamWStep` | Per-FIELD optimizer step over one packed weight object — |
| gpt2 | `optimizer_head` | optimizer | `AdamWStep` | Per-FIELD optimizer step over one packed weight object — |
| llama3 | `block_bwd` | backward | `BlockBwd` | Transformer-block backward: MLP-tail then attention backward, per |
| llama3 | `block_fwd` | forward | `BlockFwd` | Transformer-block forward: runs the STAGES list, writing saved |
| llama3 | `block_recompute` | recompute (planner/derived) | `BlockRecompute` | Derived recompute: replays the forward stages through the last |
| llama3 | `embed_bwd` | backward | `EmbedBwd` | Embedding backward: deterministic per-row scatter of dy into |
| llama3 | `embed_fwd` | forward | `EmbedFwd` | Token-embedding lookup: tokens + W_embed -> the first hidden |
| llama3 | `head_loss` | backward | `HeadLoss` | Fused final-norm + LM head + CE loss + head backward, micro-chunked |
| llama3 | `optimizer_block` | optimizer | `AdamWStep` | Per-FIELD optimizer step over one packed weight object — |
| llama3 | `optimizer_embed` | optimizer | `AdamWStep` | Per-FIELD optimizer step over one packed weight object — |
| llama3 | `optimizer_head` | optimizer | `AdamWStep` | Per-FIELD optimizer step over one packed weight object — |
| qwen3 | `block_bwd` | backward | `Qwen3BlockBwd` | Transformer-block backward: MLP-tail then attention backward, per |
| qwen3 | `block_fwd` | forward | `Qwen3BlockFwd` | Transformer-block forward: runs the STAGES list, writing saved |
| qwen3 | `block_recompute` | recompute (planner/derived) | `Qwen3BlockRecompute` | Derived recompute: replays the forward stages through the last |
| qwen3 | `embed_bwd` | backward | `EmbedBwd` | Embedding backward: deterministic per-row scatter of dy into |
| qwen3 | `embed_fwd` | forward | `EmbedFwd` | Token-embedding lookup: tokens + W_embed -> the first hidden |
| qwen3 | `head_loss` | backward | `HeadLoss` | Fused final-norm + LM head + CE loss + head backward, micro-chunked |
| qwen3 | `optimizer_block` | optimizer | `AdamWStep` | Per-FIELD optimizer step over one packed weight object — |
| qwen3 | `optimizer_embed` | optimizer | `AdamWStep` | Per-FIELD optimizer step over one packed weight object — |
| qwen3 | `optimizer_head` | optimizer | `AdamWStep` | Per-FIELD optimizer step over one packed weight object — |
| qwen35 | `embed_bwd` | backward | `EmbedBwd` | Embedding backward: deterministic per-row scatter of dy into |
| qwen35 | `embed_fwd` | forward | `EmbedFwd` | Token-embedding lookup: tokens + W_embed -> the first hidden |
| qwen35 | `gattn_bwd` | backward | `Qwen35AttnBlockBwd` | Transformer-block backward: MLP-tail then attention backward, per |
| qwen35 | `gattn_fwd` | forward | `Qwen35AttnBlockFwd` | Transformer-block forward: runs the STAGES list, writing saved |
| qwen35 | `gattn_recompute` | recompute (planner/derived) | `Qwen35AttnBlockRecompute` | Derived recompute: replays the forward stages through the last |
| qwen35 | `head_loss` | backward | `HeadLoss` | Fused final-norm + LM head + CE loss + head backward, micro-chunked |
| qwen35 | `linattn_bwd` | backward | `Qwen35LinBlockBwd` | Transformer-block backward: MLP-tail then attention backward, per |
| qwen35 | `linattn_fwd` | forward | `Qwen35LinBlockFwd` | Transformer-block forward: runs the STAGES list, writing saved |
| qwen35 | `linattn_recompute` | recompute (planner/derived) | `Qwen35LinBlockRecompute` | Derived recompute: replays the forward stages through the last |
| qwen35 | `optimizer_block` | optimizer | `AdamWStep` | Per-FIELD optimizer step over one packed weight object — |
| qwen35 | `optimizer_embed` | optimizer | `AdamWStep` | Per-FIELD optimizer step over one packed weight object — |
| qwen35 | `optimizer_head` | optimizer | `AdamWStep` | Per-FIELD optimizer step over one packed weight object — |
| olmoe | `embed_bwd` | backward | `EmbedBwd` | Embedding backward: deterministic per-row scatter of dy into |
| olmoe | `embed_fwd` | forward | `EmbedFwd` | Token-embedding lookup: tokens + W_embed -> the first hidden |
| olmoe | `head_loss` | backward | `HeadLoss` | Fused final-norm + LM head + CE loss + head backward, micro-chunked |
| olmoe | `moeattn_bwd` | backward | `OlmoeBlockBwd` | Metadata-object plumbing for pure-MoE families: the layer's M |
| olmoe | `moeattn_fwd` | forward | `OlmoeBlockFwd` | Metadata-object plumbing for pure-MoE families: the layer's M |
| olmoe | `moeattn_recompute` | recompute (planner/derived) | `OlmoeBlockRecompute` | Metadata-object plumbing for pure-MoE families: the layer's M |
| olmoe | `optimizer_block` | optimizer | `AdamWStep` | Per-FIELD optimizer step over one packed weight object — |
| olmoe | `optimizer_embed` | optimizer | `AdamWStep` | Per-FIELD optimizer step over one packed weight object — |
| olmoe | `optimizer_head` | optimizer | `AdamWStep` | Per-FIELD optimizer step over one packed weight object — |
| olmoe | `prologue_round` | forward | `RoundPrologue` | The round-boundary task: publishes the CURRENT ROUND both as an |
| qwen35moe | `embed_bwd` | backward | `EmbedBwd` | Embedding backward: deterministic per-row scatter of dy into |
| qwen35moe | `embed_fwd` | forward | `EmbedFwd` | Token-embedding lookup: tokens + W_embed -> the first hidden |
| qwen35moe | `gattnmoe_bwd` | backward | `Qwen35MoeAttnBlockBwd` | Metadata-object plumbing for pure-MoE families: the layer's M |
| qwen35moe | `gattnmoe_fwd` | forward | `Qwen35MoeAttnBlockFwd` | Metadata-object plumbing for pure-MoE families: the layer's M |
| qwen35moe | `gattnmoe_recompute` | recompute (planner/derived) | `Qwen35MoeAttnBlockRecompute` | Metadata-object plumbing for pure-MoE families: the layer's M |
| qwen35moe | `head_loss` | backward | `HeadLoss` | Fused final-norm + LM head + CE loss + head backward, micro-chunked |
| qwen35moe | `linmoe_bwd` | backward | `Qwen35MoeLinBlockBwd` | Metadata-object plumbing for pure-MoE families: the layer's M |
| qwen35moe | `linmoe_fwd` | forward | `Qwen35MoeLinBlockFwd` | Metadata-object plumbing for pure-MoE families: the layer's M |
| qwen35moe | `linmoe_recompute` | recompute (planner/derived) | `Qwen35MoeLinBlockRecompute` | Metadata-object plumbing for pure-MoE families: the layer's M |
| qwen35moe | `optimizer_block` | optimizer | `AdamWStep` | Per-FIELD optimizer step over one packed weight object — |
| qwen35moe | `optimizer_embed` | optimizer | `AdamWStep` | Per-FIELD optimizer step over one packed weight object — |
| qwen35moe | `optimizer_head` | optimizer | `AdamWStep` | Per-FIELD optimizer step over one packed weight object — |
| qwen35moe | `prologue_round` | forward | `RoundPrologue` | The round-boundary task: publishes the CURRENT ROUND both as an |
| qwen3moe | `embed_bwd` | backward | `EmbedBwd` | Embedding backward: deterministic per-row scatter of dy into |
| qwen3moe | `embed_fwd` | forward | `EmbedFwd` | Token-embedding lookup: tokens + W_embed -> the first hidden |
| qwen3moe | `head_loss` | backward | `HeadLoss` | Fused final-norm + LM head + CE loss + head backward, micro-chunked |
| qwen3moe | `optimizer_block` | optimizer | `AdamWStep` | Per-FIELD optimizer step over one packed weight object — |
| qwen3moe | `optimizer_embed` | optimizer | `AdamWStep` | Per-FIELD optimizer step over one packed weight object — |
| qwen3moe | `optimizer_head` | optimizer | `AdamWStep` | Per-FIELD optimizer step over one packed weight object — |
| qwen3moe | `prologue_round` | forward | `RoundPrologue` | The round-boundary task: publishes the CURRENT ROUND both as an |
| qwen3moe | `q3moeattn_bwd` | backward | `Qwen3MoeBlockBwd` | Metadata-object plumbing for pure-MoE families: the layer's M |
| qwen3moe | `q3moeattn_fwd` | forward | `Qwen3MoeBlockFwd` | Metadata-object plumbing for pure-MoE families: the layer's M |
| qwen3moe | `q3moeattn_recompute` | recompute (planner/derived) | `Qwen3MoeBlockRecompute` | Metadata-object plumbing for pure-MoE families: the layer's M |
| dsv3 | `embed_bwd` | backward | `EmbedBwd` | Embedding backward: deterministic per-row scatter of dy into |
| dsv3 | `embed_fwd` | forward | `EmbedFwd` | Token-embedding lookup: tokens + W_embed -> the first hidden |
| dsv3 | `head_loss` | backward | `HeadLoss` | Fused final-norm + LM head + CE loss + head backward, micro-chunked |
| dsv3 | `mladense_bwd` | backward | `Dsv3DenseBlockBwd` | Transformer-block backward: MLP-tail then attention backward, per |
| dsv3 | `mladense_fwd` | forward | `Dsv3DenseBlockFwd` | Transformer-block forward: runs the STAGES list, writing saved |
| dsv3 | `mladense_recompute` | recompute (planner/derived) | `Dsv3DenseBlockRecompute` | Derived recompute: replays the forward stages through the last |
| dsv3 | `mlamoe_bwd` | backward | `Dsv3MoeBlockBwd` | Metadata-object plumbing for pure-MoE families: the layer's M |
| dsv3 | `mlamoe_fwd` | forward | `Dsv3MoeBlockFwd` | Metadata-object plumbing for pure-MoE families: the layer's M |
| dsv3 | `mlamoe_recompute` | recompute (planner/derived) | `Dsv3MoeBlockRecompute` | Metadata-object plumbing for pure-MoE families: the layer's M |
| dsv3 | `optimizer_block` | optimizer | `AdamWStep` | Per-FIELD optimizer step over one packed weight object — |
| dsv3 | `optimizer_embed` | optimizer | `AdamWStep` | Per-FIELD optimizer step over one packed weight object — |
| dsv3 | `optimizer_head` | optimizer | `AdamWStep` | Per-FIELD optimizer step over one packed weight object — |
| dsv3 | `prologue_round` | forward | `RoundPrologue` | The round-boundary task: publishes the CURRENT ROUND both as an |
| dsv32 | `dsadense_bwd` | backward | `Dsv32DenseBlockBwd` | Metadata-object plumbing (one implementation for fwd/rc/bwd): the |
| dsv32 | `dsadense_fwd` | forward | `Dsv32DenseBlockFwd` | Metadata-object plumbing (one implementation for fwd/rc/bwd): the |
| dsv32 | `dsadense_recompute` | recompute (planner/derived) | `Dsv32DenseBlockRecompute` | Metadata-object plumbing (one implementation for fwd/rc/bwd): the |
| dsv32 | `dsamoe_bwd` | backward | `Dsv32MoeBlockBwd` | Metadata-object plumbing (one implementation for fwd/rc/bwd): the |
| dsv32 | `dsamoe_fwd` | forward | `Dsv32MoeBlockFwd` | Metadata-object plumbing (one implementation for fwd/rc/bwd): the |
| dsv32 | `dsamoe_recompute` | recompute (planner/derived) | `Dsv32MoeBlockRecompute` | Metadata-object plumbing (one implementation for fwd/rc/bwd): the |
| dsv32 | `embed_bwd` | backward | `EmbedBwd` | Embedding backward: deterministic per-row scatter of dy into |
| dsv32 | `embed_fwd` | forward | `EmbedFwd` | Token-embedding lookup: tokens + W_embed -> the first hidden |
| dsv32 | `head_loss` | backward | `HeadLoss` | Fused final-norm + LM head + CE loss + head backward, micro-chunked |
| dsv32 | `optimizer_block` | optimizer | `AdamWStep` | Per-FIELD optimizer step over one packed weight object — |
| dsv32 | `optimizer_embed` | optimizer | `AdamWStep` | Per-FIELD optimizer step over one packed weight object — |
| dsv32 | `optimizer_head` | optimizer | `AdamWStep` | Per-FIELD optimizer step over one packed weight object — |
| dsv32 | `prologue_round` | forward | `RoundPrologue` | The round-boundary task: publishes the CURRENT ROUND both as an |
| glm52 | `embed_bwd` | backward | `EmbedBwd` | Embedding backward: deterministic per-row scatter of dy into |
| glm52 | `embed_fwd` | forward | `EmbedFwd` | Token-embedding lookup: tokens + W_embed -> the first hidden |
| glm52 | `gdl_bwd` | backward | `Glm52DlBlockBwd` | Leader backward KL: dI = sigma - (p_own + dM)/N when followers |
| glm52 | `gdl_fwd` | forward | `Glm52DlBlockFwd` | Own M + (followers) producer's M + (grouped bwds) dM |
| glm52 | `gdl_recompute` | recompute (planner/derived) | `Glm52DlBlockRecompute` | Own M + (followers) producer's M + (grouped bwds) dM |
| glm52 | `gmf_bwd` | backward | `Glm52MfBlockBwd` | Follower backward: sparse core bwd on the shared selection + the |
| glm52 | `gmf_fwd` | forward | `Glm52MfBlockFwd` | Follower forward: PLAIN dsv3 MLA stages (no indexer tap, no |
| glm52 | `gmf_recompute` | recompute (planner/derived) | `Glm52MfBlockRecompute` | Follower forward: PLAIN dsv3 MLA stages (no indexer tap, no |
| glm52 | `gml_bwd` | backward | `Glm52MlBlockBwd` | Leader backward KL: dI = sigma - (p_own + dM)/N when followers |
| glm52 | `gml_fwd` | forward | `Glm52MlBlockFwd` | Own M + (followers) producer's M + (grouped bwds) dM |
| glm52 | `gml_recompute` | recompute (planner/derived) | `Glm52MlBlockRecompute` | Own M + (followers) producer's M + (grouped bwds) dM |
| glm52 | `head_loss` | backward | `HeadLoss` | Fused final-norm + LM head + CE loss + head backward, micro-chunked |
| glm52 | `optimizer_block` | optimizer | `AdamWStep` | Per-FIELD optimizer step over one packed weight object — |
| glm52 | `optimizer_embed` | optimizer | `AdamWStep` | Per-FIELD optimizer step over one packed weight object — |
| glm52 | `optimizer_head` | optimizer | `AdamWStep` | Per-FIELD optimizer step over one packed weight object — |
| glm52 | `prologue_round` | forward | `RoundPrologue` | The round-boundary task: publishes the CURRENT ROUND both as an |
