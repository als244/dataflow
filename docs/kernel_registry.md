# Kernel registry

GENERATED — regenerate with `python tools/list_kernels.py >
docs/kernel_registry.md` after registering ops or implementations.
The registry (`dataflow_training/kernels/registry.py`) selects, per op, the
highest-priority implementation whose `requires(caps)` passes on this
machine; `DATAFLOW_KERNELS=eager` forces the priority-0 fallbacks for
bisection. The chosen set is stamped into profiles (measured costs are
measurements of a SPECIFIC kernel set). Contract for what
implementations may do: docs/task-contract.md; adding ops:
docs/extending.md §1.

Column notes: *resolved* = the impl selected on the machine that
generated this doc; *ws* = declared workspace (none / internal
estimate); *alloc* = allocator discipline (`none` = no allocations in
the launch path, `torch` = op-internal torch scratch, measured by
profiling).

| op | impls (priority) | resolved | det | ws | alloc | signature | description |
|---|---|---|---|---|---|---|---|
| `adamw_step` | triton(10), eager(0) | triton | yes | none | none | `(kctx, w, g, m, v, *, lr, beta1, beta2, eps, weight_decay, step)` | AdamW step: fused Triton (default) + eager fallback |
| `causal_conv1d_silu_bwd` | fla-triton(10), eager(0) | fla-triton | yes | internal | torch | `(kctx, x, dy, w, dx_out, dw_out, cu_seqlens)` | causal_conv1d_silu family: depthwise causal conv1d + silu (the DeltaNet |
| `causal_conv1d_silu_fwd` | fla-triton(10), eager(0) | fla-triton | yes | internal | torch | `(kctx, x, w, out, cu_seqlens)` | causal_conv1d_silu family: depthwise causal conv1d + silu (the DeltaNet |
| `ce_loss_fwd_bwd` | triton(10), eager(0) | triton | yes | internal | torch | `(kctx, logits, targets, loss, dlogits, total_rows=None)` | Fused cross-entropy loss fwd+bwd: Triton (default) + eager fallback |
| `dsa_index_bwd` | triton(10), eager(0) | triton | yes | internal | torch | `(kctx, d_scores, q_idx, k_idx, wts, dq_out, dk_out, dwts_out, *, n_heads, head_dim, seq_bounds)` | DSA (DeepSeek-V3.2) sparse-attention kernel family — eager v1 |
| `dsa_index_scores` | triton(10), eager(0) | triton | yes | internal | torch | `(kctx, q_idx, k_idx, wts, scores_out, *, n_heads, head_dim, seq_bounds)` | DSA (DeepSeek-V3.2) sparse-attention kernel family — eager v1 |
| `dsa_pack_bits` | eager(0) | eager | yes | internal | torch | `(kctx, idx, bits_out, *, seq_bounds)` | OR-correct bit packer: one-hot bool (R, L) then 64-bit pack via |
| `dsa_probs_sum` | triton(10), eager(0) | triton | yes | internal | torch | `(kctx, q, kf, idx, lse, p_out, *, n_heads, head_dim, seq_bounds, bits_by_seq=None)` | DSA (DeepSeek-V3.2) sparse-attention kernel family — eager v1 |
| `dsa_sparse_attn_bwd` | triton(10), eager(0) | triton | yes | internal | torch | `(kctx, d_attn, q, kf, vp, idx, lse, dq_out, dk_out, dv_out, *, n_heads, head_dim, seq_bounds, out...` | DSA (DeepSeek-V3.2) sparse-attention kernel family — eager v1 |
| `dsa_sparse_attn_fwd` | triton(10), eager(0) | triton | yes | internal | torch | `(kctx, q, kf, vp, idx, out, lse_out, *, n_heads, head_dim, seq_bounds, bits_by_seq=None, v_head_d...` | DSA (DeepSeek-V3.2) sparse-attention kernel family — eager v1 |
| `dsa_sparse_attn_fwd_absorbed` | flashmla(20), eager(0) | eager | yes | internal | vendor | `(kctx, q_abs, kv, idx, out, lse_out, *, n_heads, d_qk, d_v, seq_bounds)` | Eager anchor for the absorbed layout (any dims, any device): |
| `dsa_topk` | eager(0) | eager | yes | internal | torch | `(kctx, scores, idx_out)` | DSA (DeepSeek-V3.2) sparse-attention kernel family — eager v1 |
| `embed_bwd_accum` | triton(10), eager(0) | triton | yes | internal | torch | `(kctx, tokens, dy, dw_embed, *, zero_first)` | Deterministic embedding-gradient accumulation |
| `gated_rmsnorm_bwd` | fla-fused(10), eager(0) | fla-fused | yes | internal | torch | `(kctx, dy, o, z, w, rstd, do_out, dz_out, dw_out, y_out)` | gated_rmsnorm family: silu(z) * rmsnorm(o) * w over lin_v_head_dim rows |
| `gated_rmsnorm_fwd` | fla-fused(10), eager(0) | fla-fused | yes | internal | torch | `(kctx, o, z, w, out, rstd_out)` | gated_rmsnorm family: silu(z) * rmsnorm(o) * w over lin_v_head_dim rows |
| `moe_aux_lb_grad` | triton(10), eager(0) | triton | yes | none | none | `(kctx, logits, counts, dlogits, *, alpha, top_k)` | MoE router ops: fused top-k+softmax, router backward, aux-loss gradient |
| `moe_combine_fwd` | triton(10), eager(0) | triton | yes | none | none | `(kctx, yp, slot_of, route_w, resid, out)` | MoE dispatch/combine ops — the expert-parallelism seam |
| `moe_dispatch_bwd` | triton(10), eager(0) | triton | yes | none | none | `(kctx, dxp, slot_of, out)` | MoE dispatch/combine ops — the expert-parallelism seam |
| `moe_dispatch_fwd` | aten(10) | aten | yes | internal | torch | `(kctx, x, order, out, *, top_k)` | MoE dispatch/combine ops — the expert-parallelism seam |
| `moe_grouped_mm_dgrad` | triton(20), aten-grouped(5), eager(0) | triton | yes | internal | torch | `(kctx, dy, w, offsets, dx_out=None)` | Grouped GEMM over expert-contiguous row segments (the MoE experts stage) |
| `moe_grouped_mm_fwd` | triton(20), aten-grouped(5), eager(0) | triton | yes | internal | torch | `(kctx, x, w, offsets, out=None)` | Grouped GEMM over expert-contiguous row segments (the MoE experts stage) |
| `moe_grouped_mm_wgrad` | triton(20), aten-grouped(5), eager(0) | triton | yes | internal | torch | `(kctx, x, dy, offsets, dw, *, accumulate: 'bool')` | Grouped GEMM over expert-contiguous row segments (the MoE experts stage) |
| `moe_router_bwd` | triton(10), eager(0) | triton | yes | none | none | `(kctx, dprob, route_w, route_ids, logits, dlogits_out, *, mode)` | MoE router ops: fused top-k+softmax, router backward, aux-loss gradient |
| `moe_router_bwd_sigmoid` | eager(0) | eager | yes | internal | torch | `(kctx, dprob, route_w, route_ids, logits, dlogits_out)` | w_j = c*s_j/S (c = routed_scaling = sum_j w_j, S = sum_j s_j over |
| `moe_rowdot` | triton(10), eager(0) | triton | yes | none | none | `(kctx, a, b, out)` | MoE dispatch/combine ops — the expert-parallelism seam |
| `moe_scale_rows` | triton(10), eager(0) | triton | yes | none | none | `(kctx, x, srw)` | MoE dispatch/combine ops — the expert-parallelism seam |
| `moe_seq_aux_grad` | eager(0) | eager | yes | internal | torch | `(kctx, logits, route_ids, dlogits, *, alpha, top_k, seq_bounds)` | DeepSeek-V3 sequence-wise complementary aux, injected analytically: |
| `moe_sort` | aten(10) | aten | yes | internal | torch | `(kctx, route_ids, order_out, offsets_out, *, n_experts)` | MoE dispatch/combine ops — the expert-parallelism seam |
| `moe_topk_sigmoid_noaux` | eager(0) | eager | yes | internal | torch | `(kctx, logits, bias, route_w_out, route_ids_out, *, top_k, n_group, topk_group, routed_scaling)` | MoE router ops: fused top-k+softmax, router backward, aux-loss gradient |
| `moe_topk_softmax` | triton(10), eager(0) | triton | yes | none | none | `(kctx, logits, route_w_out, route_ids_out, *, top_k, mode)` | MoE router ops: fused top-k+softmax, router backward, aux-loss gradient |
| `muon_step` | aten(10) | aten | yes | none | none | `(kctx, w, g, m, *, shape, lr, beta, eps, weight_decay)` | muon_step: Nesterov momentum + quintic Newton-Schulz orthogonalized |
| `rmsnorm_apply` | triton(10), eager(0) | triton | yes | none | none | `(kctx, x, rstd, w, out)` | rmsnorm family: fused Triton (default) + eager fallback |
| `rmsnorm_bwd` | triton(10), eager(0) | triton | yes | internal | torch | `(kctx, dy, x, rstd, w, dx_out, dw_out)` | rmsnorm family: fused Triton (default) + eager fallback |
| `rmsnorm_fwd` | triton(10), eager(0) | triton | yes | none | none | `(kctx, x, w, out, rstd_out)` | rmsnorm family: fused Triton (default) + eager fallback |
| `rmsnorm_noweight` | triton(10), eager(0) | triton | yes | none | none | `(kctx, x, out, rstd_out)` | rmsnorm family: fused Triton (default) + eager fallback |
| `rope_bwd` | triton(10), eager(0) | triton | yes | none | none | `(kctx, dx, out, pos, h, hd, base, **kw)` | rope (llama rotate-half): fused Triton (default) + eager fallback |
| `rope_fwd` | triton(10), eager(0) | triton | yes | none | none | `(kctx, x, out, pos, h, hd, base, **kw)` | rope (llama rotate-half): fused Triton (default) + eager fallback |
| `swiglu_bwd` | triton(10), eager(0) | triton | yes | none | none | `(kctx, ds, x1, x3, dx1, dx3)` | swiglu forward/backward: fused Triton (default) + eager fallback |
| `swiglu_fwd_out` | triton(10), eager(0) | triton | yes | none | none | `(kctx, x1, x3, out)` | swiglu forward/backward: fused Triton (default) + eager fallback |
| `swiglu_packed_bwd` | triton(10), eager(0) | triton | yes | none | none | `(kctx, ds, h13, dh13)` | swiglu forward/backward: fused Triton (default) + eager fallback |
| `swiglu_packed_fwd` | triton(10), eager(0) | triton | yes | none | none | `(kctx, h13, out)` | swiglu forward/backward: fused Triton (default) + eager fallback |
