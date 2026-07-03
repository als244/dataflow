"""Batch>1 + grad-accum correctness (GPU): the runtime under a tight budget
must match plain-PyTorch autograd when sequences are batched (causal masks
must not leak across sequences) and gradients accumulate across rounds
(summed, no scaling — matching the runtime's mutate-accumulate contract)."""
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from dataflow.models.llama3_reference import GoldenLlama3  # noqa: E402
from dataflow.runtime import Engine  # noqa: E402
from dataflow.runtime.device.cuda import CudaBackend  # noqa: E402
from dataflow.runtime.device.fake import FakeBackend  # noqa: E402
from dataflow.tasks.interop import torch_view  # noqa: E402
from dataflow.tasks.llama3_blocks import build_resolver  # noqa: E402
from dataflow.tasks import ops  # noqa: E402
from dataflow.training.llama3_lowering import dims_of, initial_values, lower_llama3  # noqa: E402
from dataflow.training.planning import plan_program  # noqa: E402
from dataflow.training.shaped_llama3 import ShapedLlamaConfig  # noqa: E402
from dataflow.training.testing.gradcheck import rel_l2  # noqa: E402

pytestmark = pytest.mark.gpu

CFG = ShapedLlamaConfig(
    n_layers=2, d_model=256, n_heads=8, n_kv_heads=2, d_ff=512,
    vocab_size=512, seq_len=128, batch=2, grad_accum_rounds=2,
)


def test_causal_mask_does_not_leak_across_batch():
    """Batched attention == per-sequence attention, concatenated."""
    s, d = 128, 256
    q = torch.randn(2 * s, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2 * s, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(2 * s, 64, device="cuda", dtype=torch.bfloat16)
    batched, _ = ops.flash_fwd(q, k, v, 8, 2, 32, seq_len=s)
    per0, _ = ops.flash_fwd(q[:s], k[:s], v[:s], 8, 2, 32, seq_len=s)
    per1, _ = ops.flash_fwd(q[s:], k[s:], v[s:], 8, 2, 32, seq_len=s)
    assert torch.equal(batched[:s], per0)
    assert torch.equal(batched[s:], per1)


def test_batch_ga_model_step_matches_golden():
    dims = dims_of(CFG)
    program = lower_llama3(CFG)
    planned = plan_program(program, fast_memory_capacity=10 * 1024 * 1024)  # tight

    backend = CudaBackend()
    values = initial_values(planned.program, CFG, backend, seed=3)

    def pinned(name):
        buf = values[name]
        return torch_view(buf, (buf.size_bytes,), torch.uint8).clone()

    golden = GoldenLlama3.from_packed_bytes(
        dims, CFG.n_layers,
        pinned("W_embed"),
        [pinned(f"W_{i}") for i in range(CFG.n_layers)],
        pinned("W_head"),
    )
    # grad accumulation sums per-round mean losses: backward once on the sum
    total = None
    for r in range(CFG.grad_accum_rounds):
        tokens = torch_view(values[f"tokens_0_{r}"], (dims.tokens,), torch.int32).long().cuda()
        targets = torch_view(values[f"targets_0_{r}"], (dims.tokens,), torch.int32).long().cuda()
        loss_r = golden.loss(tokens, targets)
        total = loss_r if total is None else total + loss_r
    total.backward()
    golden.step_count = 1
    golden._adamw_obj("embed", golden.w_embed)
    for i, leaves in enumerate(golden.w_blocks):
        golden._adamw_obj(f"block_{i}", leaves)
    golden._adamw_obj("head", golden.w_head)

    dry = Engine(FakeBackend()).execute(planned.program, initial_buffers=values)
    result = Engine(backend).execute(
        planned.program, resolver=build_resolver(dims),
        initial_buffers=values, pool_prewarm=dry.pool_demand,
    )

    def worst_field_err(object_id: str) -> float:
        from dataflow.tasks.interop import TORCH_DTYPE_BY_NAME

        rec = result.objects.get(object_id)
        slot = rec.backing or rec.fast
        layout, leaves = golden.final_leaves(object_id)
        return max(
            rel_l2(
                torch_view(slot.buffer, f.shape, TORCH_DTYPE_BY_NAME[f.dtype],
                           offset_bytes=f.offset_bytes),
                leaves[f.name],
            )
            for f in layout.fields
        )

    assert worst_field_err("W_embed") < 3e-2
    for i in range(CFG.n_layers):
        assert worst_field_err(f"W_{i}") < 3e-2, f"W_{i}"
    assert worst_field_err("W_head") < 3e-2
    result.close()
