"""M4 core semantics (GPU): multi-step training with state carryover through
pinned buffers + session pool reuse, verified against the golden model
step-for-step."""
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from dataflow.models.llama3_reference import GoldenLlama3  # noqa: E402
from dataflow.runtime.device.cuda import CudaBackend  # noqa: E402
from dataflow.tasks.interop import torch_view  # noqa: E402
from dataflow.training.llama3_lowering import dims_of, lower_llama3  # noqa: E402
from dataflow.training.planning import plan_program  # noqa: E402
from dataflow.training.shaped_llama3 import ShapedLlamaConfig  # noqa: E402
from dataflow.training.testing.gradcheck import rel_l2  # noqa: E402
from dataflow.training.train_loop import train  # noqa: E402

pytestmark = pytest.mark.gpu

CFG = ShapedLlamaConfig(
    n_layers=2, d_model=256, n_heads=8, n_kv_heads=2, d_ff=512,
    vocab_size=512, seq_len=128, batch=1,
)
STEPS = 3


def test_multistep_matches_golden_and_loss_decreases():
    dims = dims_of(CFG)
    program = lower_llama3(CFG)
    planned = plan_program(program, fast_memory_capacity=8 * 1024 * 1024)
    backend = CudaBackend()

    # fixed token stream shared with the golden model
    gen = torch.Generator().manual_seed(99)
    one_batch = (
        torch.randint(0, dims.vocab_size, (dims.tokens,), generator=gen, dtype=torch.int32),
        torch.randint(0, dims.vocab_size, (dims.tokens,), generator=gen, dtype=torch.int32),
    )
    # the same batch every step: memorization must drive the loss down,
    # which also proves optimizer state (m, v) carries across steps
    batches = [one_batch] * STEPS

    # golden N-step trajectory from the same initial bytes: rebuild the
    # initial pinned values in isolation to snapshot them first
    from dataflow.training.llama3_lowering import initial_values

    snapshot_values = initial_values(planned.program, CFG, backend, seed=5)

    def pinned(name):
        buf = snapshot_values[name]
        return torch_view(buf, (buf.size_bytes,), torch.uint8).clone()

    golden = GoldenLlama3.from_packed_bytes(
        dims, CFG.n_layers,
        pinned("W_embed"),
        [pinned(f"W_{i}") for i in range(CFG.n_layers)],
        pinned("W_head"),
    )
    golden_losses = [
        golden.train_step(t.long().cuda(), g.long().cuda()) for t, g in batches
    ]

    report = train(
        planned.program, CFG, backend,
        steps=STEPS, seed=5, token_stream=lambda s: batches[s],
    )

    # same loss trajectory
    for ours, ref in zip(report.losses, golden_losses):
        assert abs(ours - ref) / max(abs(ref), 1e-9) < 3e-2, (report.losses, golden_losses)
    # training trains
    assert report.losses[-1] < report.losses[0]
    # steady state must be overflow-free (step 0 may overflow at this
    # deliberately tight budget; those buffers join the pool and get reused)
    assert all(n == 0 for n in report.step_slab_overflows[1:]), report.step_slab_overflows
    # multi-step invariance held (final state persisted through pinned buffers)
    assert report.peak_fast_bytes <= 8 * 1024 * 1024


def test_dynamic_mode_matches_static():
    """placement_mode is an independent optimization: the dynamic slab path
    (required for variable-length programs) must produce the same training
    trajectory as static placement.

    Equivalence is TOLERANCE-based, not bitwise: buffer addresses differ
    between allocators, and cuBLASLt selects GEMM algorithms by pointer
    alignment — different bf16 reduction orders shift results ~1e-6 rel.
    Within one allocator the run IS bitwise repeatable (determinism gate)."""
    dims = dims_of(CFG)
    planned = plan_program(lower_llama3(CFG), fast_memory_capacity=8 * 1024 * 1024)
    backend = CudaBackend()

    gen = torch.Generator().manual_seed(41)
    batch = (
        torch.randint(0, dims.vocab_size, (dims.tokens,), generator=gen, dtype=torch.int32),
        torch.randint(0, dims.vocab_size, (dims.tokens,), generator=gen, dtype=torch.int32),
    )
    losses = {}
    for mode in ("static", "dynamic"):
        report = train(
            planned.program, CFG, backend, steps=2, seed=13,
            token_stream=lambda s: batch, placement_mode=mode,
        )
        losses[mode] = report.losses
        if mode == "static":
            assert report.placement_extent_bytes > 0
        else:
            assert report.placement_extent_bytes == 0  # knob honored
    for a, b in zip(losses["static"], losses["dynamic"]):
        assert abs(a - b) / max(abs(b), 1e-9) < 1e-4, losses


