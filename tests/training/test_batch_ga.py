"""Batch>1 + grad-accum correctness (GPU): the runtime under a tight budget
must match plain-PyTorch autograd when sequences are batched (causal masks
must not leak across sequences) and gradients accumulate across rounds
(summed, no scaling — matching the runtime's mutate-accumulate contract)."""
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from dataflow.runtime import Engine  # noqa: E402
from dataflow.runtime.device.cuda import CudaBackend  # noqa: E402
from dataflow.runtime.device.fake import FakeBackend  # noqa: E402
from dataflow.tasks.interop import torch_view  # noqa: E402
from dataflow.tasks.models.llama3_blocks import build_resolver  # noqa: E402
from dataflow.tasks import ops  # noqa: E402
from dataflow.training.models.llama3 import dims_of, initial_values, lower_llama3  # noqa: E402
from dataflow.training.planning import plan_program  # noqa: E402
from dataflow.training.models.llama3 import ShapedLlamaConfig  # noqa: E402
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
    seg2 = ops.Segments.uniform(s, 2).on(q.device)
    seg1 = ops.Segments.uniform(s, 1).on(q.device)
    batched, _ = ops.flash_fwd(q, k, v, 8, 2, 32, cu_seqlens=seg2.cu, max_seqlen=seg2.max_len)
    per0, _ = ops.flash_fwd(q[:s], k[:s], v[:s], 8, 2, 32, cu_seqlens=seg1.cu, max_seqlen=seg1.max_len)
    per1, _ = ops.flash_fwd(q[s:], k[s:], v[s:], 8, 2, 32, cu_seqlens=seg1.cu, max_seqlen=seg1.max_len)
    assert torch.equal(batched[:s], per0)
    assert torch.equal(batched[s:], per1)


def test_batch_ga_model_step_matches_reference():
    """Grad accumulation (2 rounds) through the engine == the isolated
    twin accumulating the same per-round losses — final params compared
    in twin-name space through the bridge."""
    from dataflow.pretrain import bridges
    from dataflow.pretrain.driver import adamw_field_step
    from dataflow.tasks.base_blocks import AdamWHyper
    from dataflow.training.testing.gradcheck import EngineFinalBytes

    dims = dims_of(CFG)
    program = lower_llama3(CFG)
    planned = plan_program(program, fast_memory_capacity=10 * 1024 * 1024)  # tight

    backend = CudaBackend()
    values = initial_values(planned.program, CFG, backend, seed=3)

    model = bridges.build_reference_model(CFG)
    bridges.load_reference_init(model, CFG, dims,
                                bridges.get_bytes_from_values(values))
    model.train()
    B = dims.tokens // CFG.seq_len
    total = None
    for r in range(CFG.grad_accum_rounds):
        tokens = torch_view(values[f"tokens_0_{r}"], (dims.tokens,),
                            torch.int32).long().cuda().view(B, CFG.seq_len)
        targets = torch_view(values[f"targets_0_{r}"], (dims.tokens,),
                             torch.int32).long().cuda().view(B, CFG.seq_len)
        loss_r = model.loss(tokens, targets)
        total = loss_r if total is None else total + loss_r
    total.backward()
    hp = AdamWHyper()
    for par in model.parameters():
        if par.grad is None:
            continue
        m = torch.zeros_like(par)
        v = torch.zeros_like(par)
        adamw_field_step(par.data, par.grad, m, v, lr=hp.lr,
                         beta1=hp.beta1, beta2=hp.beta2, eps=hp.eps,
                         weight_decay=hp.weight_decay, step=1)

    from dataflow.runtime.engine import uniform_segments

    dry = Engine(FakeBackend()).execute(planned.program, initial_buffers=values)
    result = Engine(backend).execute(
        planned.program, resolver=build_resolver(dims),
        initial_buffers=values, pool_prewarm=dry.pool_demand,
        run_args={"segments": uniform_segments(dims, planned.program)},
    )

    engine_state = bridges.to_reference_state_dict(
        CFG, EngineFinalBytes(result))
    twin_state = dict(model.state_dict())
    for name, engine_tensor in engine_state.items():
        err = rel_l2(engine_tensor, twin_state[name])
        assert err < 3e-2, (name, err)
    result.close()
