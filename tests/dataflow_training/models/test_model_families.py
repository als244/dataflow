"""Generic, family-parametrized correctness battery driven entirely
through a client to the shared out-of-process server.

Each registered family's smoke-scale preset is exercised through the same
checks that were previously copy-pasted per family: the program lowers,
validates and plans; one training step reproduces the pure-torch twin
(loss, parameters, gradients, MoE assignment counts) through the client;
two gradient-accumulation rounds match the twin's accumulated update; and
a battery of engine-invariance properties hold — a fixed seed is
bitwise-reproducible, the free-poisoning debug option changes nothing,
per-task runtime jitter changes nothing, and the result is identical
across two planning memory budgets and across a recompute re-plan.

Every server-using test takes the ``client`` fixture (conftest): ONE
out-of-process server is shared by the whole tests/dataflow_training run, and
the fixture wipes it to a blank store before each test, so each test sees a
fresh server without paying a boot. The free-poison test is the one exception
— poison-on-free is a boot flag, so it spawns its own poison-booted server
inline. Two guards prove the reset is clean: wiping and re-seeding restores the
pristine weights (engine), and a freshly-built twin is bitwise-reproducible
(reference).

Every engine value is read back as a host copy via the client
(``get_object`` / run ``fetch``); no in-process engine, device view, or
backend is constructed here. Tokens are generated deterministically
in-process and staged with ``put_object``; the grad-accumulation twin is
built from the server's own seeded init weights, fetched through the
client. This keeps the whole module on the memory-safe workload-test
transport.

Family-specific checks (sparse-attention warm-up and indexer training for
the MLA/DSA families, linear-attention kernels and tied or short-sequence
variants for the hybrid families) stay in the per-family modules.

Tests:
- test_lowering_validates_and_plans: each family's smoke preset lowers, validates, and plans to a positive-makespan schedule.
- test_golden_model_step: one client-run training step reproduces the pure-torch twin within the family's calibrated bands.
- test_golden_model_step_batch2_packed: the same golden comparison on a batch-of-two short-sequence packing.
- test_grad_accum_two_rounds: two gradient-accumulation rounds through the client leave final parameters matching the twin's accumulated update.
- test_fixed_seed_bitwise_deterministic: two same-seed client runs produce identical loss and weight bytes.
- test_poison_on_free_changes_nothing: booting the server with free-poisoning leaves loss and weights unchanged and non-NaN.
- test_result_invariant_to_runtime_jitter: injecting a per-task random launch delay leaves loss and weights unchanged.
- test_plan_invariance: planning the same step at two fast-memory budgets yields identical loss and weights.
- test_measured_costs_replan_still_golden: re-planning the step onto a recompute schedule yields identical loss and weights.
- test_reseed_restores_pristine_init: wiping the store and re-seeding restores the pristine weight bytes and a follow-on run reproduces an earlier run, so no trained engine state (weights, optimizer moments, aux counts) survives the per-test reset.
- test_reference_twin_build_is_stateless: two freshly-built twins, seeded and stepped identically, agree bit-for-bit, so the per-test reference twin carries no process-global state to leak into the next test.
"""
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no GPU", allow_module_level=True)

from dataflow.core import validate_program  # noqa: E402
from dataflow.core.jsonio import program_to_dict  # noqa: E402
from dataflow_training.blocks.base_blocks import AdamWHyper  # noqa: E402
from dataflow_training.lowering.planning import (  # noqa: E402
    plan_program,
    simulate_program,
)
from dataflow_training.model_families import bridges  # noqa: E402
from dataflow_training.model_families.families import resolve_family  # noqa: E402
from dataflow_training.run import presets as P  # noqa: E402
from dataflow_training.run.driver import adamw_field_step, init_model  # noqa: E402
from dataflow_training.run.presets import cfg_dict, resolver_family  # noqa: E402
from dataflow_training.testing.server_process import out_of_process_server  # noqa: E402
from dataflow_training.testing.client_parity import (  # noqa: E402
    ClientFinalBytes,
    adamw_hyper_spec,
    client_model_step,
)
from dataflow_training.testing.gradcheck import (  # noqa: E402
    family_gate_kwargs,
    match_field_atol,
    rel_l2,
)

pytestmark = pytest.mark.gpu

# The backing slab the free-poison test's own server pins (the shared server's
# is set in the conftest). Every smoke program's peak residency sits far under.
BACKING_GIB = 4.0

# family -> zero-arg smoke preset builder on ``run.presets`` (the same
# table the engine-vs-reference battery parametrizes over).
FAMILY_PRESETS = {
    "gpt2": "gpt2_smoke_preset",
    "llama3": "smoke_preset",
    "qwen3": "qwen3_smoke_preset",
    "qwen35": "qwen35_smoke_preset",
    "qwen35moe": "qwen35moe_smoke_preset",
    "olmoe": "olmoe_smoke_preset",
    "qwen3moe": "qwen3moe_smoke_preset",
    "dsv3": "dsv3_smoke_preset",
    "dsv32": "dsv32_smoke_preset",
    "glm52": "glm52_smoke_preset",
}

# Sign-lottery envelopes for the vs-twin comparisons. A zero-init bias
# whose one-step update is dominated by bf16-vs-AdamW sign flips on
# sub-noise gradients compares under an absolute band instead of a relative
# one (the same envelopes the per-family modules pin). The smoke presets run
# load-balancing OFF, so router-bias / indexer sign-lottery fields stay
# frozen and need no envelope.
#
# FIELD_ATOL entries also silence the gradient check for the field (its
# gradient is genuinely sub-noise); PARAM_ATOL entries gate only the
# parameter and keep the gradient check live (the sharp instrument).
FIELD_ATOL = {
    "qwen35": {"dt_bias": 2.5e-4},            # DeltaNet state-path bias
    "qwen35moe": {"dt_bias": 2.5e-4},
}
PARAM_ATOL = {
    "gpt2": {"bias": 3e-2},                    # zero-init attn/MLP biases
}


def envelope(name):
    """Merged absolute-tolerance table for the family's sign-lottery params
    (used where only the parameter is compared, e.g. accumulation)."""
    return {**FIELD_ATOL.get(name, {}), **PARAM_ATOL.get(name, {})}

SEED = 7
GA_TOL = 3e-2                          # final-param band, one accum step

# The near-tie routing bands in ``family_gate_kwargs`` are calibrated on each
# family's tiny config; a smoke preset routes more tokens per round, so the
# same benign near-tie mechanism draws a larger flip count AND a noisier
# routing-gradient direction (the per-family plan-invariance gates widen the
# flip budget for the same reason). A totals mismatch still fails hard (it
# scores infinite), and the loss / parameter / non-router gradients stay at
# the tight defaults — only the router gradient's near-tie band is widened.
SMOKE_FLIP_BUDGET = 12
ROUTER_GRAD_TOL = 0.5
ROUTER_MIN_COSINE = 0.85

# Two different plans compute the same math, but a bf16 engine only
# reproduces it bit-for-bit when the plan preserves reduction order. Moving
# bytes (offload) keeps the order; re-planning onto a recompute schedule
# reorders the order-sensitive MoE combine, so the cross-plan comparison is
# gated at an accumulation-order band, not bit-equality (the fixed-seed test
# is the bit-equality gate).
CROSS_PLAN_TOL = 2e-2

# Every smoke program's peak fast residency sits well under 300 MiB.
PLAN_BUDGET = 2 << 30                  # 2 GiB: everything resident, no spill
TIGHT_BUDGET = 160 << 20              # 160 MiB: below peak -> forces offload


def preset(name):
    return getattr(P, FAMILY_PRESETS[name])()


def gate_kwargs(name):
    """The family's calibrated gradient/counts gate, with the near-tie
    routing bands (flip budget + router-gradient direction/magnitude)
    widened for the busier smoke geometry. Non-MoE families carry no flip
    budget and keep their tight bands untouched."""
    kw = dict(family_gate_kwargs(name))
    if kw.get("counts_budget") is not None:
        kw["counts_budget"] = max(kw["counts_budget"], SMOKE_FLIP_BUDGET)
        kw["grad_tol"] = max(kw.get("grad_tol") or 0.0, ROUTER_GRAD_TOL)
        kw["min_cosine"] = min(kw.get("min_cosine") or 1.0, ROUTER_MIN_COSINE)
    return kw


def uniform_boundaries(dims):
    """Segment boundaries for a round of ``max_tokens`` packed as equal
    ``seq_len`` sequences: ``[0, seq_len, 2*seq_len, ..., max_tokens]``."""
    b = [0]
    for _ in range(dims.max_tokens // dims.seq_len):
        b.append(b[-1] + dims.seq_len)
    return b


def token_bytes(cfg, dims, seed):
    """Deterministic in-vocabulary tokens + targets for one round, as the
    int32 wire bytes ``put_object`` expects. Every target is valid, so the
    round's valid-token count is the full ``max_tokens``. The values need
    only be reproducible and in range: the engine-vs-engine invariance
    checks feed identical bytes to both runs, and the accumulation twin is
    fed the very same bytes as the engine."""
    gen = torch.Generator().manual_seed(seed)
    tok = torch.randint(0, cfg.vocab_size, (dims.max_tokens,),
                        generator=gen, dtype=torch.int32)
    tgt = torch.randint(0, cfg.vocab_size, (dims.max_tokens,),
                        generator=gen, dtype=torch.int32)
    return tok.numpy().tobytes(), tgt.numpy().tobytes()


def valid_rows_of(target_bytes):
    return int((torch.frombuffer(bytearray(target_bytes),
                                 dtype=torch.int32) >= 0).sum())


def weight_ids(program):
    return sorted(o.id for o in program.initial_objects
                  if o.id.startswith("W_"))


def run_family(client, cfg, *, seed=SEED, jitter=None,
               capacity=PLAN_BUDGET, recompute=None):
    """One family training step on ``client``; returns ``{"loss": float,
    W_id: bytes, ...}`` with each weight object read back post-step as raw
    host bytes.

    The store is blank at the start of a test (the conftest wiped it) and
    init_model re-seeds the model, so a plain put stages the tokens/targets.
    Grad accumulation is forced to a single round so one token/target pair
    fully feeds the program. ``jitter`` rides the resolver spec as the
    engine's per-task launch-delay knob; ``capacity`` is the planner's
    fast-memory budget and ``recompute`` a per-activation recompute map —
    the last two change the PLAN without changing the math. Free-poisoning is
    a boot flag, so it is exercised by passing a poison-booted client."""
    cfg = replace(cfg, grad_accum_rounds=1)
    fam = resolve_family(cfg)
    dims = fam.derive_dims(cfg)
    program = (fam.lower(cfg, recompute_levels=recompute) if recompute
               else fam.lower(cfg))
    planned = plan_program(program, fast_memory_capacity=capacity)
    tok, tgt = token_bytes(cfg, dims, seed)
    boundaries = uniform_boundaries(dims)
    spec = {"kind": "model_family", "family": resolver_family(cfg),
            "cfg": cfg_dict(cfg), "hyper": adamw_hyper_spec(AdamWHyper())}
    if jitter:
        spec["debug_jitter"] = jitter

    init_model(client, resolver_family(cfg), cfg_dict(cfg), seed=seed)
    client.put_object("tokens_0_0", tok)
    client.put_object("targets_0_0", tgt)
    reg = client.register_program(program_to_dict(planned.program),
                                  resolver=spec)
    assert not reg["bindings"]["missing_inputs"], reg
    res = client.run(reg["prog_id"],
                     args={"step": 0, "valid_rows": valid_rows_of(tgt),
                           "seq_lens": {"0": boundaries}},
                     fetch=["loss_0_0"])
    assert res.get("state") == "done", res
    out = {"loss": res["fetched"]["loss_0_0"]}
    for wid in weight_ids(program):
        out[wid] = bytes(client.get_object(wid))
    return out


def assert_same_result(a, b, tol=1e-3):
    """Two runs of the same math must agree: loss within ``tol`` and every
    weight object equal. Byte-identical is the fast path; otherwise decode
    bf16 and compare after zeroing NaNs — the free-poison run leaves
    allocator-padding bytes (outside every field) as 0xFF, which reads as
    NaN under the bf16 view but is never read by the math."""
    assert abs(a["loss"] - b["loss"]) / max(abs(b["loss"]), 1e-9) < tol, \
        (a["loss"], b["loss"])
    for k in a:
        if k == "loss":
            continue
        if a[k] == b[k]:
            continue
        va = torch.nan_to_num(torch.frombuffer(bytearray(a[k]),
                                               dtype=torch.bfloat16).float())
        vb = torch.nan_to_num(torch.frombuffer(bytearray(b[k]),
                                               dtype=torch.bfloat16).float())
        assert rel_l2(va, vb) < tol, f"{k}: rel_l2={rel_l2(va, vb)}"


# --- lowering ----------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(FAMILY_PRESETS))
def test_lowering_validates_and_plans(name):
    cfg = preset(name)
    fam = resolve_family(cfg)
    assert fam.name == name
    program = fam.lower(cfg)
    validate_program(program)
    planned = plan_program(program, fast_memory_capacity=PLAN_BUDGET)
    log = simulate_program(planned.program)
    assert max(iv.end for iv in log.task_intervals) > 0


# --- golden model step (client vs pure-torch twin) ---------------------------


@pytest.mark.parametrize("name", sorted(FAMILY_PRESETS))
def test_golden_model_step(name, client):
    client_model_step(preset(name), seed=SEED, client=client,
                      field_atol=FIELD_ATOL.get(name),
                      param_atol=PARAM_ATOL.get(name),
                      **gate_kwargs(name)).assert_ok()


@pytest.mark.parametrize("name", sorted(FAMILY_PRESETS))
def test_golden_model_step_batch2_packed(name, client):
    cfg = replace(preset(name), batch=2, seq_len=64)
    client_model_step(cfg, seed=SEED, client=client,
                      field_atol=FIELD_ATOL.get(name),
                      param_atol=PARAM_ATOL.get(name),
                      **gate_kwargs(name)).assert_ok()


# --- gradient accumulation (client vs twin, two rounds) ----------------------


@pytest.mark.parametrize("name", sorted(FAMILY_PRESETS))
def test_grad_accum_two_rounds(name, client):
    """Two accumulation rounds through the client leave final parameters
    matching the twin's accumulated update. ``client_model_step`` forces a
    single round, so the two-round leg is driven here directly: the same
    two token/target pairs feed both the engine (one packed ``run`` with a
    per-round ``seq_lens`` map and the global valid-token denominator) and
    the twin (two forwards, each scaled to the global denominator, summed
    into one backward, then one AdamW step). The twin is initialized from
    the server's own seeded weights, fetched through the client, so both
    legs share an exact init."""
    cfg = replace(preset(name), grad_accum_rounds=2)
    fam = resolve_family(cfg)
    dims = fam.derive_dims(cfg)
    rounds = cfg.grad_accum_rounds
    b_rows = dims.max_tokens // dims.seq_len
    program = fam.lower(cfg)
    planned = plan_program(program, fast_memory_capacity=PLAN_BUDGET)
    payloads = [token_bytes(cfg, dims, SEED * 16 + r) for r in range(rounds)]
    total_valid = sum(valid_rows_of(tgt) for _tok, tgt in payloads)
    boundaries = uniform_boundaries(dims)
    spec = {"kind": "model_family", "family": resolver_family(cfg),
            "cfg": cfg_dict(cfg), "hyper": adamw_hyper_spec(AdamWHyper())}

    init_model(client, resolver_family(cfg), cfg_dict(cfg), seed=SEED)

    # twin from the server's seeded init weights (read pre-run)
    twin = bridges.build_reference_model(cfg)
    bridges.load_reference_init(twin, cfg, dims, ClientFinalBytes(client))
    twin.train()

    for r, (tok, tgt) in enumerate(payloads):
        client.put_object(f"tokens_0_{r}", tok)
        client.put_object(f"targets_0_{r}", tgt)

    loss_total = None
    for tok, tgt in payloads:
        toks = torch.frombuffer(bytearray(tok), dtype=torch.int32) \
            .long().cuda().view(b_rows, dims.seq_len)
        tgts = torch.frombuffer(bytearray(tgt), dtype=torch.int32) \
            .long().cuda().view(b_rows, dims.seq_len)
        scale = valid_rows_of(tgt) / total_valid
        ce = twin.loss(toks, tgts) * scale
        loss_total = ce if loss_total is None else loss_total + ce
    loss_total.backward()
    hp = AdamWHyper()
    for par in twin.parameters():
        if par.grad is None:
            continue
        m = torch.zeros_like(par)
        v = torch.zeros_like(par)
        adamw_field_step(par.data, par.grad, m, v, lr=hp.lr,
                         beta1=hp.beta1, beta2=hp.beta2, eps=hp.eps,
                         weight_decay=hp.weight_decay, step=1)

    reg = client.register_program(program_to_dict(planned.program),
                                  resolver=spec)
    assert not reg["bindings"]["missing_inputs"], reg
    res = client.run(
        reg["prog_id"],
        args={"step": 0, "valid_rows": total_valid,
              "seq_lens": {str(r): boundaries for r in range(rounds)}},
        fetch=[f"loss_0_{r}" for r in range(rounds)])
    assert res.get("state") == "done", res
    engine_state = bridges.to_reference_state_dict(
        cfg, ClientFinalBytes(client))

    twin_state = dict(twin.state_dict())
    field_atol = envelope(name)
    for pname, engine_tensor in engine_state.items():
        atol = match_field_atol(pname, field_atol)
        if atol is not None:
            # zero-init sign-lottery field (a state-path bias whose one-step
            # update is dominated by bf16-vs-AdamW sign flips): the honest
            # comparison is an absolute envelope, not a relative one.
            gap = float((engine_tensor.float().cpu()
                         - twin_state[pname].float().cpu()).abs().max())
            assert gap <= atol, (name, pname, gap)
            continue
        err = rel_l2(engine_tensor, twin_state[pname])
        assert err < GA_TOL, (name, pname, err)


# --- engine-invariance battery (run A vs run B) ------------------------------


@pytest.mark.parametrize("name", sorted(FAMILY_PRESETS))
def test_fixed_seed_bitwise_deterministic(name, client):
    cfg = preset(name)
    a = run_family(client, cfg, seed=SEED)
    b = run_family(client, cfg, seed=SEED)
    assert a["loss"] == b["loss"]
    for k in a:
        if k != "loss":
            assert a[k] == b[k], k


@pytest.mark.parametrize("name", sorted(FAMILY_PRESETS))
def test_poison_on_free_changes_nothing(name, client):
    cfg = preset(name)
    base = run_family(client, cfg, seed=SEED)
    with out_of_process_server(backing_gib=BACKING_GIB,
                               poison_on_free=True) as poison_client:
        poisoned = run_family(poison_client, cfg, seed=SEED)
    assert_same_result(poisoned, base)
    assert poisoned["loss"] == poisoned["loss"]        # not NaN


@pytest.mark.parametrize("name", sorted(FAMILY_PRESETS))
def test_result_invariant_to_runtime_jitter(name, client):
    """A per-task random launch delay makes the tasks' ACTUAL runtimes
    diverge from the plan's cost estimates. The engine sequences work on
    completion events, not on the estimates, so the result is invariant:
    correctness does not depend on the cost model being accurate."""
    cfg = preset(name)
    base = run_family(client, cfg, seed=SEED)
    jittered = run_family(client, cfg, seed=SEED,
                          jitter={"min_us": 20, "max_us": 400, "seed": 123})
    assert_same_result(jittered, base)


@pytest.mark.parametrize("name", sorted(FAMILY_PRESETS))
def test_plan_invariance(name, client):
    cfg = preset(name)
    generous = run_family(client, cfg, seed=SEED, capacity=PLAN_BUDGET)
    tight = run_family(client, cfg, seed=SEED, capacity=TIGHT_BUDGET)
    assert_same_result(generous, tight, tol=CROSS_PLAN_TOL)


@pytest.mark.parametrize("name", sorted(FAMILY_PRESETS))
def test_measured_costs_replan_still_golden(name, client):
    """Re-planning the step so the schedule changes must not change the
    math. Real measured-cost profiling of the backward signatures needs a
    device backend (forbidden on this client-only path) and is covered
    family-agnostically by tests/dataflow/runtime/test_engine_stress.py;
    here the plan is changed by forcing a full recompute schedule, which
    is the same invariance claim without the profiler."""
    cfg = preset(name)
    base = run_family(client, cfg, seed=SEED)
    levels = {f"A_0_0_{i}": 1 for i in range(cfg.n_layers)}
    replanned = run_family(client, cfg, seed=SEED, recompute=levels)
    assert_same_result(replanned, base, tol=CROSS_PLAN_TOL)


# --- state-leak guards (the per-test reset + per-test twin) -------------------


@pytest.mark.parametrize("name", sorted(FAMILY_PRESETS))
def test_reseed_restores_pristine_init(name, client):
    """The reset the shared server relies on: wiping the store and re-seeding
    (init_model) restores every persistent weight to its pristine seeded
    bytes, and a follow-on run reproduces an earlier run's loss and weights
    bit-for-bit. So no trained weight, optimizer moment, or aux count survives
    the per-test reset to leak into the next test."""
    cfg = replace(preset(name), grad_accum_rounds=1)
    wids = weight_ids(resolve_family(cfg).lower(cfg))

    init_model(client, resolver_family(cfg), cfg_dict(cfg), seed=SEED)
    pristine = {w: bytes(client.get_object(w)) for w in wids}

    first = run_family(client, cfg, seed=SEED)          # trains W in place

    client.wipe("all")                                  # the per-test reset
    init_model(client, resolver_family(cfg), cfg_dict(cfg), seed=SEED)
    reset = {w: bytes(client.get_object(w)) for w in wids}
    assert reset == pristine, name                      # weights fully reset

    second = run_family(client, cfg, seed=SEED)         # moments + aux reset
    assert second["loss"] == first["loss"], name
    for k in first:
        if k != "loss":
            assert second[k] == first[k], k


@pytest.mark.parametrize("name", sorted(FAMILY_PRESETS))
def test_reference_twin_build_is_stateless(name, client):
    """Two freshly-built reference twins, seeded from the same server init
    and stepped on the same tokens, agree bit-for-bit. The pure-torch twin
    is rebuilt per test (never shared), so the only way it could leak across
    tests is process-global state — an MoE step counter or balance bias that
    outlives a build; this proves there is none."""
    cfg = replace(preset(name), grad_accum_rounds=1)
    fam = resolve_family(cfg)
    dims = fam.derive_dims(cfg)
    b_rows = dims.max_tokens // dims.seq_len
    tok, tgt = token_bytes(cfg, dims, SEED)

    init_model(client, resolver_family(cfg), cfg_dict(cfg), seed=SEED)
    toks = torch.frombuffer(bytearray(tok), dtype=torch.int32) \
        .long().cuda().view(b_rows, dims.seq_len)
    tgts = torch.frombuffer(bytearray(tgt), dtype=torch.int32) \
        .long().cuda().view(b_rows, dims.seq_len)

    losses, states = [], []
    for _ in range(2):
        torch.manual_seed(SEED)                         # isolate any RNG order
        twin = bridges.build_reference_model(cfg)
        bridges.load_reference_init(twin, cfg, dims, ClientFinalBytes(client))
        twin.train()
        loss = twin.loss(toks, tgts)
        loss.backward()
        hp = AdamWHyper()
        for par in twin.parameters():
            if par.grad is None:
                continue
            m = torch.zeros_like(par)
            v = torch.zeros_like(par)
            adamw_field_step(par.data, par.grad, m, v, lr=hp.lr,
                             beta1=hp.beta1, beta2=hp.beta2, eps=hp.eps,
                             weight_decay=hp.weight_decay, step=1)
        losses.append(float(loss.detach()))
        states.append({k: v.detach().float().cpu().clone()
                       for k, v in twin.state_dict().items()})

    assert losses[0] == losses[1], (name, losses)
    for k in states[0]:
        assert torch.equal(states[0][k], states[1][k]), (name, k)
