"""The shared client parity helper on the out-of-process daemon.

client_model_step runs a family step through a real out-of-process dataflowd and
reproduces the in-process check_model_step verdict at FULL PARITY — loss, final
params, one-step gradients, and MoE assignment counts — reading every engine
value as a host copy via the client. This gate proves it boots/reaps the daemon
cleanly, produces a passing verdict, and reproduces the in-process report key by
key (same deterministic engine, same twin) for a dense family (llama3) and an
MoE family with count entries (olmoe).

Tests:
- test_out_of_process_daemon_boots_and_reaps: a spawned daemon serves a store
  roundtrip and is reaped on exit.
- test_client_model_step_llama3_passes: the client-path verdict for llama3
  passes its calibrated band against the twin.
- test_client_model_step_matches_in_process_llama3: the client-path report
  reproduces the in-process check_model_step report — loss, params, grads.
- test_client_model_step_matches_in_process_olmoe: the client-path report
  reproduces the in-process report including the MoE counts entries.
"""
from dataclasses import replace as dc_replace

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no GPU", allow_module_level=True)

from dataflow_training.run import presets as P                          # noqa: E402
from dataflow_training.testing.client_daemon import out_of_process_daemon  # noqa: E402
from dataflow_training.testing.client_parity import client_model_step   # noqa: E402
from dataflow_training.testing.gradcheck import (                       # noqa: E402
    check_model_step, family_gate_kwargs)

pytestmark = [pytest.mark.gpu]

SEED = 7
# The out-of-process engine leg is bitwise-deterministic against the
# in-process one, so every shared error (loss, params, grads, counts)
# reproduces the check_model_step value to the float (measured: 0.0 across
# both families). This envelope is generous over that yet still tight
# enough to catch an optimizer-step (hyper) regression, which perturbs the
# init-dominated param entries by ~1e-2 — the sharpness the param-space
# comparison exists to provide.
MATCH_ATOL = 1e-3


def assert_reproduces(client_report, inproc_report):
    """Both verdicts pass and every shared error key reproduces the
    in-process value (categorically for the inf sentinels). Returns the
    shared key set for per-test coverage assertions."""
    assert inproc_report.ok, inproc_report.errors
    assert client_report.ok, client_report.errors
    shared = set(client_report.errors) & set(inproc_report.errors)
    for name in shared:
        cv = client_report.errors[name]
        iv = inproc_report.errors[name]
        if cv == float("inf") or iv == float("inf"):
            assert cv == iv, (name, cv, iv)
        else:
            assert abs(cv - iv) < MATCH_ATOL, (name, cv, iv)
    return shared


def test_out_of_process_daemon_boots_and_reaps():
    with out_of_process_daemon(backing_gib=2.0) as client:
        client.put_object("k", b"payload")
        assert bytes(client.get_object("k")) == b"payload"


def test_client_model_step_llama3_passes():
    report = client_model_step(P.smoke_preset(), seed=SEED,
                               **family_gate_kwargs("llama3"))
    report.assert_ok()


def test_client_model_step_matches_in_process_llama3():
    gate = family_gate_kwargs("llama3")
    inproc = check_model_step(
        dc_replace(P.smoke_preset(), grad_accum_rounds=1),
        fast_memory_capacity=4 << 30, seed=SEED, **gate)
    client = client_model_step(P.smoke_preset(), seed=SEED, **gate)
    shared = assert_reproduces(client, inproc)
    # the reproduced report carries the param-space entries, not only grads
    assert "loss" in shared
    assert any(k.startswith("grad:") for k in shared), sorted(shared)
    assert any(k != "loss" and not k.startswith(("grad:", "counts:"))
               for k in shared), sorted(shared)


def test_client_model_step_matches_in_process_olmoe():
    gate = family_gate_kwargs("olmoe")
    inproc = check_model_step(
        dc_replace(P.olmoe_smoke_preset(), grad_accum_rounds=1),
        fast_memory_capacity=4 << 30, seed=SEED, **gate)
    client = client_model_step(P.olmoe_smoke_preset(), seed=SEED, **gate)
    shared = assert_reproduces(client, inproc)
    # the MoE assignment-count entries are reproduced and pass the budget
    counts_keys = [k for k in shared if k.startswith("counts:")]
    assert counts_keys, sorted(shared)
    for k in counts_keys:
        assert client.errors[k] == 0.0, (k, client.errors[k])
