"""The shared client parity helper on the out-of-process daemon.

client_model_step runs a family step through a real out-of-process dataflowd and
compares loss + gradients against the pure-torch twin, reading every engine
value as a host copy via the client. This gate proves it boots/reaps the daemon
cleanly, produces a passing verdict for llama3, and reproduces the in-process
check_model_step verdict (same deterministic engine gradients, same twin).

Tests:
- test_out_of_process_daemon_boots_and_reaps: a spawned daemon serves a store
  roundtrip and is reaped on exit.
- test_client_model_step_llama3_passes: the client-path verdict for llama3
  passes its gradient band against the twin.
- test_client_model_step_matches_in_process_llama3: the client-path gradient
  errors match the in-process check_model_step errors.
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


def test_out_of_process_daemon_boots_and_reaps():
    with out_of_process_daemon(backing_gib=2.0) as client:
        client.put_object("k", b"payload")
        assert bytes(client.get_object("k")) == b"payload"


def test_client_model_step_llama3_passes():
    report = client_model_step(P.smoke_preset(), seed=SEED,
                               grad_tol=3e-2, min_cosine=0.999)
    report.assert_ok()


def test_client_model_step_matches_in_process_llama3():
    gate = family_gate_kwargs("llama3")
    grad_tol = gate.get("grad_tol", 3e-2)
    min_cosine = gate.get("min_cosine", 0.999)
    inproc = check_model_step(
        dc_replace(P.smoke_preset(), grad_accum_rounds=1),
        fast_memory_capacity=4 << 30, seed=SEED, **gate)
    client = client_model_step(P.smoke_preset(), seed=SEED,
                               grad_tol=grad_tol, min_cosine=min_cosine)
    assert inproc.ok
    assert client.ok
    # the client engine dW is deterministic and equals the in-process dW, so
    # each shared gradient error matches closely -> the verdict reproduces
    for name, err in client.errors.items():
        if name in inproc.errors:
            assert abs(err - inproc.errors[name]) < 5e-3, \
                (name, err, inproc.errors[name])
