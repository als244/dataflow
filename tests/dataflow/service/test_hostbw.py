"""Boot host-bandwidth probe: sane, positive, and present in
engine_status (fake boots skip the measurement)."""
import pytest

torch = pytest.importorskip("torch")

from dataflow.service.hostbw import measure_host_bw


def test_probe_reports_positive_lanes():
    out = measure_host_bw(32)
    assert out["host_copy_gbs"] > 0.1
    assert out["host_bf16_add_gbs"] > 0.1
    if torch.cuda.is_available():
        assert out["h2d_gbs"] > 0.5 and out["d2h_gbs"] > 0.5


def test_zero_disables():
    assert measure_host_bw(0) == {}
