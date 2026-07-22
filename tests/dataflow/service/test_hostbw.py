"""Boot host-bandwidth probe: sane, positive, and present in
engine_status (fake boots skip the measurement).

Tests:
- test_probe_reports_positive_lanes: measure_host_bw returns positive host-copy and bf16-add lanes, plus positive h2d/d2h lanes when CUDA is available.
- test_zero_disables: measure_host_bw(0) returns an empty dict, disabling the measurement.
"""
import pytest

torch = pytest.importorskip("torch")

from dataflow.service.hostbw import measure_host_bw


@pytest.mark.gpu
def test_probe_reports_positive_lanes():
    out = measure_host_bw(32)
    assert out["host_copy_gbs"] > 0
    assert out["host_bf16_add_gbs"] > 0
    if torch.cuda.is_available():
        assert out["h2d_gbs"] > 0 and out["d2h_gbs"] > 0


def test_zero_disables():
    assert measure_host_bw(0) == {}
