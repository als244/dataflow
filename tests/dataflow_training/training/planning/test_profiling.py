"""Fast-profile configuration and bounded PCIe contender accounting."""

from dataflow_training.run.profiling import (DEFAULT_CONTEND_PCIE,
                                             DEFAULT_REPEATS,
                                             DEFAULT_SOAK_SECONDS,
                                             PRODUCTION_SAMPLE_SECONDS,
                                             _cover_count)


def test_pytest_uses_the_fast_profile_configuration():
    assert PRODUCTION_SAMPLE_SECONDS == 0.0
    assert DEFAULT_SOAK_SECONDS == 0.0
    assert DEFAULT_CONTEND_PCIE is False
    assert DEFAULT_REPEATS == 1


def test_contender_rounding_never_builds_an_unbounded_backlog():
    chunk_us = 14_000.0
    interval_us = 50_000.0
    credit_us = 0.0
    copies = 0

    for _ in range(400):
        count, credit_us = _cover_count(interval_us, chunk_us, credit_us)
        copies += count
        assert 0.0 <= credit_us < chunk_us

    queued_us = copies * chunk_us
    requested_us = 400 * interval_us
    assert requested_us <= queued_us < requested_us + chunk_us
