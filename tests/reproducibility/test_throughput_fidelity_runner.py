"""Control-flow regressions for the reproducibility campaign runner.

Tests:
- test_measure_stage_accepts_an_infeasible_empty_selection: an empty selected-cell set is valid evidence, emits empty measurement files, and returns without launching a subprocess.
"""

from types import SimpleNamespace

from reproducibility.throughput_fidelity import run_experiment


def test_measure_stage_accepts_an_infeasible_empty_selection(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    cells = tmp_path / "cells.json"
    cells.write_text("[]\n")
    cfg = SimpleNamespace(
        cells_json=cells,
        data=data,
        opts=("adamw",),
        resume=False,
    )

    # ``cfg`` deliberately omits every subprocess-launch field. If the empty
    # selection ever falls through to the launch path, the test fails fast.
    run_experiment.stage_measure(cfg, {"preset": "unused", "backing_gib": 1})

    assert (data / "measure_adamw.jsonl").read_text() == ""
