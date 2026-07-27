"""Profiling device-memory lifecycle gates.

The profiler measures many cost tables per process. Its contract
(run/profiling.py module docstring): tables share one process-lifetime
stream trio so kernel scratch reuses instead of stranding per pass, and
each table returns torch's cache to the driver on exit — so a sweep's
reserved memory never grows with the NUMBER of tables. The regression
this pins: a fresh stream per table stranded every table's cached
scratch (reserved grew ~3 GiB/table tier on an 8B sweep) until raw
operand allocations starved at free~0 mid-sweep on a 24 GiB card.

Tests:
- test_tables_leave_no_reserved_memory: building several cost tables
  back-to-back (cold disk cache) leaves torch's reserved memory at the
  baseline after EVERY table — the per-table release holds and nothing
  accumulates with table count.
"""
import pytest

pytestmark = [pytest.mark.gpu]


def test_tables_leave_no_reserved_memory(tmp_path):
    pytest.importorskip("cuda.bindings.runtime")
    import torch
    from dataclasses import replace

    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow_training.model_families.families import family
    from dataflow_training.run.profiling import measured_profile_table

    fam = family("llama3")
    cfg = fam.config_type.tiny()
    backend = CudaBackend()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_reserved()
    allowance = 128 << 20        # headroom for allocator bookkeeping

    # distinct geometries => distinct signatures => real device work per
    # table (cold cache under tmp_path), which is what would accumulate
    for ga, batch in ((2, 1), (2, 2), (4, 1)):
        geo = replace(cfg, grad_accum_rounds=ga, batch=batch)
        dims = fam.derive_dims(geo)
        resolver = fam.build_resolver(dims)
        profiles = measured_profile_table(fam, geo, resolver, backend,
                                          cache_dir=str(tmp_path))
        assert profiles, (ga, batch)
        reserved = torch.cuda.memory_reserved()
        assert reserved <= baseline + allowance, (
            f"table (ga={ga}, batch={batch}) left "
            f"{(reserved - baseline) / 2**20:.0f} MiB reserved — the "
            f"per-table release regressed and a long sweep will strand "
            f"memory per table again")
