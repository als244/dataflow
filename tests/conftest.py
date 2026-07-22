"""Suite-wide requirement gating: every environment a test needs is a
MARKER, every marker is a PROBE, and a machine that lacks something
SKIPS with a precise reason instead of failing.

The contract (portability): download this repo onto any machine — one
GPU or eight, any vendor's NIC or none, no topology.toml, no datasets,
no optional extras — and `pytest` must end green: passes plus clean
skips, never machine-caused failures. Tests therefore never assert
facts about hardware (link speeds, VRAM sizes, architectures); they
declare requirements here and derive any bound they need from probes
or from structure.

Markers (register in pyproject.toml):
  gpu              a CUDA device
  fleet            multi-daemon gates (opt-in lane: pytest -m fleet)
  sim              the dataflow_sim extra (planner/simulator interop)
  corpus           the shard corpus at datasets/fineweb10B
  topology_remote  a topology.toml declaring at least one remote host
  rdma             an rdma device with an ACTIVE RoCE v2 IPv4 port
  ncclbind         a loadable libnccl
  vram(gib=N)      at least N GiB of free device memory at collection
"""
from pathlib import Path

import pytest


def has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def has_sim() -> bool:
    import importlib.util

    return importlib.util.find_spec("dataflow_sim") is not None


def has_corpus() -> bool:
    from dataflow_training.distributed.topology import repo_root

    return (repo_root() / "datasets" / "fineweb10B").exists()


def has_topology_remote() -> bool:
    try:
        from dataflow_training.distributed.topology import (
            load_topology_or_none,
        )

        topo = load_topology_or_none()
        return topo is not None and bool(topo.remotes())
    except Exception:
        return False


def has_active_roce_port() -> bool:
    """An rdma device with an ACTIVE port carrying a RoCE v2
    IPv4-mapped GID — sysfs only, no pyverbs needed for the probe."""
    root = Path("/sys/class/infiniband")
    if not root.is_dir():
        return False
    for dev in root.iterdir():
        port = dev / "ports" / "1"
        try:
            if not (port / "state").read_text().strip().startswith("4"):
                continue
            types = port / "gid_attrs" / "types"
            for entry in types.iterdir():
                if entry.read_text().strip() != "RoCE v2":
                    continue
                gid = (port / "gids" / entry.name).read_text().strip()
                if gid.startswith("0000:0000:0000:0000:0000:ffff"):
                    return True
        except OSError:
            continue
    return False


def has_libnccl() -> bool:
    try:
        from dataflow.service.peer import nccl

        return nccl.available()
    except Exception:
        return False


def free_vram_gib() -> float:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        free, _total = torch.cuda.mem_get_info()
        return free / (1 << 30)
    except Exception:
        return 0.0


PROBES = {
    "gpu": (has_cuda, "no CUDA device"),
    "sim": (has_sim, "dataflow_sim extra not installed"),
    "corpus": (has_corpus, "shard corpus not present "
                           "(datasets/fineweb10B)"),
    "topology_remote": (has_topology_remote,
                        "no topology.toml with a remote host"),
    "rdma": (has_active_roce_port,
             "no ACTIVE RoCE v2 IPv4 rdma port"),
    "ncclbind": (has_libnccl, "libnccl not loadable"),
}

PROBE_CACHE: dict = {}


def probe(name: str) -> bool:
    if name not in PROBE_CACHE:
        PROBE_CACHE[name] = PROBES[name][0]()
    return PROBE_CACHE[name]


def pytest_collection_modifyitems(config, items):
    for item in items:
        for name, (_fn, why) in PROBES.items():
            if item.get_closest_marker(name) is not None \
                    and not probe(name):
                item.add_marker(pytest.mark.skip(reason=why))
        vram = item.get_closest_marker("vram")
        if vram is not None:
            need = float(vram.kwargs.get("gib", vram.args[0]
                                         if vram.args else 0))
            got = free_vram_gib()
            if got < need:
                item.add_marker(pytest.mark.skip(
                    reason=f"needs {need:g} GiB free device memory, "
                           f"{got:.1f} available"))
