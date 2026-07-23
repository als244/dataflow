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
  gpu              a GPU torch can drive (CUDA or ROCm) — most
                   modeling/training tests need only this
  cuda             an NVIDIA CUDA runtime specifically — the CUDA backend
                   and cuda-python paths, which are CUDA-only until an AMD
                   backend exists (a GPU is not necessarily CUDA)
  fleet            multi-daemon gates (opt-in lane: pytest -m fleet)
  corpus           the shard corpus at datasets/fineweb10B
  internet         a route to the public internet — a ready gate for a
                   future test that must fetch a remote asset; nothing
                   uses it yet, so offline nodes have nothing to skip
  topology_remote  a topology.toml declaring at least one remote host
  rdma             an active RDMA port the engine can bring up — RoCE
                   today; InfiniBand is a documented drop-in, so an
                   IB-only host skips these rather than failing
  ncclbind         a loadable libnccl
  vram(gib=N)      at least N GiB of free device memory at collection
"""
from pathlib import Path

import pytest


def has_gpu() -> bool:
    """A GPU torch can drive — CUDA or ROCm (torch's ROCm build exposes
    HIP through the torch.cuda API). Modeling/training tests only need a
    GPU, not CUDA specifically."""
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def has_cuda() -> bool:
    """An NVIDIA CUDA runtime specifically (not ROCm) — for tests of the
    CUDA backend and the cuda-python paths, which are CUDA-only until an
    AMD backend exists. torch.version.cuda distinguishes a CUDA build
    from a ROCm one."""
    try:
        import torch

        return torch.cuda.is_available() and torch.version.cuda is not None
    except Exception:
        return False


def has_corpus() -> bool:
    from dataflow_training.distributed.topology import repo_root

    return (repo_root() / "datasets" / "fineweb10B").exists()


def has_internet() -> bool:
    """A route to the public internet, for a future test that must fetch
    a remote asset (a tokenizer vocabulary, a hub download). Nothing needs
    it yet — the gate is wired so the day one does, a compute node with no
    outbound route skips it cleanly instead of failing. Probes a short TCP
    connect to a well-known anycast address by raw IP, so a missing or
    broken DNS resolver can't hang the check."""
    import socket

    try:
        with socket.create_connection(("1.1.1.1", 443), timeout=1.5):
            return True
    except OSError:
        return False


def has_topology_remote() -> bool:
    try:
        from dataflow_training.distributed.topology import (
            load_topology_or_none,
        )

        topo = load_topology_or_none()
        return topo is not None and bool(topo.remotes())
    except Exception:
        return False


def has_active_rdma_port() -> bool:
    """An active RDMA port the engine can currently bring a connection up
    on — sysfs only, no pyverbs needed for the probe.

    The engine wires RDMA over RoCE (an Ethernet-link-layer port with a
    RoCE v2 IPv4 GID). InfiniBand is a documented drop-in that isn't
    wired yet, so an IB-only host has no usable port here and its
    rdma-marked tests skip cleanly instead of failing."""
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
    "gpu": (has_gpu, "no GPU available"),
    "cuda": (has_cuda, "no NVIDIA CUDA runtime (ROCm or CPU-only)"),
    "corpus": (has_corpus, "shard corpus not present "
                           "(datasets/fineweb10B)"),
    "internet": (has_internet, "no route to the public internet"),
    "topology_remote": (has_topology_remote,
                        "no topology.toml with a remote host"),
    "rdma": (has_active_rdma_port,
             "no active RDMA port the engine can use (RoCE today)"),
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
