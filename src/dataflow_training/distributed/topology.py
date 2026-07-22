"""Fleet topology: every machine-specific fact — host names, ssh
destinations, data-plane addresses, RDMA devices, memory sizes,
groups — is read from a topology.toml file. The codebase itself names
no machines, addresses, or filesystem paths.

Search order: an explicit path argument, else topology.toml in the
current directory, else at the repo root. topology.toml is gitignored
(per-setup); topology.example.toml documents the schema.
"""
from __future__ import annotations

import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path


def repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root (pyproject.toml) not found above "
                       + str(Path(__file__).resolve()))


@dataclass(frozen=True)
class HostSpec:
    name: str
    peer_listen: str               # data-plane bind, "ip:port"
    ssh: str | None = None         # None => local host
    python: str = sys.executable
    repo: str = ""
    nsys: str = "nsys"
    ib_dev: str | None = None
    gid_index: int = 3             # RoCE v2 IPv4 GID index on ib_dev
                                   # (NCCL_IB_GID_INDEX; the populate
                                   # helper will probe this per device)
    iface: str | None = None       # fast-link netdev (NCCL_SOCKET_IFNAME)
    backing_gib: float = 8.0
    budget_gib: float = 8.0
    device: int = 0                # CUDA device index (multi-GPU hosts
                                   # use one topology entry per GPU)

    def is_local(self) -> bool:
        return self.ssh is None

    def peer_ip(self) -> str:
        return self.peer_listen.rsplit(":", 1)[0]

    def peer_addr(self, port: int | None = None) -> str:
        """The data-plane address, optionally rebased onto another
        port (test lanes keep the topology's IP but their own port)."""
        if port is None:
            return self.peer_listen
        return f"{self.peer_ip()}:{port}"


@dataclass(frozen=True)
class GroupSpec:
    name: str
    members: tuple
    backend: str = "hostmem"


@dataclass(frozen=True)
class Topology:
    conductor: str
    hosts: dict
    groups: dict
    source: str

    def host(self, name: str) -> HostSpec:
        if name not in self.hosts:
            raise KeyError(f"host {name!r} not in topology "
                           f"({self.source}); have {sorted(self.hosts)}")
        return self.hosts[name]

    def local(self) -> HostSpec:
        return self.host(self.conductor)

    def remotes(self) -> list:
        return [h for h in self.hosts.values() if not h.is_local()]

    def group(self, name: str) -> GroupSpec:
        if name not in self.groups:
            raise KeyError(f"group {name!r} not in topology "
                           f"({self.source}); have {sorted(self.groups)}")
        return self.groups[name]

    def group_hosts(self, name: str) -> list:
        return [self.host(m) for m in self.group(name).members]


def find_topology(explicit: str | None = None) -> Path | None:
    if explicit:
        return Path(explicit)
    for cand in (Path.cwd() / "topology.toml",
                 repo_root() / "topology.toml"):
        if cand.is_file():
            return cand
    return None


def build_host(name: str, entry: dict) -> HostSpec:
    ssh = entry.get("ssh")
    if "peer_listen" not in entry:
        raise ValueError(f"host {name!r}: peer_listen is required")
    if ssh is not None:
        for key in ("python", "repo"):
            if not entry.get(key):
                raise ValueError(f"remote host {name!r}: {key!r} is "
                                 f"required (absolute path)")
    return HostSpec(
        name=name,
        peer_listen=entry["peer_listen"],
        ssh=ssh,
        python=entry.get("python", sys.executable),
        repo=entry.get("repo", str(repo_root())),
        nsys=entry.get("nsys", "nsys"),
        ib_dev=entry.get("ib_dev"),
        gid_index=int(entry.get("gid_index", 3)),
        iface=entry.get("iface"),
        backing_gib=float(entry.get("backing_gib", 8.0)),
        budget_gib=float(entry.get("budget_gib", 8.0)),
        device=int(entry.get("device", 0)))


def load_topology(path: str | None = None) -> Topology:
    found = find_topology(path)
    if found is None:
        raise FileNotFoundError(
            "no topology.toml found (looked in the current directory "
            "and the repo root); copy topology.example.toml to "
            "topology.toml and edit it for your setup")
    raw = tomllib.loads(found.read_text())

    hosts = {}
    for name, entry in raw.get("hosts", {}).items():
        hosts[name] = build_host(name, entry)
    if not hosts:
        raise ValueError(f"{found}: no [hosts.*] entries")

    groups = {}
    for name, entry in raw.get("groups", {}).items():
        members = tuple(entry.get("members", ()))
        if not members:
            raise ValueError(f"{found}: group {name!r} has no members")
        for m in members:
            if m not in hosts:
                raise ValueError(f"{found}: group {name!r} member "
                                 f"{m!r} is not a [hosts.*] entry")
        groups[name] = GroupSpec(name=name, members=members,
                                 backend=entry.get("backend", "hostmem"))

    conductor = raw.get("conductor")
    if conductor not in hosts:
        raise ValueError(f"{found}: conductor {conductor!r} must name "
                         f"a [hosts.*] entry")
    if hosts[conductor].ssh is not None:
        raise ValueError(f"{found}: conductor host {conductor!r} must "
                         f"be local (no ssh key)")
    return Topology(conductor=conductor, hosts=hosts, groups=groups,
                    source=str(found))


def load_topology_or_none(path: str | None = None) -> Topology | None:
    """For test gates: None when no topology file exists (the fleet
    lanes skip instead of failing)."""
    try:
        return load_topology(path)
    except FileNotFoundError:
        return None


def local_topology(*, budget_gib: float = 8.0, backing_gib: float = 8.0,
                   device: int = 0, peer_port: int = 29711) -> "Topology":
    """Zero-config world-1: one localhost member, one group ("local").
    The conductor launches the daemon as a LOCAL CHILD process (the
    HostSpec.ssh=None lane) — the child-daemon pattern at world 1."""
    import os

    host = HostSpec(name="local", peer_listen=f"127.0.0.1:{peer_port}",
                    ssh=None, repo=os.getcwd(),
                    backing_gib=backing_gib, budget_gib=budget_gib,
                    device=device)
    return Topology(conductor="local", hosts={"local": host},
                    groups={"local": GroupSpec(name="local",
                                               members=("local",),
                                               backend="hostmem")},
                    source="<local world-1>")


def local_pair_topology(*, budget_gib: float = 4.0,
                        backing_gib: float = 4.0, device: int = 0,
                        ports=(29721, 29722)) -> "Topology":
    """Two localhost members sharing one GPU over the hostmem backend
    — the same-box world-2 pattern the drills and CI ride."""
    import os

    hosts = {}
    for i, port in enumerate(ports):
        name = f"local{i}"
        hosts[name] = HostSpec(name=name,
                               peer_listen=f"127.0.0.1:{port}",
                               ssh=None, repo=os.getcwd(),
                               backing_gib=backing_gib, budget_gib=budget_gib,
                               device=device)
    return Topology(conductor="local0", hosts=hosts,
                    groups={"pair": GroupSpec(name="pair",
                                              members=("local0",
                                                       "local1"),
                                              backend="hostmem")},
                    source="<local world-2 pair>")
