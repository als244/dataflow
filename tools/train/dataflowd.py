"""Engine service daemon CLI: start / status / stop.

    python tools/train/dataflowd.py start --socket ~/.dataflow/dataflowd.sock \
        --backing-gib auto --device 0
    python tools/train/dataflowd.py status
    python tools/train/dataflowd.py stop

`start` runs in the foreground (use systemd/tmux/nohup for
backgrounding); --fake boots without CUDA for tests/dev.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dataflow.service import DEFAULT_SOCKET, EngineClient, EngineConfig, Server


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    st = sub.add_parser("start")
    st.add_argument("--socket", default=DEFAULT_SOCKET)
    st.add_argument("--backing-gib", default="auto")
    st.add_argument("--device", type=int, default=0)
    st.add_argument("--kernels", default=None)
    st.add_argument("--fake", action="store_true",
                    help="CPU-only boot (no CUDA; tests/dev)")
    st.add_argument("--peer-name", default=None,
                    help="peer-plane identity (enables the NM)")
    st.add_argument("--peer-listen", default=None,
                    help="host:port for the NM's own listener")
    st.add_argument("--peer-rdma-device", default=None,
                    help="HCA for the rdma-host transport (e.g. mlx5_1)")
    st.add_argument("--plugin", action="append", default=[],
                    help="module to import before boot (self-registers "
                         "resolver kinds and/or families; repeatable — "
                         "the daemon twin of the tools' --plugin)")
    st.add_argument("--no-default-workloads", action="store_true",
                    help="skip the default dataflow_training registration "
                         "(bare engine daemon; register kinds via --plugin)")

    for name in ("status", "stop"):
        s = sub.add_parser(name)
        s.add_argument("--socket", default=DEFAULT_SOCKET)

    args = p.parse_args()
    if args.cmd == "start":
        if not args.no_default_workloads:
            # default workload: model-family programs resolve out of the
            # box (register_program_resolver("model_family", ...));
            # --no-default-workloads boots a bare engine
            from dataflow_training.register import register_all

            register_all()
        if args.plugin:
            from dataflow_training.model_families.families import load_plugins

            load_plugins(explicit=args.plugin)
        backing = (args.backing_gib if args.backing_gib == "auto"
                   else float(args.backing_gib))
        cfg = EngineConfig(socket_path=args.socket, slab_backing_gib=backing,
                           device=args.device, kernel_set=args.kernels,
                           fake=args.fake, peer_name=args.peer_name,
                           peer_listen=args.peer_listen,
                           peer_rdma_device=args.peer_rdma_device)
        print(f"dataflowd: listening on {args.socket} "
              f"(fake={args.fake}, backing={backing})")
        Server(cfg).serve_forever()
    elif args.cmd == "status":
        with EngineClient(args.socket, client_name="dataflowd-cli") as c:
            print(json.dumps(c.engine_status(), indent=2))
    elif args.cmd == "stop":
        with EngineClient(args.socket, client_name="dataflowd-cli") as c:
            print(c.shutdown())


if __name__ == "__main__":
    main()
