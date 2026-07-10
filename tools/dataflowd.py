"""Engine service daemon CLI: start / status / stop.

    python tools/dataflowd.py start --socket ~/.dataflow/dataflowd.sock \
        --slab-gib auto --device 0
    python tools/dataflowd.py status
    python tools/dataflowd.py stop

`start` runs in the foreground (use systemd/tmux/nohup for
backgrounding); --fake boots without CUDA for tests/dev.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataflow.service import DEFAULT_SOCKET, EngineClient, EngineConfig, Server


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    st = sub.add_parser("start")
    st.add_argument("--socket", default=DEFAULT_SOCKET)
    st.add_argument("--slab-gib", default="auto")
    st.add_argument("--device", type=int, default=0)
    st.add_argument("--kernels", default=None)
    st.add_argument("--fake", action="store_true",
                    help="CPU-only boot (no CUDA; tests/dev)")
    st.add_argument("--peer-name", default=None,
                    help="peer-plane identity (enables the NM)")
    st.add_argument("--peer-listen", default=None,
                    help="host:port for the NM's own listener")

    for name in ("status", "stop"):
        s = sub.add_parser(name)
        s.add_argument("--socket", default=DEFAULT_SOCKET)

    args = p.parse_args()
    if args.cmd == "start":
        slab = args.slab_gib if args.slab_gib == "auto" else float(args.slab_gib)
        cfg = EngineConfig(socket_path=args.socket, slab_backing_gib=slab,
                           device=args.device, kernel_set=args.kernels,
                           fake=args.fake, peer_name=args.peer_name,
                           peer_listen=args.peer_listen)
        print(f"dataflowd: listening on {args.socket} "
              f"(fake={args.fake}, slab={slab})")
        Server(cfg).serve_forever()
    elif args.cmd == "status":
        with EngineClient(args.socket, client_name="dataflowd-cli") as c:
            print(json.dumps(c.engine_status(), indent=2))
    elif args.cmd == "stop":
        with EngineClient(args.socket, client_name="dataflowd-cli") as c:
            print(c.shutdown())


if __name__ == "__main__":
    main()
