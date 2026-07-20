"""Generate ONE model reference page at a chosen run shape:

    python tools/gen_model_docs/gen_model_page.py --preset glm52_mini \\
        --microbatch 8 --seq-len 8192 [--out-dir docs/models] [--family glm52]

Writes `<out-dir>/<family>/<preset>_<bs>x<seq>.md` (e.g.
`docs/models/glm52/glm52_mini_8x8K.md`). The family is resolved from
the preset classmethod name automatically; pass --family only for
ambiguous preset names (`tiny` exists on every family).

Thin wrapper over tools/gen_model_docs.py's page engine — same
content, arbitrary (microbatch, seq_len). Object tables and sizes are
pure layout arithmetic at any scale; kernel sequences are traced once
at the family's tiny preset (--no-record to skip on CPU-only boxes).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gen_model_docs import (  # noqa: E402  (tools/ sibling import)
    REPO,
    family_of_preset,
    gen_page,
    shape_tag,
)

from dataflow_training.model_families import families as F


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True,
                    help="preset classmethod name, e.g. glm52_mini")
    ap.add_argument("--microbatch", type=int, required=True)
    ap.add_argument("--seq-len", type=int, required=True)
    ap.add_argument("--family", default=None,
                    help="only needed when the preset name is ambiguous "
                         "(e.g. tiny)")
    ap.add_argument("--out-dir", default=str(REPO / "docs/models"))
    ap.add_argument("--record", action=argparse.BooleanOptionalAction,
                    default=True)
    args = ap.parse_args()

    F.load_plugins()
    family = args.family or family_of_preset(args.preset)
    if family is None:
        sys.exit(f"preset {args.preset!r} is ambiguous or unknown — "
                 f"pass --family (families: {sorted(F._FAMILIES)})")

    tag = shape_tag(args.microbatch, args.seq_len)
    fdir = Path(args.out_dir) / family
    fdir.mkdir(parents=True, exist_ok=True)
    page = gen_page(family, args.preset, args.record,
                    batch=args.microbatch, seq_len=args.seq_len)
    path = fdir / f"{args.preset}_{tag}.md"
    path.write_text(page)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
