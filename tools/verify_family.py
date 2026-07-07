"""Verify a model family's mathematical correctness, end to end.

One command per family. Runs the family's canonical test module and
audits it for the canonical gate coverage — the module (not this tool)
owns tolerances and family specifics, because correctness envelopes are
family knowledge (e.g. sign-lottery ``field_atol`` params).

What the canonical module covers, per docs/extending.md:

- PER OP: launch form vs reference forward; hand-written backward vs
  autograd on the reference (bf16-honest rel_l2).
- PER TASK (fwd / recompute / bwd): per-kind block ladders — dx and
  EVERY packed dW field vs the golden's autograd, for save+bwd AND
  recompute+bwd (recompute-equivalence, byte-compared int/meta fields),
  gradient-accumulation semantics.
- PER MODEL: ``check_model_step`` through the REAL engine — loss, every
  final parameter field, and optimizer state vs the golden model after
  full fwd+bwd+optimizer step(s); at multiple budgets (plan-invariance:
  different plans, identical math) and under forced recompute; poison +
  interleave stress; measured-costs-replan (profiling E2E); fixed-seed
  determinism twice (byte-compare).

Usage:
    python tools/verify_family.py --family glm52
    python tools/verify_family.py --list
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# the canon: (label, regex over the family test module source)
CANON = [
    ("op pins (launch vs reference / bwd vs autograd)", r"reference|autograd"),
    ("per-kind block ladder (fwd+bwd)", r"check_block_backward|_ladder|block_backward"),
    ("recompute-equivalence", r"recompute"),
    ("grad-accumulation semantics", r"accum|ga2"),
    ("model step vs golden (params + opt state)", r"model_step"),
    ("plan-invariance / multi-budget", r"plan_invariance|budgets|plan_inv"),
    ("poison-on-free stress", r"poison"),
    ("interleave stress", r"interleav"),
    ("profiling E2E (measured-costs replan)", r"measured_costs|profil"),
    ("determinism (same seed, twice)", r"determinism|bitwise|byte"),
]


# older families keep op/ladder coverage in shared modules — scanned as
# part of the family's audit surface
EXTRA_MODULES = {
    "dsv3": ["tests/tasks/test_mla_math.py", "tests/tasks/test_moe_math.py"],
    "dsv32": ["tests/tasks/test_dsa_math.py", "tests/tasks/test_moe_math.py"],
    "glm52": ["tests/tasks/test_dsa_math.py", "tests/tasks/test_moe_math.py"],
    "olmoe": ["tests/tasks/test_moe_math.py"],
    "qwen3moe": ["tests/tasks/test_moe_math.py"],
    "qwen35moe": ["tests/tasks/test_moe_math.py"],
}
# fleet-level gates: exercised through the shared engine/runtime tests
# (they run real programs); a family missing one in ITS module gets a
# [~] marker, not a failure
FLEET_MODULES = [
    "tests/runtime/test_cuda_backend.py",
    "tests/runtime/test_placement.py",
    "tests/training/test_profiling.py",
]
FLEET_OK = {"poison-on-free stress", "interleave stress",
            "profiling E2E (measured-costs replan)",
            "determinism (same seed, twice)"}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", help="family name (see --list)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--audit-only", action="store_true",
                    help="coverage audit without running the tests")
    args = ap.parse_args()

    if args.list or not args.family:
        from dataflow.training import families as F

        print("families:", ", ".join(sorted(F._FAMILIES)))
        return

    mod = REPO / f"tests/tasks/test_{args.family}_math.py"
    if not mod.exists():
        sys.exit(f"MISSING {mod} — a family without its canonical test "
                 f"module is unverified. Copy the NEWEST family's module "
                 f"as the template (docs/extending.md, ladder canon).")

    src = mod.read_text()
    for extra in EXTRA_MODULES.get(args.family, ()):
        src += (REPO / extra).read_text()
    fleet_src = "".join((REPO / f).read_text() for f in FLEET_MODULES)
    print(f"canon coverage of {mod.name} (+ shared modules):")
    missing = []
    for label, pat in CANON:
        if re.search(pat, src, re.I):
            print(f"  [x] {label}")
        elif label in FLEET_OK and re.search(pat, fleet_src, re.I):
            print(f"  [~] {label} (fleet-level via shared engine tests)")
        else:
            print(f"  [ ] {label}")
            missing.append(label)

    trip = (REPO / "tests/training/test_lowering_stability.py").read_text()
    pinned = args.family in trip
    print(f"  [{'x' if pinned else ' '}] lowering tripwire hash pinned "
          f"(test_lowering_stability.py)")
    if not pinned:
        missing.append("lowering tripwire")

    if args.audit_only:
        sys.exit(f"missing canon gates: {missing}" if missing else None)

    print(f"\nrunning pytest {mod.name} ...")
    r = subprocess.run([sys.executable, "-m", "pytest", str(mod),
                        str(REPO / "tests/training/test_lowering_stability.py"),
                        "-q", "-p", "no:warnings"], cwd=REPO)
    print()
    if r.returncode != 0:
        sys.exit(f"FAILED: {mod.name}")
    if missing:
        sys.exit(f"tests green but canon gates MISSING: {missing}")
    print(f"{args.family}: canonical module green, full canon coverage.")


if __name__ == "__main__":
    main()
