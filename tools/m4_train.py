"""RENAMED: tools/bench_train.py (the mX milestone naming is retired).
This shim forwards and will be removed after a few sessions."""
import runpy
import sys

print("[m4_train] renamed -> tools/bench_train.py (forwarding)", file=sys.stderr)
runpy.run_path(__path__ if False else __file__.replace("m4_train", "bench_train"), run_name="__main__")
