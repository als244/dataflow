"""RENAMED: tools/bench_tables.py. Forwarding shim."""
import sys
import runpy

print("[m4_tables] renamed -> tools/bench_tables.py (forwarding)", file=sys.stderr)
runpy.run_path(__file__.replace("m4_tables", "bench_tables"), run_name="__main__")
