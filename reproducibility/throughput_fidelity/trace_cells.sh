#!/usr/bin/env bash
# Trace a few campaign cells executing their OWN saved plans, on the H100.
#
#   bash trace_cells.sh <results-dir> <out-dir>
#
# The question these cells answer: after subtracting inter-task dispatch at a
# CONSTANT 324.9 us/task, a residual remains, and it tracks TOKENS/ROUND
# (8K:-0.24%  16K:-0.54%  32K:+1.57%) rather than budget — the budget
# correlation was confounded, since tight budgets favour 32K rounds.
#
# The two headline cells are a near-control: same sequence, nearly the same
# task count (403 vs 413), so a constant per-task gap predicts the SAME
# absolute dispatch cost — yet their residuals differ by 2.8%. If the measured
# per-task gap turns out to scale with task size, the "residual" is just a
# wrong constant in the dispatch model and not a separate mechanism at all.
set -uo pipefail
RES=${1:?results dir, e.g. results_h100_fa3_corrected}
OUT=${2:-traces_plan_matched}
PY=${PY:-$HOME/.conda/envs/dataflow/bin/python}
NSYS=${NSYS:-nsys}
TRACE="cuda,nvtx,osrt,cublas,cudnn"
mkdir -p "$OUT"

# seq  t_round  t_step  budget  steps  label
CELLS=(
  "8192 32768 131072 12  6 tr32K_resid_plus2.07"
  "8192 16384  65536 8.8 6 tr16K_resid_minus0.73"
  "8192 32768  65536 12  6 tr32K_short_resid_plus1.83"
)

for c in "${CELLS[@]}"; do
  read -r SEQ TR TS BUD STEPS LABEL <<<"$c"
  NAME="s${SEQ}_tr${TR}_ts${TS}_b${BUD}_${LABEL}"
  echo "=== $NAME  $(date +%H:%M:%S)"
  # a previous cell's daemon must never bleed into this trace
  pkill -f '[d]ataflowd' 2>/dev/null
  sleep 2
  "$NSYS" profile --trace="$TRACE" \
      --capture-range=cudaProfilerApi --capture-range-end=stop \
      --force-overwrite true -o "$OUT/$NAME" \
      "$PY" "$(dirname "$0")/trace_cell.py" \
        --results "$RES" --opt adamw --seq "$SEQ" --t-round "$TR" \
        --t-step "$TS" --budget "$BUD" --steps "$STEPS" --warmup 2 \
        --out "$OUT/$NAME.json" 2>&1 | tail -12
  echo "    rc=$? $(ls -la "$OUT/$NAME.nsys-rep" 2>/dev/null | awk '{print $5}')"
done
echo "DONE $(date +%H:%M:%S)"
ls -la "$OUT"
