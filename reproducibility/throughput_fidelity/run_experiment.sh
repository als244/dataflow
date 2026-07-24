#!/bin/bash
# One command, any box:
#
#     bash reproducibility/throughput_fidelity/run_experiment.sh
#
# Nothing about the machine is hard-coded. The probe reads this device and this
# host's real limits and writes env.json (preset, budget ladder, backing
# ceiling, geometry axes); the prediction pass plans every candidate and records
# what the planner cannot fit; the selector picks a representative cell per
# behaviour regime plus one budget spine; only those are run on real hardware.
#
# Env overrides:  PYTHON=/path/to/python   PRESET=l3_1b   OPTS=adamw,muon
#                 TARGET_CELLS=18          STEPS=6
set -uo pipefail
SUB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SUB/../.." && pwd)"
PY="${PYTHON:-python}"
OPTS="${OPTS:-adamw,muon}"
TARGET_CELLS="${TARGET_CELLS:-18}"
cd "$REPO"; export PYTHONPATH=src
LOG="$SUB/logs"; D="$SUB/data"; mkdir -p "$LOG" "$D"
say(){ echo "[$(date +%H:%M:%S)] $*"; }
jget(){ "$PY" -c "import json;print(json.load(open('$SUB/env.json'))['$1'])"; }
jlist(){ "$PY" -c "import json;print(' '.join(str(x) for x in json.load(open('$SUB/env.json'))['$1']))"; }

say "host $(hostname)  repo $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?')"
"$PY" -c "from dataflow_training.run.profiling import PROFILE_CACHE_REV as R;print('profile cache rev',R)"

# --- P0: what can this machine sweep? ---
say "P0 environment probe"
"$PY" "$SUB/env_probe.py" ${PRESET:+--preset "$PRESET"} 2>&1 | tee "$LOG/env_probe.log" | tail -20 || exit 1
PRESET_USED=$(jget preset); STEPS="${STEPS:-$(jget steps_per_cell)}"
BACKING=$(jlist backings | tr ' ' ',')
BUDGETS=$(jlist budgets | tr ' ' ','); SEQS=$(jlist seqs)
TROUNDS=$(jlist t_rounds | tr ' ' ','); TSTEPS=$(jlist t_steps | tr ' ' ',')
say "preset=$PRESET_USED budgets=$BUDGETS backings=$BACKING steps/cell=$STEPS"

# --- P1: host<->device bandwidth on record ---
say "P1 pcie calibration"
"$PY" "$SUB/pcie_calib.py" > "$LOG/pcie_calib.log" 2>&1 && tail -3 "$LOG/pcie_calib.log" || say "  (skipped)"

# --- P2: measured-cost predictions = the feasibility pass, per-seq chunks ---
for opt in ${OPTS//,/ }; do
  say "P2 predictions ($opt) — every cost profiled on this GPU"
  : > "$D/predict_measured_$opt.jsonl"
  for seq in $SEQS; do
    "$PY" "$SUB/sweep.py" --mode predict-measured --preset "$PRESET_USED" --opt "$opt" \
      --seq "$seq" --t-round "$TROUNDS" --t-step "$TSTEPS" --budget "$BUDGETS" \
      --backing-gib "$BACKING" --out "$D/predict_measured_$opt.jsonl" \
      >> "$LOG/predict_measured_$opt.log" 2>&1 \
      && say "  seq$seq ok ($(wc -l <"$D/predict_measured_$opt.jsonl") rows)" \
      || say "  seq$seq FAILED (see $LOG/predict_measured_$opt.log)"
  done
done

# --- P2b: which of the survivors are worth real runs? ---
say "P2b selecting cells to run"
"$PY" "$SUB/select_cells.py" --opts "$OPTS" --target "$TARGET_CELLS" 2>&1 | tee "$LOG/select_cells.log" | tail -30

# --- P3: real engine runs (prediction column = the same profiled costs) ---
for opt in ${OPTS//,/ }; do
  say "P3 measuring ($opt)"
  : > "$D/measure_$opt.jsonl"
  "$PY" "$SUB/sweep.py" --mode measure --preset "$PRESET_USED" --opt "$opt" --steps "$STEPS" \
    --cells "$SUB/cells.json" --backing-gib "$BACKING" --out "$D/measure_$opt.jsonl" \
    >> "$LOG/measure_$opt.log" 2>&1 \
    && say "  $opt done ($(wc -l <"$D/measure_$opt.jsonl") rows)" \
    || say "  $opt FAILED (see $LOG/measure_$opt.log)"
done

# --- P4: the shipped bench commands, on this preset, measured-cost ---
say "P4 shipped-command validation"
SPINE=$("$PY" -c "
import json;c=[x for x in json.load(open('$SUB/cells.json')) if 'budget_spine' in x['spines']]
c=c or json.load(open('$SUB/cells.json'))
b=sorted({x['budget'] for x in c});print(f\"{c[0]['seq']} {c[0]['t_round']} {c[0]['t_step']} {','.join(str(x) for x in b[:2])}\")")
read -r SQ TR TS BD <<<"$SPINE"
{
  echo "########## predict_step.py --measured ##########"
  "$PY" tools/bench/predict_step.py --preset "$PRESET_USED" --measured \
    --t-round "$TR" --tokens-step "$TS" --budget "$BD" --seq-len "$SQ" --backing "${BACKING%%,*}"
  echo; echo "########## measure_step.py --measured-plan ##########"
  "$PY" tools/bench/measure_step.py --preset "$PRESET_USED" --measured-plan \
    --t-round "$TR" --tokens-step "$TS" --budget "$BD" --seq-len "$SQ" \
    --backing-gib "${BACKING%%,*}" --steps "$STEPS"
} > "$LOG/shipped_bench.log" 2>&1 && say "P4 ok" || say "P4 had errors (see $LOG/shipped_bench.log)"

say "DONE — data in $D, logs in $LOG"
"$PY" "$SUB/analyze.py" 2>&1 | tail -40
