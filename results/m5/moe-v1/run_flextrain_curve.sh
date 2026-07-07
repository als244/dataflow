#!/usr/bin/env bash
# flextrain OLMoE-7B budget curve — APPLES-TO-APPLES (Shein): both
# systems TARGET the same peak-device budget B. flextrain's usable pool
# is (max-gpu-mem-gib - leeway); leeway is shrunk from its 5 GiB default
# to LEEWAY (default 1.5) so their planner actually plans against ~B,
# and the 1 Hz nvidia-smi sampler VERIFIES where the card peak lands
# (same <=B semantics as our --device-gib). Synthetic tokens, seq 1024,
# 65,536 tok/step, full bf16.
#
# Usage: [LEEWAY=1.5] bash run_flextrain_curve.sh [budgets...]
set -u
cd /home/shein/Documents/grad_school/research/flextrain
source ~/miniconda3/etc/profile.d/conda.sh
conda activate flextrain
OUT=/home/shein/Documents/grad_school/research/dataflow/artifacts/m5/moe-v1
LEEWAY="${LEEWAY:-1.5}"
BUDGETS=("${@:-12 16 20 24 28 31}")
[ $# -eq 0 ] && BUDGETS=(12 16 20 24 28 31)

for B in "${BUDGETS[@]}"; do
  log="$OUT/flextrain-olmoe-lw${LEEWAY}-dev${B}.log"
  peaks="$OUT/flextrain-olmoe-lw${LEEWAY}-dev${B}.peak"
  : > "$peaks"
  ( while true; do
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$peaks"
      sleep 1
    done ) &
  POLL=$!
  python train.py --model models/OLMoE-7B-A1B --mode full \
    --max-seq-len 1024 --max-global-batch-tokens 65536 \
    --data-source synthetic --synthetic-seq-len 1024 --steps 6 \
    --master-dtype bfloat16 --grad-dtype bfloat16 --opt-state-dtype bfloat16 \
    --max-gpu-mem-gib "$B" --leeway-gpu-mem-gib "$LEEWAY" > "$log" 2>&1
  ec=$?
  kill $POLL 2>/dev/null
  wait $POLL 2>/dev/null
  peak=$(sort -n "$peaks" | tail -1)
  echo "dev${B}: exit=$ec peak=${peak} MiB  log=$(basename "$log")"
done
