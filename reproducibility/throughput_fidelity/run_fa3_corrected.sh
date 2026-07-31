#!/usr/bin/env bash
# Re-run the FA3 throughput/sim-fidelity campaign on CORRECTED code.
#
# Same FlashAttention-3 kernels and the same grid as results_h100_fa3 —
# recovered from that run's env.json and stage logs, not guessed — so the
# two are directly comparable. What differs is the code underneath:
# measured task pricing with no roofline defaults, per-device hardware
# resolution, automatic profile-cache invalidation, and production-init
# profiling fills. The question this answers is whether the simulated
# frontier is now the real frontier.
#
#   bash reproducibility/throughput_fidelity/run_fa3_corrected.sh
#   bash reproducibility/throughput_fidelity/run_fa3_corrected.sh --smoke
#   bash reproducibility/throughput_fidelity/run_fa3_corrected.sh --stages report
#
# Run from the repo root on a GPU node. Any extra arguments are passed
# through to run_experiment.py, so --resume / --stages / --budgets work
# as documented there.
#
# BUDGET: the full grid is 84 frontier cells x 2 optimizers. FA3's own
# logs put the measure stage at ~224 min of stepping per optimizer
# (eta ~353 min with per-cell overhead), plus profile and predict. Plan
# on 12-14 h; --smoke is ~10 min and exists to prove all seven stages
# run before committing to that.
set -o pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1

PY=${PY:-python}
HERE=reproducibility/throughput_fidelity
SMOKE=0
EXTRA=()
for a in "$@"; do
    if [ "$a" = "--smoke" ]; then SMOKE=1; else EXTRA+=("$a"); fi
done

if [ "$SMOKE" = "1" ]; then
    # one sequence, two budgets, 2 steps: a shape test, not a measurement
    RESULTS=$HERE/results_smoke
    GRID=(--seqs 1024 --t-rounds 32768 --t-steps 65536
          --budgets 18,35 --steps 2)
    echo "SMOKE run -> $RESULTS (shape test only; numbers are not results)"
else
    RESULTS=$HERE/results_h100_fa3_corrected
    # FA3's grid verbatim. --all-frontier takes every frontier cell, which
    # makes select's --target inert, so it is not passed.
    GRID=(--seqs 1024,2048,4096,8192
          --t-rounds 1024,2048,4096,8192,16384,32768,65536
          --t-steps 65536,131072,262144
          --budgets 8.8,12,18,25,35,50,67.3
          --steps 6)
    echo "FULL FA3-corrected campaign -> $RESULTS  (expect 12-14 h)"
fi

# a leftover daemon from a previous stage poisons both timing and memory
# peaks; sweep before starting (bracketed so this line cannot match itself)
pkill -f '[d]ataflow''d' || true
rm -f /tmp/dataflow-test-*.sock
sleep 2

echo "commit: $(git log --oneline -1)"
echo "profiles: $(ls artifacts/profile-cache/profiles-*.json 2>/dev/null | wc -l) cached \
(--fresh-profiles re-measures regardless)"

set -x
$PY $HERE/run_experiment.py \
    --preset llama3_8b \
    --opts adamw,muon \
    "${GRID[@]}" \
    --backing-gib 128 \
    --all-frontier \
    --fresh-profiles \
    --results "$RESULTS" \
    "${EXTRA[@]}"
RC=$?
set +x

echo "CAMPAIGN_RC=$RC  results in $RESULTS"
echo "compare against $HERE/results_h100_fa3 (same grid, pre-fix code)"
exit $RC
