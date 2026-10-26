#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
test "$(git branch --show-current)" = prd-203-1-seed3407
test "$(git rev-parse origin/codex/prd-progress)" = "$1"
exec 9>.state/prd_mechanism_203.lock
flock -n 9
state=.state/p03_d01_20261003
mkdir "$state"
exec >"$state/runner.log" 2>&1
echo $$ >"$state/runner.pid"
date --iso-8601=seconds >"$state/launched_at.txt"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
set +e
timeout --signal=TERM --kill-after=10s 300s /root/miniconda3/envs/prd-common/bin/python scripts/p03_d01.py --claim "$1" --out "$PWD/$state"
code=$?
echo "$code" >"$state/analysis_exit_code.txt"
if [ "$code" -eq 0 ]; then
    timeout --signal=TERM --kill-after=10s 300s /root/miniconda3/envs/prd-common/bin/python scripts/verify_p03_d01.py
    code=$?
fi
echo "$code" >"$state/exit_code.txt"
date --iso-8601=seconds >"$state/ended_at.txt"
exit "$code"
