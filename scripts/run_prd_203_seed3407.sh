#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source /root/prd-common-activate.sh
queue=.state/prd_seed3407_bs16_queue_20260927
exec 7>"$queue/queue.lock"
flock -n 7 || exit 1
test "$(git branch --show-current)" = prd-203-1-seed3407
test -z "$(git status --porcelain)"
set -C
exec >"$queue/runner.log" 2>&1
task=none
finish() {
    code=$?
    trap - EXIT
    "$PYTHON_BIN" scripts/prd_203_event.py "$queue" "$task" "$code"
    exit "$code"
}
trap finish EXIT
echo "$$" >"$queue/runner.pid"
for experiment in t1 t2_blur t3_identity; do
    test "$(cat "$queue/smoke_${experiment}.exit")" = 0
    grep -q FORMAL_BATCH_SMOKE_PASS "$queue/smoke_${experiment}.log"
done
for experiment in t1 t2_blur t3_identity; do
    task=prd_${experiment}_seed3407_bs16_203_20260927
    mkdir ".state/$task"
    "$PYTHON_BIN" scripts/prd_203_event.py "$queue" "$task" start
    set +e
    bash scripts/launch_prd_training.sh "training/config/detector/prd_${experiment}_seed3407.yaml" "$task"
    code=$?
    set -e
    test "$code" -eq 0 || exit "$code"
    "$PYTHON_BIN" scripts/finalize_prd_203.py "$task" "$code" >".state/$task/finalize.log" 2>&1
    git add -- "scripts/experiment_summaries/$task"
    git -c user.name=Codex -c user.email=codex@localhost commit -m "记录203 $experiment seed3407 BS16实验结果" -- "scripts/experiment_summaries/$task"
    git rev-parse HEAD >".state/$task/result_commit.txt"
    "$PYTHON_BIN" scripts/prd_203_event.py "$queue" "$task" archived
done
task=all_complete
