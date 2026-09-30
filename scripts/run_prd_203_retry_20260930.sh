#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source /root/prd-common-activate.sh
queue=.state/prd_seed3407_bs16_retry_20260930
smoke=.state/prd_seed3407_bs16_queue_20260927
exec 7>"$queue/queue.lock"
flock -n 7 || exit 1
# Share the previous batch's queue lock as well as the launcher's GPU lock.
exec 6>.state/prd_seed3407_bs16_tmux_20260927/queue.lock
flock -n 6 || exit 1
test "$(git branch --show-current)" = prd-203-1-seed3407
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
    test "$(cat "$smoke/smoke_${experiment}.exit")" = 0
    grep -q FORMAL_BATCH_SMOKE_PASS "$smoke/smoke_${experiment}.log"
done
for experiment in t1 t2_blur t3_identity; do
    for attempt in 1 2 3; do
        task=prd_${experiment}_seed3407_bs16_203_retry_20260930_a${attempt}
        config=training/config/detector/prd_${experiment}_seed3407.yaml
        mkdir ".state/$task"
        "$PYTHON_BIN" scripts/prd_203_retry_preflight.py "$task" "$config"
        "$PYTHON_BIN" scripts/prd_203_event.py "$queue" "$task" start
        set +e
        bash scripts/launch_prd_training.sh "$config" "$task"
        code=$?
        set -e
        if test "$code" -eq 0; then
            # Archive or Git failure must stop; never retry a successful training run.
            "$PYTHON_BIN" scripts/finalize_prd_203_retry.py "$task" "$code" >".state/$task/finalize.log" 2>&1
            git add -- "scripts/experiment_summaries/$task"
            git -c user.name=Codex -c user.email=codex@localhost commit -m "记录203 $experiment seed3407 重跑a$attempt实验结果" -- "scripts/experiment_summaries/$task"
            git rev-parse HEAD >".state/$task/result_commit.txt"
            "$PYTHON_BIN" scripts/prd_203_event.py "$queue" "$task" archived
            break
        fi
        printf '%s\n' "$code" >".state/$task/training_exit_code.txt"
        "$PYTHON_BIN" scripts/prd_203_event.py "$queue" "$task" "$code"
        test "$attempt" -lt 3 || exit "$code"
        # A fresh task name preserves all failed outputs; the launcher rejects live trains.
        sleep 60
    done
done
task=all_complete
