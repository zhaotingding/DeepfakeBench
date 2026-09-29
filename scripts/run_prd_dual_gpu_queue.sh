#!/usr/bin/env bash
# Finite split experiment matrix authorized on 2026-09-29.
set -euo pipefail
cd "$(dirname "$0")/.."
gpu=$1
[[ "$gpu" =~ ^[01]$ ]] || exit 2
queue=${2:-prd_gpu${gpu}_queue_20260929}
[[ "$queue" =~ ^prd_gpu[01]_queue_[a-zA-Z0-9_]+$ ]] || exit 2
mkdir -p ".state/$queue" logs/RealTime
exec 7>".state/$queue/queue.lock"
flock -n 7
set -C
exec >".state/$queue/runner.log" 2>&1
trap 'code=$?; echo "QUEUE_EXIT_CODE=$code at $(date -Is)"' EXIT
export PRD_DEADLINE="2026-10-07 00:00:00 +0800"
if test "$gpu" = 0; then
    echo "WAITING_FOR=prd_seed2026_queue_20260926"
    while ! test -f .state/prd_seed2026_queue_20260926/completion.json; do
        kill -0 1251186 || { echo "PREVIOUS_QUEUE_STOPPED_WITHOUT_COMPLETION"; exit 1; }
        test "$(date +%s)" -lt "$(date -d "$PRD_DEADLINE" +%s)" || exit 2
        sleep 60
    done
    /usr/bin/python3 -c 'import json; assert json.load(open(".state/prd_seed2026_queue_20260926/completion.json"))["status"] == "complete"'
    experiments=(f1_orig_seed42 f2_residual_seed42)
else
    experiments=(r2_abs_diff_seed2026 t1_seed42 t2_blur_seed42 t3_identity_seed42 r2_abs_diff_seed42)
fi
for experiment in "${experiments[@]}"; do
    test "$(date +%s)" -lt "$(date -d "$PRD_DEADLINE" +%s)" || { echo RESOURCE_WINDOW_ENDED; exit 2; }
    task=prd_${experiment}_gpu${gpu}_20260929
    config=training/config/detector/prd_${experiment}.yaml
    mkdir ".state/$task"
    echo "QUEUE_START=$task at $(date -Is) commit=$(git rev-parse HEAD)"
    set +e
    bash scripts/launch_prd_training.sh "$config" "$task" "$gpu"
    training_code=$?
    set -e
    /home/zhaoting.ding/miniconda3/envs/PRD/bin/python scripts/finalize_prd_run.py "$task" "$training_code" >".state/$task/finalize.log" 2>&1
    shopt -s nullglob
    summaries=(scripts/experiment_summaries/lora_"$task"_*_实验摘要.md)
    test "${#summaries[@]}" -eq 1
    git add -- "${summaries[0]}"
    git commit -m "记录 GPU$gpu ${experiment} 实验结果" -- "${summaries[0]}"
    git rev-parse HEAD >".state/$task/result_commit.txt"
    echo "QUEUE_COMMITTED=$task at $(date -Is) commit=$(git rev-parse HEAD)"
done
printf '{"status":"complete","queue":"%s","git_commit":"%s"}\n' "$queue" "$(git rev-parse HEAD)" >".state/$queue/completion.json"
echo QUEUE_COMPLETE_AND_COMMITTED
