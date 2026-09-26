#!/usr/bin/env bash
# Finite paired seed-2026 queue. Archive and commit before advancing.
set -euo pipefail
cd /home/zhaoting.ding/disk/zhaoting.ding/DeepfakeBench
queue=prd_seed2026_queue_20260926
set -C
exec > ".state/$queue/runner.log" 2>&1
exec 7>".state/$queue/queue.lock"
flock -n 7
trap 'code=$?; echo "QUEUE_EXIT_CODE=$code at $(date -Is)"' EXIT
for experiment in t3_identity f1_orig f2_residual; do
    task=prd_${experiment}_seed2026_20260926
    config=training/config/detector/prd_${experiment}_seed2026.yaml
    echo "QUEUE_START=$task at $(date -Is) commit=$(git rev-parse HEAD)"
    set +e
    bash scripts/launch_prd_training.sh "$config" "$task"
    training_code=$?
    set -e
    /home/zhaoting.ding/miniconda3/envs/PRD/bin/python scripts/finalize_prd_run.py "$task" "$training_code" > ".state/$task/finalize.log" 2>&1
    shopt -s nullglob
    summaries=(scripts/experiment_summaries/lora_"$task"_*_实验摘要.md)
    test "${#summaries[@]}" -eq 1
    git add -- "${summaries[0]}"
    git commit -m "Record $experiment seed 2026 result" -- "${summaries[0]}"
    git rev-parse HEAD > ".state/$task/result_commit.txt"
    echo "QUEUE_COMMITTED=$task at $(date -Is) commit=$(git rev-parse HEAD)"
done
printf '{"status":"complete","queue":"%s","git_commit":"%s"}\n' "$queue" "$(git rev-parse HEAD)" > ".state/$queue/completion.json"
echo "QUEUE_COMPLETE_AND_COMMITTED"
