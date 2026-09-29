#!/usr/bin/env bash
# Per-GPU launcher; legacy two-argument calls retain GPU0.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
test "$#" -ge 2 && test "$#" -le 3 || { echo 'Usage: launch_prd_training.sh CONFIG TASK [GPU]'; exit 2; }
config=$1
task=$2
gpu=${3:-0}
[[ "$gpu" =~ ^[01]$ ]] && [[ "$task" =~ ^[a-zA-Z0-9_]+$ ]] || exit 2
test -f "$config"
control=/home/zhaoting.ding/disk/zhaoting.ding/DeepfakeBench/.state
lock="$control/prd_training.lock"
test "$gpu" = 0 || lock="$control/prd_training_gpu1.lock"
exec 9>"$lock"
flock -n 9 || { echo "GPU$gpu launcher already holds its lock."; exit 1; }
/usr/bin/python3 scripts/prd_gpu_guard.py "$gpu"
mkdir -p "logs/$task"
set -o noclobber
exec 8>"logs/$task/console.log"
export DEEPFAKE_DATA_ROOT=/home/zhaoting.ding/local_datasets
export PRD_PRETRAINED_ROOT=/home/zhaoting.ding/disk/zhaoting.ding/pretrained
export CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
limit=()
if test -n "${PRD_DEADLINE:-}"; then
    seconds=$(( $(date -d "$PRD_DEADLINE" +%s) - $(date +%s) ))
    test "$seconds" -gt 0 || exit 2
    limit=(timeout --signal=TERM --kill-after=120s "${seconds}s")
fi
set +e
{
    echo "Training launch: $(date -Is); config=$config; task=$task; GPU=$gpu"
    "${limit[@]}" /home/zhaoting.ding/miniconda3/envs/PRD/bin/python -u training/train.py \
        --detector_path "$config" --task_target "$task"
    code=$?
    echo "TRAINING_EXIT_CODE=$code at $(date -Is)"
    exit "$code"
} 2>&1 | tee /dev/fd/8
status=("${PIPESTATUS[@]}")
test "${status[0]}" -eq 0 || exit "${status[0]}"
exit "${status[1]}"
