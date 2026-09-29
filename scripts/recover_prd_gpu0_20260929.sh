#!/usr/bin/env bash
# Continue only the unstarted tail; never rerun completed F1.
set -euo pipefail
cd "$(dirname "$0")/.."
queue=prd_gpu0_recovery_20260929
mkdir -p ".state/$queue"
exec 7>".state/$queue/queue.lock"
flock -n 7
set -C
exec >".state/$queue/runner.log" 2>&1
trap 'code=$?; echo "QUEUE_EXIT_CODE=$code at $(date -Is)"' EXIT
export PRD_DEADLINE="2026-10-07 00:00:00 +0800"
/usr/bin/python3 - <<'PY'
import json, pathlib, subprocess
root=pathlib.Path('.')
for task,filename in [('prd_t3_identity_seed2026_20260926','completion.json'),('prd_f1_orig_seed2026_20260926','completion.recovered.json')]:
 p=root/'.state'/task
 assert json.loads((p/filename).read_text())['status']=='complete'
 commit=(p/'result_commit.txt').read_text().strip()
 subprocess.run(['git','cat-file','-e',commit+'^{commit}'],check=True)
assert not (root/'logs/prd_f2_residual_seed2026_20260926/console.log').exists()
assert not (root/'.state/prd_f2_residual_seed2026_20260926/completion.json').exists()
PY
task=prd_f2_residual_seed2026_20260926
echo "RECOVERY_START=$task at $(date -Is)"
set +e
bash scripts/launch_prd_training.sh training/config/detector/prd_f2_residual_seed2026.yaml "$task" 0
code=$?
set -e
/home/zhaoting.ding/miniconda3/envs/PRD/bin/python scripts/finalize_prd_run.py "$task" "$code" >".state/$task/finalize.log" 2>&1
shopt -s nullglob
summaries=(scripts/experiment_summaries/lora_"$task"_*_实验摘要.md)
test "${#summaries[@]}" -eq 1
git add -- "${summaries[0]}"
git commit -m "记录 F2 残差特征 seed=2026 实验结果" -- "${summaries[0]}"
git rev-parse HEAD >".state/$task/result_commit.txt"
printf '{"status":"complete","queue":"prd_seed2026_queue_20260926","recovered_via":"%s","git_commit":"%s"}\n' "$queue" "$(git rev-parse HEAD)" >.state/prd_seed2026_queue_20260926/completion.json
printf '{"status":"complete","queue":"%s","git_commit":"%s"}\n' "$queue" "$(git rev-parse HEAD)" >".state/$queue/completion.json"
echo "RECOVERY_COMPLETE at $(date -Is)"
# Unique continuation state preserves the earlier stopped waiting-run log.
exec bash scripts/run_prd_dual_gpu_queue.sh 0 prd_gpu0_queue_resumed_20260929
