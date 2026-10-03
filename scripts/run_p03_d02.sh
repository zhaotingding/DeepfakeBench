#!/usr/bin/env bash
set -euo pipefail
cd /root/DeepfakeBench-prd-common
source /root/prd-common-activate.sh
test "$(git branch --show-current)" = prd-203-1-seed3407
claim="$1"
test "$(git rev-parse origin/codex/prd-progress)" = "$claim"
git show "$claim:PRD实验总表.md" | grep -q 'P03-D02'
exec 198>.state/prd_training.lock
flock -n 198
exec 199>.state/prd_mechanism_203.lock
flock -n 199
q=.state/p03_d02_queue_20261003
mkdir "$q"
echo $$ > "$q/runner.pid"
date -Iseconds > "$q/started_at.txt"
exec > "$q/runner.log" 2>&1
set +e
timeout --signal=TERM --kill-after=30s 1800 "$PYTHON_BIN" -u scripts/p03_d02.py --claim "$claim"
rc=$?
echo "$rc" > "$q/diagnostic_exit_code.txt"
if [ "$rc" -eq 0 ]; then
  "$PYTHON_BIN" scripts/finalize_p03_d02.py
  rc=$?
fi
echo "$rc" > "$q/exit_code.txt"
date -Iseconds > "$q/ended_at.txt"
exit "$rc"
