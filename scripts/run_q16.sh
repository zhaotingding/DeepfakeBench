#!/usr/bin/env bash
set -euo pipefail
cd /root/DeepfakeBench-prd-common
source /root/prd-common-activate.sh
claim=$1
test "$(git branch --show-current)" = prd-203-1-seed3407
test "$(git rev-parse origin/codex/prd-progress)" = "$claim"
state=.state/q16_p01_20261008
exec 9>.state/prd_training.lock
flock -n 9
exec 7>.state/prd_mechanism_203.lock
flock -n 7
set -C
exec >"$state/runner.log" 2>&1
echo $$ >"$state/runner.pid"
date --iso-8601=seconds >"$state/launched_at.txt"
set +e
timeout --signal=TERM --kill-after=30s 14400s "$PYTHON_BIN" -u scripts/train_q16.py --claim "$claim"
code=$?
echo "$code" >"$state/exit_code.txt"
Q16_EXIT="$code" "$PYTHON_BIN" -c 'import os,json,pathlib,datetime; p=pathlib.Path(".state/q16_p01_20261008/status.json"); q=p.with_suffix(".tmp"); c=int(os.environ["Q16_EXIT"]); q.write_text(json.dumps(dict(status="completed" if c==0 else "failed",exit_code=c,time=datetime.datetime.now().astimezone().isoformat()))); q.replace(p)'
exit "$code"
