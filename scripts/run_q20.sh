#!/usr/bin/env bash
set -euo pipefail
cd /root/DeepfakeBench-prd-common
source /root/prd-common-activate.sh
claim=$1
test "$(git branch --show-current)" = prd-203-1-seed3407
test "$(git rev-parse origin/codex/prd-progress)" = "$claim"
state=.state/q20_t01_20261009
exec 9>.state/prd_training.lock
flock -n 9
exec 7>.state/prd_mechanism_203.lock
flock -n 7
set -C
exec >"$state/train_runner.log" 2>&1
echo $$ >"$state/train_runner.pid"
date --iso-8601=seconds >"$state/train_launched_at.txt"
set +e
timeout --signal=TERM --kill-after=30s 1200s "$PYTHON_BIN" -u scripts/train_q20.py --smoke --claim "$claim"
code=$?
echo "$code" >"$state/smoke_exit_code.txt"
if [ "$code" -eq 0 ]; then
    for arm in original_teacher prd_teacher; do
        date --iso-8601=seconds >"$state/${arm}_started_at.txt"
        timeout --signal=TERM --kill-after=30s 85000s "$PYTHON_BIN" -u scripts/train_q20.py --arm "$arm" --claim "$claim"
        code=$?
        echo "$code" >"$state/${arm}_exit_code.txt"
        if [ "$code" -ne 0 ]; then break; fi
    done
fi
if [ "$code" -eq 0 ]; then
    timeout --signal=TERM --kill-after=30s 1200s "$PYTHON_BIN" -u scripts/train_q20.py --finish --claim "$claim"
    code=$?
fi
echo "$code" >"$state/exit_code.txt"
Q20_EXIT="$code" "$PYTHON_BIN" -c 'import os,json,pathlib,datetime; p=pathlib.Path(".state/q20_t01_20261009/status.json"); q=p.with_suffix(".tmp"); c=int(os.environ["Q20_EXIT"]); q.write_text(json.dumps(dict(status="completed" if c==0 else "failed",exit_code=c,time=datetime.datetime.now().astimezone().isoformat()))); q.replace(p)'
exit "$code"
