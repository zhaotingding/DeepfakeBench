#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source /root/prd-common-activate.sh
claim=$1
test "$(git branch --show-current)" = prd-203-1-seed3407
test "$(git rev-parse origin/codex/prd-progress)" = "$claim"
state=.state/q12_t01_20261008_24h_v2
exec 9>.state/prd_training.lock
flock -n 9
exec 7>.state/prd_mechanism_203.lock
flock -n 7
set -C
exec >"$state/train_runner.log" 2>&1
echo $$ >"$state/train_runner.pid"
date --iso-8601=seconds >"$state/train_launched_at.txt"
set +e
timeout --signal=TERM --kill-after=30s 600s "$PYTHON_BIN" -u scripts/train_q12_24h.py --smoke --claim "$claim"
code=$?
echo "$code" >"$state/smoke_exit_code.txt"
if [ "$code" -eq 0 ]; then
    date --iso-8601=seconds >"$state/formal_started_at.txt"
    timeout --signal=TERM --kill-after=30s 85200s "$PYTHON_BIN" -u scripts/train_q12_24h.py --claim "$claim"
    code=$?
fi
echo "$code" >"$state/exit_code.txt"
Q12_EXIT="$code" "$PYTHON_BIN" -c 'import os,json,pathlib,datetime; p=pathlib.Path(".state/q12_t01_20261008_24h_v2/status.json"); q=p.with_suffix(".tmp"); c=int(os.environ["Q12_EXIT"]); q.write_text(json.dumps(dict(status="completed" if c==0 else "failed",exit_code=c,time=datetime.datetime.now().astimezone().isoformat()))); q.replace(p)'
exit "$code"
