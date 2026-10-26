#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source /root/prd-common-activate.sh
claim=$1
test "$(git branch --show-current)" = prd-203-1-seed3407
test "$(git rev-parse origin/codex/prd-progress)" = "$claim"
state=.state/q17_d01_20261009_v2
exec 9>.state/prd_training.lock
flock -n 9
exec 7>.state/prd_mechanism_203.lock
flock -n 7
set -C
exec >"$state/runner.log" 2>&1
echo $$ >"$state/runner.pid"
date --iso-8601=seconds >"$state/launched_at.txt"
set +e
timeout --signal=TERM --kill-after=30s 7000s "$PYTHON_BIN" -u scripts/analyze_q17_v2.py --claim "$claim"
code=$?
echo "$code" >"$state/exit_code.txt"
Q17_EXIT="$code" "$PYTHON_BIN" -c 'import os,json,pathlib,datetime; p=pathlib.Path(".state/q17_d01_20261009_v2/status.json"); q=p.with_suffix(".tmp"); c=int(os.environ["Q17_EXIT"]); q.write_text(json.dumps(dict(status="completed" if c==0 else "failed",exit_code=c,time=datetime.datetime.now().astimezone().isoformat()))); q.replace(p)'
exit "$code"
