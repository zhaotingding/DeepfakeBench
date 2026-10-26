#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source /root/prd-common-activate.sh
mode=$1
claim=$2
test "$mode" = smoke || test "$mode" = train
test "$(git branch --show-current)" = prd-203-1-seed3407
test "$(git rev-parse origin/codex/prd-progress)" = "$claim"
mkdir -p .state/q05_t04_20261006
state=.state/q05_t04_20261006
exec 9>.state/prd_training.lock
flock -n 9
exec 7>.state/prd_mechanism_203.lock
flock -n 7
set -C
exec >"$state/${mode}_runner.log" 2>&1
echo $$ >"$state/${mode}_runner.pid"
date --iso-8601=seconds >"$state/${mode}_launched_at.txt"
set +e
if [ "$mode" = smoke ]; then
 timeout --signal=TERM --kill-after=30s 300s "$PYTHON_BIN" -u scripts/train_q05_t04.py --claim "$claim" --smoke
 code=$?
 echo "$code" >"$state/smoke_exit_code.txt"
 exit "$code"
fi
test "$(cat "$state/smoke_exit_code.txt")" = 0
if [ "$?" -ne 0 ]; then exit 1; fi
timeout --signal=TERM --kill-after=30s 10800s "$PYTHON_BIN" -u scripts/train_q05_t04.py --claim "$claim"
code=$?
echo "$code" >"$state/training_exit_code.txt"
if [ "$code" -eq 0 ]; then
 "$PYTHON_BIN" scripts/finalize_q05_t04.py
 code=$?
fi
echo "$code" >"$state/exit_code.txt"
N02_EXIT="$code" "$PYTHON_BIN" -c 'import os,json,pathlib,datetime; p=pathlib.Path(".state/q05_t04_20261006/status.json"); q=p.with_suffix(".tmp"); c=int(os.environ["N02_EXIT"]); q.write_text(json.dumps(dict(status="completed" if c==0 else "failed",exit_code=c,time=datetime.datetime.now().astimezone().isoformat()))); q.replace(p)'
exit "$code"
