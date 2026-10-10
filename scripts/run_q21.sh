#!/usr/bin/env bash
set -euo pipefail
cd /root/DeepfakeBench-prd-common
source /root/prd-common-activate.sh
claim=$1
test "$(git branch --show-current)" = prd-203-1-seed3407
test "$(git rev-parse origin/codex/prd-progress)" = "$claim"
state=.state/q21_t01_20261010
exec 9>.state/prd_training.lock
flock -n 9
exec 7>.state/prd_mechanism_203.lock
flock -n 7
set -C
exec >"$state/train_runner.log" 2>&1
echo $$ >"$state/train_runner.pid"
date --iso-8601=seconds >"$state/train_launched_at.txt"
set +e
code=0
for spec in "600 --budget budget" "60000 --cache cache" "1200 --smoke smoke" "54000 --arm=aligned aligned" "54000 --arm=permuted permuted" "1200 --finish finish"; do
 read -r bound option stage <<< "$spec"
 date --iso-8601=seconds >"$state/${stage}_started_at.txt"
 timeout --signal=TERM --kill-after=30s "${bound}s" "$PYTHON_BIN" -u scripts/train_q21.py "$option" --claim "$claim"
 code=$?
 echo "$code" >"$state/${stage}_exit_code.txt"
 if [ "$code" -ne 0 ]; then break; fi
done
echo "$code" >"$state/exit_code.txt"
Q21_EXIT="$code" "$PYTHON_BIN" -c 'import os,json,pathlib,datetime; p=pathlib.Path(".state/q21_t01_20261010/status.json"); q=p.with_suffix(".tmp"); c=int(os.environ["Q21_EXIT"]); q.write_text(json.dumps(dict(status="completed" if c==0 else "failed",exit_code=c,time=datetime.datetime.now().astimezone().isoformat()))); q.replace(p)'
exit "$code"
