#!/usr/bin/env bash
set -euo pipefail
cd /root/DeepfakeBench-prd-common
source /root/prd-common-activate.sh
mode=$1
claim=$2
test "$mode" = smoke || test "$mode" = train
test "$(git branch --show-current)" = prd-203-1-seed3407
test "$(git rev-parse origin/codex/prd-progress)" = "$claim"
# Only the previously recorded non-training local files may be dirty.
"$PYTHON_BIN" - <<'PY'
import subprocess
allowed={'scripts/PRD_MONITOR.md','scripts/gpu_monitor.py','scripts/test_gpu_monitor.py','scripts/run_prd_203_continue_20261003.sh'}
lines=subprocess.check_output(['git','status','--porcelain'],text=True).splitlines()
assert all(x[3:] in allowed for x in lines),lines
assert not subprocess.check_output(['git','diff','--cached','--name-only'],text=True).strip()
PY
mkdir -p .state/n01_t04_20261004
state=.state/n01_t04_20261004
exec 9>.state/prd_training.lock
flock -n 9
exec 7>.state/prd_mechanism_203.lock
flock -n 7
set -C
exec >"$state/${mode}_runner.log" 2>&1
echo $$ >"$state/${mode}_runner.pid"
date --iso-8601=seconds >"$state/${mode}_launched_at.txt"
N01_MODE="$mode" "$PYTHON_BIN" - <<'PY'
import os,hashlib,json,subprocess,pathlib
paths=['scripts/PRD_MONITOR.md','scripts/gpu_monitor.py','scripts/test_gpu_monitor.py','scripts/run_prd_203_continue_20261003.sh']
data=dict(head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),status=subprocess.check_output(['git','status','--short'],text=True),hashes={p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in paths})
with pathlib.Path('.state/n01_t04_20261004/'+os.environ['N01_MODE']+'_source_status.json').open('x') as f:json.dump(data,f,indent=2)
PY
set +e
if [ "$mode" = smoke ]; then
 timeout --signal=TERM --kill-after=30s 300s "$PYTHON_BIN" -u scripts/train_n01_t04.py --claim "$claim" --smoke
 code=$?
 echo "$code" >"$state/smoke_exit_code.txt"
 exit "$code"
fi
test "$(cat "$state/smoke_exit_code.txt")" = 0
if [ "$?" -ne 0 ]; then exit 1; fi
timeout --signal=TERM --kill-after=30s 21600s "$PYTHON_BIN" -u scripts/train_n01_t04.py --claim "$claim"
code=$?
echo "$code" >"$state/training_exit_code.txt"
if [ "$code" -eq 0 ]; then
 "$PYTHON_BIN" scripts/finalize_n01_t04.py
 code=$?
fi
echo "$code" >"$state/exit_code.txt"
N01_EXIT="$code" "$PYTHON_BIN" -c 'import os,json,pathlib,datetime; p=pathlib.Path(".state/n01_t04_20261004/status.json"); q=p.with_suffix(".tmp"); c=int(os.environ["N01_EXIT"]); q.write_text(json.dumps(dict(status="completed" if c==0 else "failed",exit_code=c,time=datetime.datetime.now().astimezone().isoformat()))); q.replace(p)'
exit "$code"
