#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source /root/prd-common-activate.sh
exec 9>.state/prd_training.lock
flock -n 9
exec 8>.state/p04_d02.lock
flock -n 8
state=.state/p04_d02_20261003_a2
mkdir "$state"
echo $$ > "$state/runner.pid"
exec >"$state/runner.log" 2>&1
date -Is
set +e
timeout --signal=TERM --kill-after=30s 1800 "$PYTHON_BIN" -u scripts/p04_d02.py
rc=$?
set -e
printf '%s\n' "$rc" > "$state/exit_code.txt"
date -Is > "$state/ended_at.txt"
exit "$rc"
