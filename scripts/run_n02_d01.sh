#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source /root/prd-common-activate.sh
q=.state/n02_d01_queue_20261003
mkdir "$q"
echo $$ > "$q/runner.pid"
date -Iseconds > "$q/started_at.txt"
exec > "$q/runner.log" 2>&1
set +e
timeout --signal=TERM --kill-after=30s 1800 "$PYTHON_BIN" -u scripts/n02_d01.py --claim "$1"
rc=$?
echo "$rc" > "$q/exit_code.txt"
date -Iseconds > "$q/ended_at.txt"
exit "$rc"
