"""Record the exact non-training changes allowed for the authorized retry."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ALLOWED = {' M scripts/PRD_MONITOR.md', '?? scripts/gpu_monitor.py', '?? scripts/test_gpu_monitor.py'}
status = subprocess.check_output(['git', 'status', '--porcelain'], text=True).splitlines()
assert set(status) <= ALLOWED, f'Unexpected working-tree changes: {status}'
record = dict(git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              git_status=status,
              changed_file_sha256={line[3:]: hashlib.sha256(Path(line[3:]).read_bytes()).hexdigest() for line in status},
              config_sha256=hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest())
with (Path('.state') / sys.argv[1] / 'source_status.json').open('x') as f:
    json.dump(record, f, indent=2)
