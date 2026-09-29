"""Reject an occupied physical GPU without blocking the other card."""
import os
import subprocess
import sys
from pathlib import Path

gpu = sys.argv[1]
assert gpu in {"0", "1"}
uuid = subprocess.check_output(
    ["nvidia-smi", "-i", gpu, "--query-gpu=uuid", "--format=csv,noheader"],
    text=True).strip()
apps = subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
    text=True)
busy = [line for line in apps.splitlines() if uuid in line]
# Catch launch-to-CUDA initialization gaps in same-user training processes.
for proc in Path("/proc").iterdir():
    if not proc.name.isdigit():
        continue
    try:
        if proc.stat().st_uid != os.getuid():
            continue
        args = (proc / "cmdline").read_bytes().split(b"\0")
        if not any(a.endswith(b"/train.py") or a == b"train.py" for a in args):
            continue
        env = (proc / "environ").read_bytes().split(b"\0")
        visible = next((x.split(b"=", 1)[1].decode() for x in env
                        if x.startswith(b"CUDA_VISIBLE_DEVICES=")), None)
        if visible is None or gpu in visible.split(",") or uuid in visible.split(","):
            busy.append("training PID=" + proc.name)
    except (FileNotFoundError, ProcessLookupError):
        continue
if busy:
    print("GPU occupied:", gpu, uuid, busy)
    sys.exit(1)
print("GPU available:", gpu, uuid)
