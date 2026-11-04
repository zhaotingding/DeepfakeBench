"""只运行已认领的 R01 末轮增量评估，不加载训练入口。"""
import argparse
import concurrent.futures
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path, payload):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def run_batch(plan_file, claim):
    worktree = Path(__file__).resolve().parents[1]
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    root = Path.home() / "prd-runs" / "末轮八域补评估-20261104"
    state = root / "queue-four"
    state.mkdir(parents=True, exist_ok=True)
    lock = (state / "supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (state / "status.json").exists():
        raise RuntimeError("已有队列状态，拒绝重复启动")
    code = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=worktree, text=True).strip()
    if plan["code_commit"] != code or len(set(plan["runs"])) != len(plan["runs"]):
        raise ValueError("代码身份或任务清单不正确")
    record = dict(started=now(), pid=os.getpid(), code=code, claim=claim,
                  domains=plan["domains"], rows={name: {"stage": "queued"} for name in plan["runs"]})
    update_lock = threading.Lock()
    atomic_json(state / "status.json", record)

    def update(run, **changes):
        with update_lock:
            record["rows"][run].update(changes)
            record["checked"] = now()
            atomic_json(state / "status.json", record)

    overall_deadline = time.monotonic() + 72 * 3600
    def evaluate(gpu, run):
        free = int(subprocess.check_output(["nvidia-smi", "-i", str(gpu),
            "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
        if free < plan["minimum_free_mib"]:
            update(run, stage="blocked_gpu", free_mib=free, time=now())
            return False
        cmd = [sys.executable, "-u", str(worktree / "scripts/八域末轮评估.py"), "run",
               "--run", run, "--domains", *plan["domains"]]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID",
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   TOKENIZERS_PARALLELISM="false", PYTHONUTF8="1", HF_HUB_OFFLINE="1")
        with (state / (run + ".log")).open("x", encoding="utf-8") as log:
            proc = subprocess.Popen(cmd, env=env, cwd=worktree, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            update(run, stage="running", gpu=gpu, pid=proc.pid, started=now(), launch_free_mib=free)
            try:
                limit = min(12 * 3600, overall_deadline - time.monotonic())
                rc = proc.wait(timeout=max(0, limit))
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=40)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                update(run, stage="time_limit", ended=now(), exit=proc.returncode)
                return True
        files = [root / "results" / run / d / "verified.json" for d in plan["domains"]]
        ok = rc == 0 and all(p.is_file() and json.loads(p.read_text(encoding="utf-8")) for p in files)
        update(run, stage="verified_new_four" if ok else "failed", exit=rc, ended=now())
        return True

    def slot(gpu):
        resource_dir = Path("/tmp") / ("prd-" + str(os.getuid()))
        resource_dir.mkdir(mode=0o700, exist_ok=True)
        with (resource_dir / f"eight-terminal-gpu{gpu}.lock").open("a") as gpu_lock:
            fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            while time.monotonic() < overall_deadline:
                with update_lock:
                    run = next((n for n, value in record["rows"].items() if value["stage"] == "queued"), None)
                    if run is None:
                        return
                    record["rows"][run].update(stage="assigned", gpu=gpu)
                    atomic_json(state / "status.json", record)
                try:
                    if not evaluate(gpu, run):
                        return
                except Exception as error:
                    update(run, stage="supervision_error", error=repr(error), ended=now())
                    return

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(plan["gpus"])) as pool:
        futures = [pool.submit(slot, gpu) for gpu in plan["gpus"]]
        for future in futures:
            future.result()
    record.update(ended=now(), stage="finished" if all(v["stage"] == "verified_new_four"
                  for v in record["rows"].values()) else "finished_with_gaps")
    atomic_json(state / "status.json", record)
    return 0 if record["stage"] == "finished" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("claim")
    args = parser.parse_args()
    sys.exit(run_batch(args.plan, args.claim))

