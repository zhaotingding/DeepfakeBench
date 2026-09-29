import sys, json, hashlib, shutil
from pathlib import Path
from datetime import datetime
root=Path(__file__).resolve().parents[1]
task=sys.argv[1]
state=root/".state"/task
recovery = len(sys.argv) == 4 and sys.argv[3] == "--recover-launcher-exit"
completion_name = "completion.recovered.json" if recovery else "completion.json"
if recovery:
    previous = json.loads((state/"completion.json").read_text())
    assert previous["training_exit_code"] == 127 and int(sys.argv[2]) == 0
    console = (root/"logs"/task/"console.log").read_text()
    assert "TRAINING_EXIT_CODE=0 at " in console
result={"task":task,"training_exit_code":int(sys.argv[2]),"finished_at":datetime.now().isoformat(),"status":"failed"}
try:
    if recovery:
        result.update(recovered_from=str(state/"completion.json"), launcher_exit_code=127,
                      recovery_reason="Training exited 0; in-place edit of live Bash launcher corrupted its subsequent shell read.")
    assert result["training_exit_code"]==0, "Training failed; inspect console.log"
    runs=list((root/"logs/training/csy").glob("lora_"+task+"_*"))
    assert len(runs)==1, "Expected exactly one run directory"
    run=runs[0]
    assert "Stop Training on best Testing metric" in (run/"training.log").read_text()
    summary=list((root/"scripts/experiment_summaries").glob(run.name+"_*.md"))
    assert len(summary)==1, "Training summary missing"
    dst=root/"logs/checkpoint_backups"/task
    dst.mkdir()
    def sha(p):
        h=hashlib.sha256()
        with p.open("rb") as f:
            for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
        return h.hexdigest()
    src=run/"test/avg/ckpt_best.pth"
    digest=sha(src)
    for p in [src,run/"test/avg/metric_dict_best.pickle",run/"run_metadata.json",summary[0]]:
        with p.open("rb") as i,(dst/p.name).open("xb") as o: shutil.copyfileobj(i,o,8*1024*1024)
    assert digest==sha(dst/src.name)==sha(src), "Checkpoint checksum mismatch"
    manifest={"source":str(src),"sha256":digest,"bytes":src.stat().st_size,"summary":str(summary[0]),"run_metadata":str(run/"run_metadata.json")}
    with (dst/"manifest.json").open("x") as f: json.dump(manifest,f,indent=2)
    result.update(status="complete",run_dir=str(run),backup=str(dst),**manifest)
except Exception as error:
    result["error"]=str(error)
finally:
    with (state/completion_name).open("x") as f: json.dump(result,f,indent=2)
    with (root/"logs/RealTime"/(task+"_"+completion_name)).open("x") as f: json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)
sys.exit(0 if result["status"]=="complete" else 1)
