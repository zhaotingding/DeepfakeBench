import sys, json, hashlib, shutil
from pathlib import Path
from datetime import datetime
root=Path(__file__).resolve().parents[1]
task=sys.argv[1]
state=root/".state"/task
result={"task":task,"training_exit_code":int(sys.argv[2]),"finished_at":datetime.now().isoformat(),"status":"failed"}
try:
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
    with (state/"completion.json").open("x") as f: json.dump(result,f,indent=2)
    with (root/"logs/RealTime"/(task+"_completion.json")).open("x") as f: json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)
sys.exit(0 if result["status"]=="complete" else 1)
