"""Persist finite queue progress and terminal state atomically."""
import json, os, subprocess, sys
from datetime import datetime
from pathlib import Path
queue, task, event = sys.argv[1:]
now=datetime.now().astimezone().isoformat()
record=dict(task=task,event=event,time=now,queue_pid=os.getppid(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
if event=='start':
    record['status']='running'
    with (Path('.state')/task/'started.json').open('x') as f: json.dump(dict(started_at=now,**record),f,indent=2)
elif event=='archived': record['status']='archived'
else: record['status']='complete' if task=='all_complete' and event=='0' else 'failed'
p=Path(queue)/'status.json'
tmp=p.with_suffix('.tmp'); tmp.write_text(json.dumps(record,indent=2)); tmp.replace(p)
with (Path(queue)/'events.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
print(json.dumps(record),flush=True)
