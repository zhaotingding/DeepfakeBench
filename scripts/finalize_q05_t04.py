import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

r=Path(__file__).resolve().parents[1]
s=r/'.state/q05_t04_20261006'
out=Path('/root/autodl-tmp/prd-common/outputs/logs/mechanism/q05_t04_20261006')
assert (s/'training_exit_code.txt').read_text().strip()=='0'
d=json.loads((out/'result.json').read_text())
meta=json.loads((out/'run_metadata.json').read_text())
rows=json.loads((out/'evaluation_manifest.json').read_text())
assert d['steps_per_arm']==3596 and meta['initial_trainable_sha256']['frozen_increment']==meta['initial_trainable_sha256']['adapted_increment']
assert d['newly_trained_arms']==['frozen_increment'] and d['control_batch_trace_identical']
assert (out/'batch_trace.jsonl').read_text()==Path('/root/autodl-tmp/prd-common/outputs/logs/mechanism/q05_t02_20261005/batch_trace.jsonl').read_text()
assert d['encoder_updates']==0 and d['encoder_frozen'] and d['head_only_trained']
assert d['frozen_lora_initial_sha256']==d['frozen_lora_final_sha256']
assert d['checkpoint_selection']=='terminal_after_two_epochs'
assert len((out/'batch_trace.jsonl').read_text().splitlines())==3596
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=r,text=True).strip()
archive=r/'scripts/experiment_summaries/q05_t04_20261006'
archive.mkdir(exist_ok=False)
backup=Path('/root/autodl-tmp/prd-common/outputs/logs/checkpoint_backups/q05_t04_20261006')
backup.mkdir(exist_ok=False)
labels=np.array([x['label'] for x in rows])
for arm,v in d['outputs'].items():
 p=Path(v['checkpoint']);assert sha(p)==v['checkpoint_sha256'] and v['strict_reload']
 logits=np.array(v['logits']);ce=np.logaddexp(logits[:,0],logits[:,1])-logits[np.arange(192),labels]
 assert np.max(np.abs(ce-np.array(v['ce'])))<1e-5
 for domain,m in v['metrics'].items():
  ix=[i for i,x in enumerate(rows) if x['domain']==domain]
  assert abs(roc_auc_score(labels[ix],np.array(v['probabilities'])[ix])-m['auc'])<1e-12
  assert abs(float(ce[ix].mean())-m['ce'])<1e-5
 shutil.copy2(p,backup/p.name);assert sha(backup/p.name)==v['checkpoint_sha256']
for name in ['result.json','run_metadata.json','evaluation_manifest.json','batch_trace.jsonl']:
 shutil.copy2(out/name,archive/name)
shutil.copy2(r/'scripts/q05_t04_protocol.json',archive/'protocol.json')
completion=dict(d,status='completed',normal_training_exit=0,archived_at=datetime.now().astimezone().isoformat(),
 independent_metric_verification=True,backup=str(backup),
 artifacts_sha256={p.name:sha(p) for p in archive.iterdir()})
with (archive/'completion.json').open('x') as f:json.dump(completion,f,indent=2,ensure_ascii=False)
summary=f"# Q05-T04固定预算训练结果\n\n冻结编码器臂只新增分类头3596次更新，适配对照复用T03的3596次更新，正常退出并按同一固定终点评估；不使用目标AUC挑权重。\n\nΔCE={d['delta_ce_frozen_minus_adapted']:.8f}，ΔAUC={d['delta_auc_adapted_minus_frozen']:.8f}，判定：{d['verdict']}。\n\n逐集指标、起止时间、同一checkpoint路径/hash及独立复核见completion.json。两臂等单位特征残差范数，两臂同增量算子及分类头，新增臂编码器完全冻结，适配臂训练成本另列；仅2轮、1个配对seed，目标开发小清单，不能推断总体等效或最终泛化。\n"
(archive/'实验摘要.md').write_text(summary,encoding='utf-8')
subprocess.run(['git','add','--',str(archive.relative_to(r))],cwd=r,check=True)
ident=subprocess.check_output(['git','log','-1','--format=%an%n%ae'],cwd=r,text=True).splitlines()
subprocess.run(['git','-c','user.name='+ident[0],'-c','user.email='+ident[1],'commit','-m','归档Q05冻结编码器同容量头及适配对照'],cwd=r,check=True)
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip()
with (s/'result_commit.txt').open('x') as f:f.write(commit+'\n')
shutil.copy2(archive/'completion.json',s/'completion.json')
print('ARCHIVED',commit,flush=True)
