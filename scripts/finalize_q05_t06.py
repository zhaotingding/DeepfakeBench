import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

r=Path(__file__).resolve().parents[1]
s=r/'.state/q05_t06_20261006'
out=Path('/root/autodl-tmp/prd-common/outputs/logs/mechanism/q05_t06_20261006')
assert (s/'training_exit_code.txt').read_text().strip()=='0'
d=json.loads((out/'result.json').read_text())
meta=json.loads((out/'run_metadata.json').read_text())
rows=json.loads((out/'evaluation_manifest.json').read_text())
assert d['steps_per_arm']==3596 and meta['initial_trainable_sha256']['fixed_pre']==meta['initial_trainable_sha256']['moving_pre']
assert d['newly_trained_arms']==['fixed_pre'] and d['control_batch_trace_identical']
assert d['anchor_initial_sha256']==d['anchor_final_sha256'] and d['anchor_gradients_none']
assert (out/'batch_trace.jsonl').read_text()==Path('/root/autodl-tmp/prd-common/outputs/logs/mechanism/q05_t05_20261006/batch_trace.jsonl').read_text()
assert d['checkpoint_selection']=='terminal_after_two_epochs'
assert len((out/'batch_trace.jsonl').read_text().splitlines())==3596
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=r,text=True).strip()
archive=r/'scripts/experiment_summaries/q05_t06_20261006'
archive.mkdir(exist_ok=False)
backup=Path('/root/autodl-tmp/prd-common/outputs/logs/checkpoint_backups/q05_t06_20261006')
backup.mkdir(exist_ok=False)
labels=np.array([x['label'] for x in rows])
backup_paths={}
new_checkpoint=Path(d['outputs']['fixed_pre']['checkpoint'])
assert new_checkpoint.stat().st_size*2<6*1024**3
for arm,v in d['outputs'].items():
 p=Path(v['checkpoint']);assert sha(p)==v['checkpoint_sha256'] and v['strict_reload']
 logits=np.array(v['logits']);ce=np.logaddexp(logits[:,0],logits[:,1])-logits[np.arange(192),labels]
 assert np.max(np.abs(ce-np.array(v['ce'])))<1e-5
 for domain,m in v['metrics'].items():
  ix=[i for i,x in enumerate(rows) if x['domain']==domain]
  assert abs(roc_auc_score(labels[ix],np.array(v['probabilities'])[ix])-m['auc'])<1e-12
  assert abs(float(ce[ix].mean())-m['ce'])<1e-5
 if arm=='fixed_pre':
  bp=backup/p.name;shutil.copy2(p,bp)
 else:
  bp=Path('/root/autodl-tmp/prd-common/outputs/logs/checkpoint_backups/q05_t05_20261006')/p.name
 assert sha(bp)==v['checkpoint_sha256']
 backup_paths[arm]=str(bp)
for name in ['result.json','run_metadata.json','evaluation_manifest.json','batch_trace.jsonl']:
 shutil.copy2(out/name,archive/name)
shutil.copy2(r/'scripts/q05_t06_protocol.json',archive/'protocol.json')
completion=dict(d,status='completed',normal_training_exit=0,archived_at=datetime.now().astimezone().isoformat(),
 independent_metric_verification=True,backup=str(backup),backup_paths=backup_paths,
 artifacts_sha256={p.name:sha(p) for p in archive.iterdir()})
with (archive/'completion.json').open('x') as f:json.dump(completion,f,indent=2,ensure_ascii=False)
summary=f"# Q05-T06固定预算训练结果\n\n固定初始pre参照臂新增3596次LoRA及头更新，移动stop-gradient参照控制复用T05的3596次更新，正常退出并按同一固定终点评估；不使用目标AUC挑权重。\n\nΔCE={d['delta_ce_moving_minus_fixed']:.8f}，ΔAUC={d['delta_auc_fixed_minus_moving']:.8f}，判定：{d['verdict']}。\n\n逐集指标、起止时间、同一checkpoint路径/hash及独立复核见completion.json。两臂等单位特征残差范数，编码器关闭dropout但LoRA实际更新，固定初始pre编码器不更新，VAE编码器及头适配；适配后前向不同，不作纯梯度因果；仅2轮、1个配对seed，目标开发小清单，不能推断总体等效或最终泛化。\n"
(archive/'实验摘要.md').write_text(summary,encoding='utf-8')
subprocess.run(['git','add','--',str(archive.relative_to(r))],cwd=r,check=True)
ident=subprocess.check_output(['git','log','-1','--format=%an%n%ae'],cwd=r,text=True).splitlines()
subprocess.run(['git','-c','user.name='+ident[0],'-c','user.email='+ident[1],'commit','-m','归档Q05固定初始参照与移动参照控制'],cwd=r,check=True)
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip()
with (s/'result_commit.txt').open('x') as f:f.write(commit+'\n')
shutil.copy2(archive/'completion.json',s/'completion.json')
print('ARCHIVED',commit,flush=True)
