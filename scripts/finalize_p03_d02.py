import json
import shutil
import subprocess
from pathlib import Path
import numpy as np
from train_p03_t01 import ROOT, sha, stamp, save

s=ROOT/'.state/p03_d02_20261003'
assert (ROOT/'.state/p03_d02_queue_20261003/diagnostic_exit_code.txt').read_text().strip()=='0'
d=json.loads((s/'result.json').read_text())
rows=json.loads((s/'evaluation_manifest.json').read_text())
reported=json.loads((s/'per_video_ce.json').read_text())
y=np.array([r['label'] for r in rows])
assert d['status']=='complete' and len(rows)==192 and d['training_updates']==0
errors={}
for arm in ('paired','rolled'):
    f=np.load(s/(arm+'_features.npz'))
    errors[arm]={}
    for kind in ('aligned','same_label','opposite_label'):
        a=f[kind+'_logits'].astype(np.float64)
        if a.ndim==2:a=a[:,None,:]
        ce=np.logaddexp(a[:,:,0],a[:,:,1])-np.take_along_axis(a,y[:,None,None],axis=2)[:,:,0]
        ce=ce.mean(1)
        error=float(np.abs(ce-np.array(reported[arm][kind])).max())
        assert error<1e-5,(arm,kind,error)
        errors[arm][kind]=error
    for domain,m in d['metrics'][arm].items():
        ix=[i for i,r in enumerate(rows) if r['domain']==domain]
        for kind in ('aligned','same_label','opposite_label'):
            assert abs(float(np.array(reported[arm][kind])[ix].mean())-m[kind])<1e-7
for name,h in d['artifact_hashes'].items():assert sha(s/name)==h
save(s/'independent_verification.json',dict(time=stamp(),numpy_ce_max_error=errors,artifact_hashes_verified=True))
a=ROOT/'scripts/experiment_summaries/p03_d02_20261003'
a.mkdir(exist_ok=False)
for name in ('started.json','result.json','evaluation_manifest.json','donors.json','per_video_ce.json','independent_verification.json'):
    shutil.copy2(s/name,a/name)
shutil.copy2(ROOT/'scripts/p03_d02_protocol.json',a/'protocol.json')
summary=f"# P03-D02结果\n\n两份固定末权重、192视频，0次训练更新。交互J={d['macro_interaction']:.8f} nat；判定{d['verdict']}。逐域结果及限制见result.json；CPU numpy独立复算通过。特征与全部donor logits保留于203 .state/p03_d02_20261003/，hash已归档。仅为条件checkpoint响应，不能解释为方法因果或独立最终测试。\n"
(a/'实验摘要.md').write_text(summary,encoding='utf-8')
subprocess.run(['git','add','--',str(a.relative_to(ROOT))],cwd=ROOT,check=True)
ident=subprocess.check_output(['git','log','-1','--format=%an%n%ae'],cwd=ROOT,text=True).splitlines()
subprocess.run(['git','-c','user.name='+ident[0],'-c','user.email='+ident[1],'commit','-m','归档P03双末权重标签条件敏感性诊断'],cwd=ROOT,check=True)
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
save(s/'completion.json',dict(status='completed',id=d['id'],result_commit=commit,ended_at=stamp(),diagnostic_exit_code=0,
                             independent_metric_verification=True,archive=str(a),result_sha256=sha(s/'result.json')))
(s/'result_commit.txt').write_text(commit+'\n')
print('ARCHIVED',commit,flush=True)
