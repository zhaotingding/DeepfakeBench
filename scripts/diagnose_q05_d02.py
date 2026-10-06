"""Q05-D02: reciprocal reference substitution in fixed T05/T06 checkpoints."""
import argparse,copy,json,os,subprocess,shutil
from pathlib import Path
import numpy as np
import torch
from train_p03_t01 import ROOT,save,sha,stamp,target_images,evaluate
from train_q05_t06 import DirectionDetector,unit_residual


class SwapDetector(DirectionDetector):
    def forward(self,data_dict,inference=False):
        assert not self.training and not any(p.requires_grad for p in self.parameters())
        x,y=data_dict['image'],data_dict['label']
        with torch.no_grad():
            pre=self.preprocess(x)
            original=(self.anchor_encoder.forward_features(pre) if self.reference=='fixed'
                      else self.features(pre))
            reconstructed=self.features(self.vae_augmenter(x,y))
            u=unit_residual(original-reconstructed)
            logits=self.head(torch.cat([torch.zeros_like(original),u],dim=1))
        return {'cls':logits,'prob':logits.softmax(1)[:,1]}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--claim',required=True);args=parser.parse_args()
    run='q05_d02_20261006';state=ROOT/'.state'/run
    out=Path('/root/autodl-tmp/prd-common/outputs/logs/mechanism')/run
    out.mkdir(exist_ok=False);torch.set_num_threads(4)
    torch.manual_seed(3407);torch.cuda.manual_seed_all(3407)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    base=out.parent
    t05=json.loads((base/'q05_t05_20261006/result.json').read_text())
    t06=json.loads((base/'q05_t06_20261006/result.json').read_text())
    meta=json.loads((base/'q05_t06_20261006/run_metadata.json').read_text())
    config=copy.deepcopy(meta['config']);config['task_target']=run
    images,rows=target_images(config,out)
    assert sha(out/'evaluation_manifest.json')==meta['evaluation_manifest_sha256']
    assert len(rows)==192 and len({(r['domain'],r['video_id']) for r in rows})==192
    inputs={'t05':t05['outputs']['stop_pre'],'t06':t06['outputs']['fixed_pre']}
    for v in inputs.values():assert sha(Path(v['checkpoint']))==v['checkpoint_sha256']
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    metadata=dict(id='Q05-D02/v1',run_id=run,started_at=stamp(),pid=os.getpid(),cwd=str(ROOT),
                  code_commit=code,claim_commit=args.claim,config=config,checkpoint_inputs=inputs,
                  evaluation_manifest_sha256=sha(out/'evaluation_manifest.json'),
                  updates=0,torch_version=torch.__version__,bootstrap_seed=20261006,bootstrap_replicates=2000)
    save(out/'run_metadata.json',metadata)
    weights06=torch.load(inputs['t06']['checkpoint'],map_location='cpu',weights_only=True)
    anchor={k:v for k,v in weights06.items() if k.startswith('anchor_backbone.')}
    assert anchor and t06['anchor_initial_sha256']==t06['anchor_final_sha256']
    outputs={};reload_errors={}
    for identity in ['t05','t06']:
        model=SwapDetector(config)
        weights=(torch.load(inputs[identity]['checkpoint'],map_location='cpu',weights_only=True)
                 if identity=='t05' else weights06)
        if identity=='t05':weights.update(anchor)
        model.load_state_dict(weights,strict=True)
        if identity=='t05':del weights
        model.requires_grad_(False);model.eval();model.cuda()
        for reference in ['moving','fixed']:
            model.reference=reference
            outputs[identity+'_'+reference]=evaluate(model,images,rows)
            progress=dict(status='running',time=stamp(),pid=os.getpid(),updates=0,model=identity,reference=reference)
            save(state/'status.json',progress,replace=True)
            print(stamp(),'EVALUATED',identity,reference,flush=True)
        native=outputs[identity+('_moving' if identity=='t05' else '_fixed')]
        error=float(np.max(np.abs(np.array(native['logits'])-np.array(inputs[identity]['logits']))))
        assert error<1e-5,'Native checkpoint response not reproduced'
        reload_errors[identity]=error
        del model;torch.cuda.empty_cache()
    effects={};per_video=[]
    for identity in ['t05','t06']:
        moving,fixed=outputs[identity+'_moving'],outputs[identity+'_fixed']
        effects[identity]=dict(delta_ce=moving['average_ce']-fixed['average_ce'],
                              delta_auc=fixed['average_auc']-moving['average_auc'],
                              domain_ce={d:moving['metrics'][d]['ce']-fixed['metrics'][d]['ce'] for d in moving['metrics']})
        per_video.append(np.array(moving['ce'])-np.array(fixed['ce']))
    paired=np.mean(per_video,axis=0)
    groups=[np.array([i for i,r in enumerate(rows) if r['domain']==domain and r['label']==label])
            for domain in sorted({r['domain'] for r in rows}) for label in [0,1]]
    assert all(len(g)==32 for g in groups)
    rng=np.random.default_rng(20261006)
    boots=[float(paired[np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])].mean()) for _ in range(2000)]
    ci=np.quantile(boots,[.025,.975]).tolist()
    supported=all(e['delta_ce']>=.02 and e['delta_auc']>=.01 and min(e['domain_ce'].values())>=0 for e in effects.values()) and ci[0]>0
    rejected=any(e['delta_ce']<=0 or e['delta_auc']<=0 for e in effects.values())
    verdict='supports_reciprocal_reference_sensitivity' if supported else 'not_supported' if rejected else 'inconclusive'
    result=dict(id='Q05-D02/v1',run_id=run,status='completed',started_at=metadata['started_at'],ended_at=stamp(),
                code_commit=code,claim_commit=args.claim,updates=0,outputs=outputs,effects=effects,
                mean_paired_delta_ce=float(paired.mean()),paired_delta_ce_ci95=ci,native_reload_max_error=reload_errors,
                verdict=verdict,statistical_unit='video, one frame each, domain-label stratified paired bootstrap',
                limitations='Fixed-model response on reused development videos, not retrained-method causality or independent final test; shared identities not validated.')
    save(out/'result.json',result)
    archive=ROOT/'scripts/experiment_summaries'/run;archive.mkdir(exist_ok=False)
    backup=Path('/root/autodl-tmp/prd-common/outputs/logs/checkpoint_backups')/run;backup.mkdir(exist_ok=False)
    artifacts={}
    for name in ['result.json','run_metadata.json','evaluation_manifest.json']:
        shutil.copy2(out/name,archive/name);shutil.copy2(out/name,backup/name)
        assert sha(archive/name)==sha(backup/name)==sha(out/name)
        artifacts[name]=sha(out/name)
    shutil.copy2(ROOT/'scripts/q05_d02_protocol.json',archive/'protocol.json')
    artifacts['protocol.json']=sha(archive/'protocol.json')
    completion=dict(result,backup=str(backup),artifacts_sha256=artifacts,archived_at=stamp())
    save(archive/'completion.json',completion)
    (archive/'实验摘要.md').write_text(f'# Q05-D02双向参照替换\n\n0参数更新。判定：{verdict}；同末权重的推理响应，不是重训因果。完整指标及协议见completion.json。\n',encoding='utf-8')
    assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip()
    subprocess.run(['git','add','--',str(archive.relative_to(ROOT))],cwd=ROOT,check=True)
    ident=subprocess.check_output(['git','log','-1','--format=%an%n%ae'],cwd=ROOT,text=True).splitlines()
    subprocess.run(['git','-c','user.name='+ident[0],'-c','user.email='+ident[1],'commit','-m','归档Q05双向参照替换的固定模型响应'],cwd=ROOT,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    save(state/'completion.json',completion)
    with (state/'result_commit.txt').open('x') as f:f.write(commit+'\n')
    print(stamp(),'ARCHIVED',commit,verdict,flush=True)


if __name__=='__main__':main()
