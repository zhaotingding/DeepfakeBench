"""Q10: paired original/increment joint training, unchanged optimizer."""
import argparse,copy,hashlib,json,os,random,subprocess,time
from pathlib import Path
import numpy as np
import torch
from train_p03_t01 import ROOT,IndexedDataset,save,sha,stamp,trainable_hash,target_images,evaluate
from train_q05_t03 import DirectionDetector,unit_residual

RUN='q10_t01_20261006'

class JointDetector(DirectionDetector):
    def forward(self,data_dict,inference=False):
        self.backbone.eval();self.vae_augmenter.eval()
        x,y=data_dict['image'],data_dict['label']
        cpu,gpu=torch.get_rng_state(),torch.cuda.get_rng_state()
        z=self.features(x)
        if self.arm=='joint':
            with torch.no_grad():
                pre=self.preprocess(x);rec=self.vae_augmenter(x,y)
            u=unit_residual(self.features(pre)-self.features(rec))
        else:u=torch.zeros_like(z)
        torch.set_rng_state(cpu);torch.cuda.set_rng_state(gpu)
        logits=self.head(torch.cat([z,u],1))
        return {'cls':logits,'prob':logits.softmax(1)[:,1]}

def seed():
    random.seed(3407);np.random.seed(3407);torch.manual_seed(3407);torch.cuda.manual_seed_all(3407)

def configuration():
    c=json.loads((ROOT/'.state/p04_d02_20261003_a2/started.json').read_text())['config']
    c.update(cudnn=False,nEpochs=12,save_ckpt=False,save_feat=False,task_target=RUN)
    c['early_stopping']={'enabled':False}
    assert c['train_batchSize']==16 and c['test_batchSize']==64
    return c

def optimizer(model,c):
    o=c['optimizer']['adam']
    return torch.optim.Adam(model.parameters(),lr=o['lr'],betas=(o['beta1'],o['beta2']),eps=o['eps'],weight_decay=o['weight_decay'],amsgrad=o['amsgrad'])

def benchmark(claim):
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    out=ROOT/'.state'/RUN/'smoke';out.mkdir(parents=True,exist_ok=False)
    c=configuration();measurements={};inits={};traces={}
    for arm in ['original','joint']:
        seed();data=IndexedDataset(config=c,mode='train')
        loader=torch.utils.data.DataLoader(data,batch_size=16,shuffle=True,num_workers=8,collate_fn=data.collate_fn,generator=torch.Generator().manual_seed(3407))
        assert len(loader)==1798 and len(data)==28768
        model=JointDetector(c);model.arm=arm;inits[arm]=trainable_hash(model);model.cuda();opt=optimizer(model,c)
        times=[];trace=[]
        for step,(batch,ix) in enumerate(loader):
            trace.append(dict(indexes=ix,image_sha256=hashlib.sha256(batch['image'].numpy().tobytes()).hexdigest()))
            b={k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}
            torch.cuda.synchronize();start=time.monotonic();model.train();opt.zero_grad(set_to_none=True)
            p=model(b);loss=model.get_losses(b,p)['overall'];assert torch.isfinite(loss);loss.backward()
            assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if 'lora_' in n)
            opt.step();torch.cuda.synchronize();times.append(time.monotonic()-start)
            print(stamp(),'SMOKE_UPDATE',arm,step+1,times[-1],float(loss),flush=True)
            if step==7:break
        model.eval();evaltimes=[]
        with torch.inference_mode():
            for _ in range(3):
                b={'image':b['image'].repeat(4,1,1,1),'label':b['label'].repeat(4)} if b['image'].shape[0]==16 else b
                torch.cuda.synchronize();start=time.monotonic();p=model(b);assert torch.isfinite(p['cls']).all();torch.cuda.synchronize();evaltimes.append(time.monotonic()-start)
        measurements[arm]=dict(update_seconds=float(np.median(times[2:])),bs64_seconds=float(np.median(evaltimes)),peak_allocated=torch.cuda.max_memory_allocated(),steps=8)
        traces[arm]=trace;del model,opt,p,loss,b;torch.cuda.empty_cache()
    assert inits['original']==inits['joint'] and traces['original']==traces['joint']
    # 12 source validations, four192 anchors/arm; four full source/target evaluations.
    updates=21576*sum(v['update_seconds'] for v in measurements.values())
    evals=sum(v['bs64_seconds']/64 for v in measurements.values())*(12*700+4*192+2*(5600+45232))
    estimate=1.10*(updates+evals)+1800
    result=dict(status='smoke_passed',run_id=RUN,claim_commit=claim,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),time=stamp(),measurements=measurements,initial_hashes=inits,batch_trace_identical=True,projected_seconds_with_margin=estimate,budget_seconds=86400,budget_pass=estimate<86400,optimizer=c['optimizer'],formal_updates=0)
    save(out/'result.json',result);print('SMOKE_RESULT',json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--claim',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    assert a.smoke,'Formal runner is not deployed yet'
    benchmark(a.claim)
