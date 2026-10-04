"""N01-T02: unit-norm VAE signed/absolute residuals, paired finite LoRA training."""
import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
from pathlib import Path
import numpy as np
import torch
from train_p03_t01 import (ROOT, IndexedDataset, LoraDetector, evaluate, save,
                           sha, stamp, target_images, trainable_hash)


def unit_residual(residual):
    norms=residual.norm(dim=1,keepdim=True)
    assert torch.isfinite(norms).all() and norms.min()>1e-8
    result=residual/norms
    assert (result.norm(dim=1)-1).abs().max()<1e-5
    return result


class DirectionDetector(LoraDetector):
    residual_form='signed'

    def forward(self,data_dict,inference=False):
        # eval disables encoder dropout only; LoRA gradients remain enabled.
        self.backbone.eval()
        self.vae_augmenter.eval()
        x,y=data_dict['image'],data_dict['label']
        cpu_rng,gpu_rng=torch.get_rng_state(),torch.cuda.get_rng_state()
        with torch.no_grad():
            probe=self.vae_augmenter(x,y)
        original=self.features(x)
        residual=original-self.features(probe.detach())
        direction=unit_residual(residual)
        if self.residual_form=='abs':direction=direction.abs()
        torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(gpu_rng)
        logits=self.head(torch.cat([original,direction],dim=1))
        self.last_norm_range=[float(residual.norm(dim=1).min().detach()),float(residual.norm(dim=1).max().detach())]
        return {'cls':logits,'prob':logits.softmax(1)[:,1]}


def main():
    p=argparse.ArgumentParser();p.add_argument('--claim',required=True);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    state=ROOT/'.state/n01_t02_20261004'
    out=state/'smoke' if args.smoke else Path('/root/autodl-tmp/prd-common/outputs/logs/mechanism/n01_t02_20261004')
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    random.seed(3407);np.random.seed(3407);torch.manual_seed(3407);torch.cuda.manual_seed_all(3407)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    config=json.loads((ROOT/'.state/p04_d02_20261003_a2/started.json').read_text())['config']
    config.update(cudnn=False,nEpochs=1,save_ckpt=False,save_feat=False,task_target='n01_t02_20261004')
    config['early_stopping']={'enabled':False}
    assert config['train_batchSize']==16 and config['test_batchSize']==64
    data=IndexedDataset(config=config,mode='train')
    loader=torch.utils.data.DataLoader(data,batch_size=16,shuffle=True,num_workers=8,
        collate_fn=data.collate_fn,generator=torch.Generator().manual_seed(3407))
    assert len(loader)==1798 and len(data)==28768
    save(out/'train_manifest.json',data.data_dict)
    images,rows=target_images(config,out)
    signed=DirectionDetector(config)
    absolute=copy.deepcopy(signed);absolute.residual_form='abs'
    initial={a:trainable_hash(m) for a,m in [('signed',signed),('abs',absolute)]}
    assert initial['signed']==initial['abs']
    models={'signed':signed.cuda(),'abs':absolute.cuda()}
    opt=config['optimizer']['adam']
    optimizers={a:torch.optim.Adam(m.parameters(),lr=opt['lr'],betas=(opt['beta1'],opt['beta2']),
        eps=opt['eps'],weight_decay=opt['weight_decay'],amsgrad=opt['amsgrad']) for a,m in models.items()}
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    inputs={str(p):sha(p) for p in [Path(config['dataset_json_folder'])/'FaceForensics++.json',
        Path(config['clip_path'])/'model.safetensors',Path(config['vae_path'])/'diffusion_pytorch_model.bin']}
    meta=dict(id='N01-T02/v1',run_id='n01_t02_20261004',smoke=args.smoke,started_at=stamp(),pid=os.getpid(),
        cwd=str(ROOT),claim_commit=args.claim,code_commit=code,script_sha256=sha(__file__),config=config,
        input_hashes=inputs,initial_trainable_sha256=initial,train_manifest_sha256=sha(out/'train_manifest.json'),
        evaluation_manifest_sha256=sha(out/'evaluation_manifest.json'),torch_version=torch.__version__,
        gpu=torch.cuda.get_device_name(),source_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
        trainable_parameters={a:sum(p.numel() for p in m.parameters() if p.requires_grad) for a,m in models.items()},
        protocol='Same VAE probe; unit residual signed versus coordinatewise absolute; encoder eval with LoRA gradients; same fresh init, batch, head RNG; two epochs; fixed terminal only.')
    assert meta['evaluation_manifest_sha256']=='ac5121d948e23b5c00f36b30e606224380e2ef18a566a04c4dae8c60438e988a'
    save(out/'run_metadata.json',meta);print(stamp(),'TRAINING_STARTED',json.dumps(meta['trainable_parameters']),flush=True)
    step=0
    with (out/'batch_trace.jsonl').open('x') as trace:
        for epoch in range(1 if args.smoke else 2):
            for batch,indexes in loader:
                trace.write(json.dumps(dict(step=step,epoch=epoch,indexes=indexes,
                    image_sha256=hashlib.sha256(batch['image'].numpy().tobytes()).hexdigest()))+'\n')
                batch={k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}
                cpu_rng,gpu_rng=torch.get_rng_state(),torch.cuda.get_rng_state();values={};gradient_checks={}
                for arm,model in models.items():
                    torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(gpu_rng)
                    model.train();optimizers[arm].zero_grad(set_to_none=True)
                    pred=model(batch);loss=model.get_losses(batch,pred)['overall'];assert torch.isfinite(loss)
                    loss.backward()
                    if step==0 or step%100==0:
                        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                        g=sum(float(p.grad.detach().abs().sum()) for n,p in model.named_parameters() if 'lora_' in n and p.grad is not None)
                        assert g>0,'LoRA has no gradient'
                        gradient_checks[arm]=g
                    optimizers[arm].step();optimizers[arm].zero_grad(set_to_none=True)
                    values[arm]=float(loss);del pred,loss
                step+=1
                if step==1 or step%100==0:
                    trace.flush()
                    progress=dict(status='running',time=stamp(),pid=os.getpid(),epoch=epoch,steps_per_arm=step,
                        loss=values,lora_gradient_l1=gradient_checks,norm_range={a:m.last_norm_range for a,m in models.items()},
                        peak_gpu_allocated=torch.cuda.max_memory_allocated())
                    save(out/'progress.json',progress,replace=True)
                    if not args.smoke:save(state/'status.json',progress,replace=True)
                    print(stamp(),'TRAIN',json.dumps(progress),flush=True)
                if args.smoke and step==2:break
            print(stamp(),'EPOCH_END',epoch,'steps_per_arm',step,flush=True)
    assert step==(2 if args.smoke else 3596)
    outputs={}
    for arm,model in models.items():
        assert trainable_hash(model)!=initial[arm]
        outputs[arm]=evaluate(model,images,rows)
        if not args.smoke:
            path=out/(arm+'_final.pth');torch.save(model.state_dict(),path)
            weights=torch.load(path,map_location='cpu',weights_only=True);model.load_state_dict(weights,strict=True);del weights
            outputs[arm].update(checkpoint=str(path),checkpoint_sha256=sha(path),strict_reload=True)
        print(stamp(),'EVALUATED',arm,json.dumps(outputs[arm]['metrics']),flush=True)
    dce=outputs['abs']['average_ce']-outputs['signed']['average_ce'];dauc=outputs['signed']['average_auc']-outputs['abs']['average_auc']
    domain_ce=[outputs['abs']['metrics'][d]['ce']-outputs['signed']['metrics'][d]['ce'] for d in outputs['signed']['metrics']]
    verdict=('supports_sign_at_fixed_budget' if dce>=0.02 and dauc>=0.01 and min(domain_ce)>=0 else
        'not_supported_at_fixed_budget' if dce<=0 or dauc<=0 else 'inconclusive')
    result=dict(status='smoke_passed' if args.smoke else 'training_complete_pending_archive',id='N01-T02/v1',run_id='n01_t02_20261004',
        started_at=meta['started_at'],ended_at=stamp(),code_commit=code,claim_commit=args.claim,steps_per_arm=step,
        outputs=outputs,delta_ce_abs_minus_signed=dce,delta_auc_signed_minus_abs=dauc,verdict=verdict,
        initial_weights_identical=True,batch_and_augmentation_identical_by_shared_tensor=True,head_dropout_rng_matched_per_step=True,
        encoder_dropout_disabled_but_lora_trained=True,early_stop_epoch=None,
        checkpoint_selection='smoke_two_steps_not_formal' if args.smoke else 'terminal_after_two_epochs',peak_gpu_allocated=torch.cuda.max_memory_allocated())
    save(out/'result.json',result);print(stamp(),'SMOKE_PASS' if args.smoke else 'TRAINING_COMPLETE',step,verdict,flush=True)


if __name__=='__main__':main()
