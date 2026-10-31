import sys,json,importlib.util
from pathlib import Path
import numpy as np
import torch
from scipy.signal import convolve2d
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'training'))
from prd_r01.diverse_operators import sparse_code,ASSETS
from prd_r01.vendor.e4e.models.stylegan2.op.native_ops import upfirdn2d,fused_leaky_relu

def main():
    torch.set_num_threads(2);rng=np.random.default_rng(31);checks=[]
    for up,down,pad in [(1,1,(1,2)),(2,1,(1,1)),(1,2,(1,1)),(2,1,(-1,2))]:
        x=rng.normal(size=(2,3,7,9)).astype(np.float32);kernel=np.array([[1,2,1],[2,4,2],[1,2,1]],np.float32)/16
        got=upfirdn2d(torch.from_numpy(x),torch.from_numpy(kernel),up=up,down=down,pad=pad).numpy();ys=[]
        for image in x:
            channels=[]
            for channel in image:
                zero=np.zeros((channel.shape[0]*up,channel.shape[1]*up),np.float32);zero[::up,::up]=channel
                p0,p1=pad;zero=np.pad(zero,((max(p0,0),max(p1,0)),)*2)
                zero=zero[max(-p0,0):zero.shape[0]-max(-p1,0),max(-p0,0):zero.shape[1]-max(-p1,0)]
                channels.append(convolve2d(zero,kernel,mode='valid')[::down,::down])
            ys.append(channels)
        error=float(np.max(np.abs(got-np.array(ys))));assert error<1e-6;checks.append(dict(up=up,down=down,pad=pad,error=error))
    x=torch.arange(-12,12,dtype=torch.float32).reshape(2,3,2,2)/10;b=torch.tensor([.1,.2,-.3]);v=x.numpy()+b.numpy()[None,:,None,None]
    err=float(np.max(np.abs(fused_leaky_relu(x,b).numpy()-np.where(v>=0,v,v*.2)*np.sqrt(2))));assert err<1e-6
    d=torch.eye(64);y=torch.zeros(3,64);y[0,[1,8,13]]=torch.tensor([.5,-.2,.3]);y[1,4]=.7
    c=sparse_code(y,d);assert (c@d.T-y).abs().max()<1e-6 and int((c.abs()>1e-8).sum())==4
    learned=np.load(ASSETS/'source_ksvd_dictionary.npy',allow_pickle=False);meta=json.loads((ASSETS/'source_ksvd_fit.json').read_text())
    assert meta['target_data_used'] is False and len(meta['samples'])==2048 and len(meta['curve'])==10
    assert np.isfinite(learned).all() and np.allclose(np.linalg.norm(learned,axis=0),1,atol=1e-5)
    print(json.dumps(dict(status='pass',filter_checks=checks,activation_error=err,sparse_exact_signal_error=float((c@d.T-y).abs().max()),source_dictionary_checks=True)))
if __name__=='__main__':main()
