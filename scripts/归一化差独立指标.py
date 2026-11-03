def independent_metrics(logits,rows):
    import numpy as np
    from scipy.stats import rankdata
    groups={}
    for prediction,row in zip(logits,rows):
        key=(row['domain'],row['video_id']);item=groups.setdefault(key,{'label':row['label'],'values':[]})
        if item['label']!=row['label']:raise ValueError('视频标签冲突')
        item['values'].append(prediction)
    domains={}
    for domain in sorted({k[0] for k in groups}):
        items=[v for k,v in groups.items() if k[0]==domain]
        z=np.array([np.mean(v['values'],axis=0) for v in items],dtype=np.float64)
        y=np.array([v['label'] for v in items]);pos=y==1;neg=y==0
        normalizer=np.logaddexp(z[:,0],z[:,1]);scores=np.exp(z[:,1]-normalizer)
        ranks=rankdata(scores);n=pos.sum();m=neg.sum()
        auc=(ranks[pos].sum()-n*(n+1)/2)/(n*m)
        loss=normalizer-z[np.arange(len(y)),y]
        domains[domain]={'auc':float(auc),'ce':float(loss.mean()),'balanced_ce':float((loss[pos].mean()+loss[neg].mean())/2),'videos':len(y)}
    return {'domains':domains,'macro':{k:float(np.mean([v[k] for v in domains.values()])) for k in ['auc','ce','balanced_ce']}}
