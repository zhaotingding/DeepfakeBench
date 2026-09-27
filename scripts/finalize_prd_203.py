"""Archive one completed 203 run; refuse partial/ambiguous results."""
import hashlib, json, math, pickle, re, shutil, sys
from datetime import datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
from path_config import resolve_output_path

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def finalize(task, code):
    assert code == 0, f'Training failed with exit code {code}'
    runs = list(resolve_output_path('logs/training').glob('lora_'+task+'_*'))
    assert len(runs) == 1, 'Expected exactly one run'
    run = runs[0]
    text = (run/'training.log').read_text()
    assert 'Stop Training on best Testing metric' in text, 'Missing normal training end'
    meta = json.loads((run/'run_metadata.json').read_text())
    cfg = meta['config']
    assert cfg['manualSeed'] == 3407 and cfg['train_batchSize'] == 16 and cfg['test_batchSize'] == 64
    assert not meta['git_dirty'], 'Training code was dirty'
    src = run/'test/avg/ckpt_best.pth'
    with (run/'test/avg/metric_dict_best.pickle').open('rb') as f: metrics = pickle.load(f)
    per_set = {k: float(v) for k,v in metrics['dataset_dict'].items()}
    assert set(per_set) == {'Celeb-DF-v2','DFDCP','DFDC'}
    assert all(math.isfinite(v) and 0 <= v <= 1 for v in per_set.values())
    avg = float(metrics['auc'])
    assert abs(sum(per_set.values())/3 - avg) < 1e-10
    summaries = list(run.glob('*_实验摘要.md'))
    assert len(summaries) == 1 and '## 同一平均最佳 checkpoint 的结果' in summaries[0].read_text()
    dst = resolve_output_path('logs/checkpoint_backups')/task
    dst.mkdir(parents=True, exist_ok=False)
    digest = sha(src)
    for path in [src, run/'test/avg/metric_dict_best.pickle', run/'run_metadata.json', summaries[0]]:
        with path.open('rb') as i, (dst/path.name).open('xb') as o: shutil.copyfileobj(i,o,8*1024*1024)
    assert sha(dst/src.name) == digest == sha(src)
    epochs = re.findall(r'\[EarlyStop\] epoch=(\d+).*stop=(True|False)',text)
    selection = re.search(r'选择时刻：Epoch (\d+)，global step (\d+)', summaries[0].read_text())
    result = dict(status='complete', task=task, run_id=run.name, run_dir=str(run), seed=3407,
                  thread_id='01a0e12a-41bd-7b81-999d-fd62332d03cb', training_commit=meta['git_commit'],
                  started_at=json.loads((ROOT/'.state'/task/'started.json').read_text())['started_at'],
                  finished_at=datetime.now().astimezone().isoformat(), dataset_auc=per_set, average_auc=avg,
                  last_epoch=int(epochs[-1][0]) if epochs else None,
                  early_stop_epoch=int(epochs[-1][0]) if epochs and epochs[-1][1]=='True' else None,
                  selected_epoch=int(selection[1]) if selection else None,
                  checkpoint=str(src), checkpoint_sha256=digest, checkpoint_bytes=src.stat().st_size,
                  backup=str(dst), train_batch=16, test_batch=64,
                  protocol='Target-set-selected development evaluation; seed, batch, hardware and environment confounded across hosts.')
    with (dst/'manifest.json').open('x') as f: json.dump(result,f,indent=2)
    target = ROOT/'scripts/experiment_summaries'/task
    target.mkdir(parents=True,exist_ok=False)
    with (target/'result.json').open('x') as f: json.dump(result,f,indent=2)
    with (target/'summary.md').open('x') as f:
        f.write(summaries[0].read_text()+'\n\n目标三集参与checkpoint选择与早停；属于开发评估。203物理BS16，4090物理BS32，跨机差异不能全部归因于seed。\n')
    with (ROOT/'.state'/task/'completion.json').open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)

if __name__ == '__main__': finalize(sys.argv[1],int(sys.argv[2]))
