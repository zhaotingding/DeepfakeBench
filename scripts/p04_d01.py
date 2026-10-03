"""P04-D01/v1: metadata-only domain/label audit; no model or image inference."""
import hashlib
import json
import math
import resource
import subprocess
from collections import Counter
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import yaml

ROOT = Path('/root/DeepfakeBench-prd-common')
DATA = Path('/datasets/Deepfake/DeepfakeBench/config/dataset_json')
DOMAINS = ('Celeb-DF-v2', 'DFDCP', 'DFDC')
OUT = ROOT/'.state/p04_d01_20261003'

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def domain_stats(counts):
    """Exact tied-pair AUC for empirical domain priors; descriptive, not held-out."""
    totals = [sum(c[y] for c in counts.values()) for y in (0, 1)]
    scores = {d: Fraction(c[1], sum(c)) for d, c in counts.items()}
    wins = sum(Fraction(a[1]*b[0]) * (int(scores[d] > scores[e]) +
               Fraction(int(scores[d] == scores[e]), 2))
               for d, a in counts.items() for e, b in counts.items())
    n = sum(totals)
    mi = sum(c[y]/n*math.log2(c[y]*n/(sum(c)*totals[y]))
             for c in counts.values() for y in (0, 1) if c[y])
    return dict(auc=float(wins/(totals[0]*totals[1])), mutual_information_bits=mi,
                priors={d: float(s) for d, s in scores.items()})

def main():
    resource.setrlimit(resource.RLIMIT_AS, (2*1024**3, 2*1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (290, 290))
    assert domain_stats({'a': [5,5], 'b': [5,5]})['auc'] == 0.5
    assert domain_stats({'a': [10,0], 'b': [0,10]})['auc'] == 1.0
    assert domain_stats({'a': [3,1], 'b': [1,3]})['auc'] == 0.75
    OUT.mkdir(exist_ok=False)
    started = datetime.now().astimezone().isoformat()
    config_path = ROOT/'training/config/test_config.yaml'
    labels = yaml.safe_load(config_path.read_text())['label_dict']
    source_hashes, audits, rows = {}, {}, []
    counts = {d: [0,0] for d in DOMAINS}
    for domain in DOMAINS:
        p = DATA/(domain+'.json')
        source_hashes[domain] = sha(p)
        ds = json.loads(p.read_text())[domain]
        split_ids = {s: set() for s in ('train','val','test')}
        frame_counts = [0,0]
        effective_frames = [0,0]
        seen_frames = Counter()
        empty = []
        for label_str, splits in ds.items():
            assert labels[label_str] in (0,1)
            for split, videos in splits.items():
                assert split in split_ids
                for name, info in videos.items():
                    assert info['label'] == label_str
                    video_id = label_str+'_'+name
                    split_ids[split].add(video_id)
                    if split != 'test':
                        continue
                    y = labels[label_str]
                    frames = [f.replace('\\','/') for f in info['frames']]
                    if not frames:
                        empty.append(video_id)
                        continue
                    assert all(f.startswith(domain+'/') for f in frames)
                    counts[domain][y] += 1
                    frame_counts[y] += len(frames)
                    effective_frames[y] += min(32,len(frames))
                    seen_frames.update(frames)
                    frames.sort(key=lambda f: int(Path(f).stem))
                    rows.append(dict(domain=domain, label=y, label_string=label_str,
                        video_id=video_id, original_video_key=name,
                        frame=frames[len(frames)//2], available_frames=len(frames),
                        json_sha256=source_hashes[domain]))
        audits[domain] = dict(videos_real_fake=counts[domain], raw_frames_real_fake=frame_counts,
            loader_max32_frames_real_fake=effective_frames, empty_test_videos=empty,
            duplicate_test_frame_paths=sum(n-1 for n in seen_frames.values()),
            split_video_counts={s: len(v) for s,v in split_ids.items()},
            exact_video_overlap={a+'_'+b: len(split_ids[a]&split_ids[b])
                for a,b in [('train','test'),('val','test'),('train','val')]})
        assert sha(p) == source_hashes[domain], 'Input changed during read'
    assert all(n > 0 for c in counts.values() for n in c)
    k = min(32, min(n for c in counts.values() for n in c))
    selected = []
    for domain in DOMAINS:
        for y in (0,1):
            group = [r for r in rows if r['domain']==domain and r['label']==y]
            group.sort(key=lambda r: hashlib.sha256(
                f'P04-D01/v1|{domain}|{y}|{r["video_id"]}'.encode()).hexdigest())
            selected.extend(group[:k])
    assert len({(r['domain'],r['video_id']) for r in selected}) == 6*k
    assert len({r['frame'] for r in selected}) == 6*k
    manifest = OUT/'manifest.jsonl'
    with manifest.open('x') as f:
        for r in selected:
            f.write(json.dumps(r,sort_keys=True)+'\n')
    balanced = {d: [k,k] for d in DOMAINS}
    control = domain_stats(balanced)
    assert control['auc']==0.5 and control['mutual_information_bits']==0.0
    result = dict(diagnostic_id='P04-D01/v1',status='complete',started_at=started,
        finished_at=datetime.now().astimezone().isoformat(),
        claim_commit='5324e24b6db9de1b7250c675c9088ca676eaab75',
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        code_sha256=sha(Path(__file__)),config_sha256=sha(config_path),
        dataset_json_sha256=source_hashes,audits=audits,original=domain_stats(counts),
        balanced=control,selected_per_domain_label=k,selected_videos=6*k,
        manifest_sha256=sha(manifest),training_runs=0,gpu_used=False,
        limitations=['Descriptive domain-only AUC uses empirical test-label priors; not predictive validation.',
            'Cannot explain existing within-dataset AUC or show the PRD encoder uses domain cues.',
            'Video record is counting unit, not proven independent source identity; no independence CI claimed.',
            'Only metadata paths are hashed; image contents/LMDB availability and source-family overlap not verified.',
            'Target sets already participated in checkpoint selection; development diagnostic only.'])
    with (OUT/'result.json').open('x') as f:
        json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)

if __name__ == '__main__':
    main()
