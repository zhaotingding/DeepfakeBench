"""Bounded physical BS16/BS64 acceptance; never starts a full epoch."""
import copy, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_prd_environment import load_config, preflight
import torch
from torch.utils.data import DataLoader, Subset
from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from detectors import DETECTOR

config = load_config(sys.argv[1])
preflight(config)
torch.manual_seed(config['manualSeed'])
torch.cuda.manual_seed_all(config['manualSeed'])
torch.backends.cudnn.benchmark = config['cudnn']
assert config['train_batchSize'] == 16 and config['test_batchSize'] == 64
model = DETECTOR[config['model_name']](config).cuda()
adam = config['optimizer']['adam']
optimizer = torch.optim.Adam(model.parameters(), lr=adam['lr'], betas=(adam['beta1'], adam['beta2']), eps=adam['eps'], weight_decay=adam['weight_decay'], amsgrad=adam['amsgrad'])

def batches(mode, name=None):
    cfg = copy.deepcopy(config)
    if name: cfg['test_dataset'] = name
    dataset = DeepfakeAbstractBaseDataset(config=cfg, mode=mode)
    size = cfg['train_batchSize'] if mode == 'train' else cfg['test_batchSize']
    return DataLoader(Subset(dataset, list(range(size * (2 if mode == 'train' else 1)))), batch_size=size, num_workers=cfg['workers'], collate_fn=dataset.collate_fn)

def move(batch):
    batch['image'] = batch['image'].cuda()
    batch['label'] = (batch['label'].cuda() != 0).long()
    return batch

model.train()
for i, batch in enumerate(batches('train')):
    started = time.monotonic()
    batch = move(batch)
    optimizer.zero_grad()
    pred = model(batch)
    assert list(pred['cls'].shape) == [16, 2]
    loss = model.get_losses(batch, pred)['overall']
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    torch.cuda.synchronize()
    print(json.dumps(dict(phase='train', batch=i, size=16, loss=loss.item(), seconds=time.monotonic()-started, peak_bytes=torch.cuda.max_memory_allocated())), flush=True)
    del batch, pred, loss
del optimizer
torch.cuda.empty_cache()
model.eval()
for name in config['test_dataset']:
    with torch.no_grad():
        for batch in batches('test', name):
            pred = model(move(batch), inference=True)
            assert list(pred['cls'].shape) == [64, 2] and torch.isfinite(pred['prob']).all()
            print(json.dumps(dict(phase='test', dataset=name, size=64, peak_bytes=torch.cuda.max_memory_allocated())), flush=True)
            del batch, pred
print('FORMAL_BATCH_SMOKE_PASS', flush=True)
