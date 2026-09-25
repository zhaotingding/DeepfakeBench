"""Re-evaluate the archived T1 checkpoint at two test batch sizes."""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
saved_argv = sys.argv
sys.argv = ["test.py"]
import test as evaluation
sys.argv = saved_argv

config = yaml.safe_load((ROOT / "training/config/detector/prd_t1_seed2026.yaml").read_text())
config.update(yaml.safe_load((ROOT / "training/config/train_config.yaml").read_text()))
config["save_tsne"] = False
config["analysis"] = {"save_features": False, "compute_statistics": False}
evaluation.resolve_data_paths(config)
torch.manual_seed(config["manualSeed"])
torch.cuda.set_device(0)
checkpoint = ROOT / "logs/checkpoint_backups/prd_t1_seed2026_20260924/ckpt_best.pth"
model = evaluation.DETECTOR[config["model_name"]](config).to(evaluation.device)
model.load_state_dict(torch.load(checkpoint, map_location=evaluation.device), strict=True)
reference = {"Celeb-DF-v2": 0.90962631, "DFDCP": 0.89386922, "DFDC": 0.83582218}
results = {}
for batch_size in (64, 16):
    config["test_batchSize"] = batch_size
    loaders = evaluation.prepare_testing_data(config)
    metrics, _ = evaluation.test_epoch(model, loaders)
    result = {name: {key: float(value) for key, value in values.items()
                     if key not in ("pred", "label")} for name, values in metrics.items()}
    results[str(batch_size)] = result
    output = ROOT / f"scripts/experiment_summaries/prd_t1_seed2026_eval_bs{batch_size}.json"
    with output.open("x") as file:
        json.dump({"checkpoint": str(checkpoint), "batch_size": batch_size,
                   "data_root": config["data_root"], "datasets": result,
                   "completed_at": datetime.now().isoformat()}, file, indent=2)
    print(f"Completed batch {batch_size}: " + json.dumps(
        {name: values["auc"] for name, values in result.items()}), flush=True)
    if batch_size == 64:
        for name, expected in reference.items():
            actual = result[name]["auc"]
            if abs(actual - expected) > 0.0001:
                raise RuntimeError(f"Batch 64 validation differs from saved T1 metric: {name} {actual} vs {expected}")
comparison = {name: results["16"][name]["auc"] - results["64"][name]["auc"] for name in reference}
output = ROOT / "scripts/experiment_summaries/prd_t1_seed2026_eval_bs_comparison.json"
with output.open("x") as file:
    json.dump({"checkpoint": str(checkpoint), "seed": 2026,
               "batch16_minus_batch64_auc": comparison,
               "reference_batch64_auc": reference}, file, indent=2)
print("Batch 16 minus 64 AUC: " + json.dumps(comparison), flush=True)
