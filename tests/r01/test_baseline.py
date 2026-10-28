"""CPU合成验证：不使用真实训练集，不创建永久训练产物。"""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))
from prd_r01.model import ResponseModel
from prd_r01.engine import video_metrics, run_training
from prd_r01.data import load_manifest, validate_source_pair, LMDBImages
from prd_r01.config import read_protocol

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 4)
        self.dropout = nn.Dropout(.2)
        self.calls = 0
    def forward(self, x):
        self.calls += 1
        return self.dropout(self.linear(x.mean((2, 3))))

class Probe(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(.5))
        self.calls = 0
    def forward(self, x):
        self.calls += 1
        return x * self.scale

class BaselineTests(unittest.TestCase):
    def make_model(self, responses):
        torch.manual_seed(12)
        return ResponseModel(Encoder(), 4, {"vae": Probe()} if responses else {}, responses, hidden=8, dropout=.2)

    def test_original_skips_probes_and_zero_slots(self):
        model = self.make_model([])
        out = model(torch.ones(2, 3, 8, 8))
        self.assertEqual(model.encoder.calls, 1)
        self.assertEqual(tuple(out["features"].shape), (2, 12))
        self.assertEqual(torch.count_nonzero(out["features"][:, 4:]), 0)

    def test_repeated_response_reuses_compute(self):
        model = self.make_model(["vae", "vae"])
        out = model(torch.randn(2, 3, 8, 8))
        self.assertEqual(model.encoder.calls, 2)
        self.assertEqual(model.probes["vae"].calls, 1)
        self.assertTrue(torch.equal(out["features"][:, 4:8], out["features"][:, 8:]))

    def test_two_distinct_responses_have_independent_slots(self):
        torch.manual_seed(12)
        model=ResponseModel(Encoder(),4,{"vae":Probe(),"other":Probe()},["vae","other"],hidden=8)
        model.probes["other"].scale.data.fill_(.25)
        out=model(torch.randn(3,3,8,8))
        self.assertEqual(model.encoder.calls,3)
        self.assertFalse(torch.equal(out["features"][:,4:8],out["features"][:,8:]))

    def test_encoder_gets_gradient_frozen_probe_does_not(self):
        model = self.make_model(["vae"])
        out = model(torch.randn(3, 3, 8, 8))
        torch.nn.functional.cross_entropy(out["logits"], torch.tensor([0, 1, 0])).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in model.probes.parameters()))

    def test_extra_views_preserve_global_rng(self):
        images = torch.ones(3, 3, 8, 8)
        states = []
        for responses in [[], ["vae"], ["vae", "vae"]]:
            model = self.make_model(responses)
            torch.manual_seed(88)
            model(images)
            states.append(torch.get_rng_state())
        self.assertTrue(all(torch.equal(states[0], x) for x in states[1:]))

    def test_video_mean_logits_and_equal_domain_macro(self):
        rows = [dict(domain="a",video_id="r",label=0),dict(domain="a",video_id="f",label=1),
                dict(domain="b",video_id="r",label=0),dict(domain="b",video_id="f",label=1)]
        m = video_metrics(np.array([[2,0],[0,2],[0,2],[2,0]]), rows)
        self.assertEqual(m["domains"]["a"]["auc"], 1.)
        self.assertEqual(m["domains"]["b"]["auc"], 0.)
        self.assertEqual(m["macro"]["auc"], .5)

    def test_overlap_rejected(self):
        train=dict(role="source_train",rows=[dict(domain="FaceForensics++",source_ids=["1"])])
        val=dict(role="source_val",rows=[dict(domain="FaceForensics++",source_ids=["1"])])
        with self.assertRaisesRegex(ValueError,"overlap"):
            validate_source_pair(train,val)

    def test_draft_protocol_cannot_train(self):
        with self.assertRaisesRegex(ValueError,"separately approved"):
            read_protocol(ROOT / "training/config/r01/protocol.template.json")

    def test_synthetic_lmdb_and_training_lifecycle(self):
        import cv2, lmdb
        with tempfile.TemporaryDirectory(prefix="prd-r01-unit-") as temp:
            root=Path(temp)
            db=root/"DeepfakeBench/lmdb/FaceForensics++_lmdb"
            db.mkdir(parents=True)
            env=lmdb.open(str(db),map_size=4*1024**2)
            rows=[]
            with env.begin(write=True) as txn:
                for i in range(8):
                    key=f"video{i}\\frame.png"
                    raw=cv2.imencode(".png",np.full((8,8,3),20+i*25,np.uint8))[1].tobytes()
                    txn.put(key.encode(),raw)
                    rows.append(dict(domain="FaceForensics++",key=key,label=i%2,video_id=str(i),source_ids=[str(i)]))
            env.close()
            c=json.loads((ROOT/"training/config/r01/common.json").read_text())
            c.update(data_root=temp,output_root=str(root/"outputs"),workers=0,train_batch_size=2,test_batch_size=2,log_every=1)
            files=[]
            for role, group in [("source_train",rows[:4]),("source_val",rows[4:])]:
                p=root/(role+".json");p.write_text(json.dumps(dict(role=role,rows=group)));files.append(str(p))
            protocol=dict(id="synthetic-unit-only",epochs=2,seed=3,max_wall_seconds=60,
                          train_manifest=files[0],val_manifest=files[1],evaluation_manifests=[files[1]])
            # 验收源val不能冒充末端source_test，先验证拒绝。
            with self.assertRaisesRegex(ValueError,"Endpoint manifests"):
                run_training(c,protocol,"rejected",torch.device("cpu"))
            target=root/"target.json";target.write_text(json.dumps(dict(role="target_dev",rows=rows[4:])))
            protocol["evaluation_manifests"]=[str(target)]
            def fake_git(command,**kwargs):
                return "" if "status" in command else "synthetic-test-commit\n"
            with patch("subprocess.check_output",side_effect=fake_git), patch("prd_r01.engine.build_model",side_effect=lambda config,device:self.make_model([])):
                result=run_training(c,protocol,"unit",torch.device("cpu"))
            self.assertEqual(result["optimizer_updates"],4)
            out=root/"outputs/r01/unit"
            self.assertTrue((out/"source_val_epoch2.npy").is_file())
            self.assertTrue((out/"source_best.pth").is_file())
            self.assertTrue((out/"terminal.pth").is_file())
            self.assertTrue((out/"source_best_eval0.npy").is_file())
            self.assertEqual(result["endpoints"]["terminal"][0]["role"],"target_dev")
            with self.assertRaises(FileExistsError):
                with patch("subprocess.check_output",side_effect=fake_git):
                    run_training(c,protocol,"unit",torch.device("cpu"))
            bad=LMDBImages([dict(rows[0],key="missing")],c)
            with self.assertRaises(KeyError):
                bad[0]
            bad.close()

if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
