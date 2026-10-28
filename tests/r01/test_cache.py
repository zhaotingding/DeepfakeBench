"""重构缓存行为验证：映射、共享增强、梯度和训练闭环。"""
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch
import cv2
import lmdb
import numpy as np
import torch
from torch import nn
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))
from prd_r01.cache import build_cache, read_cache, bind_caches
from prd_r01.data import LMDBImages, make_loader
from prd_r01.model import ResponseModel
from prd_r01.engine import run_training
from test_baseline import Encoder, Probe

class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="prd-cache-test-")
        self.root = Path(self.temp.name)
        db = self.root / "DeepfakeBench/lmdb/FaceForensics++_lmdb"
        db.mkdir(parents=True)
        self.rows = []
        env = lmdb.open(str(db), map_size=4*1024**2)
        with env.begin(write=True) as txn:
            for i in range(8):
                key = "video" + str(i) + "\\frame.png"
                pixels = (np.arange(8*8*3).reshape(8,8,3) + i*7).astype(np.uint8)
                txn.put(key.encode(), cv2.imencode(".png", pixels)[1].tobytes())
                self.rows.append(dict(domain="FaceForensics++",key=key,label=i%2,video_id=str(i),source_ids=[str(i)]))
        env.close()
        self.config = json.loads((ROOT / "training/config/r01/common.json").read_text())
        self.config.update(data_root=str(self.root),output_root=str(self.root/"outputs"),resolution=8,
                           workers=0,train_batch_size=2,test_batch_size=2,responses=["vae"],log_every=100)
        self.spec = dict(operator="vae",recipe="test-only",resolution=8,mean=self.config["mean"],
                         std=self.config["std"],storage="normalized-float32-nchw")
        self.path = self.root / "缓存"
        build_cache(self.rows,self.config,nn.Identity(),str(self.path),self.spec,torch.device("cpu"),2,60)
        self.config["reconstruction_caches"] = {"vae": str(self.path)}

    def tearDown(self):
        self.temp.cleanup()

    def test_exact_read_and_row_order(self):
        dataset = LMDBImages(list(reversed(self.rows)),self.config)
        for i in range(len(dataset)):
            image, _, _, views = dataset[i]
            self.assertTrue(torch.equal(image,views["vae"]))  # 缓存身份由原始键定位，不依赖行号巧合。
        dataset.close()

    def test_shared_augmentation_and_worker_loading(self):
        random.seed(14); np.random.seed(14)
        dataset = LMDBImages(self.rows,self.config,augment=True)
        for i in range(32):
            image, _, _, views = dataset[i%8]
            torch.testing.assert_close(image,views["vae"],atol=3e-6,rtol=0)
        dataset.close()
        loader = make_loader(self.rows,dict(self.config,workers=2),True,31,torch.device("cpu"))
        for image, labels, indices, views in loader:
            torch.testing.assert_close(image,views["vae"],atol=3e-6,rtol=0)
        loader.dataset.close()

    def test_original_augmentation_rng_is_not_shifted_by_extra_view(self):
        original = LMDBImages(self.rows,dict(self.config,responses=[]),augment=True)
        paired = LMDBImages(self.rows,self.config,augment=True)
        def collect(dataset):
            random.seed(199); np.random.seed(199)
            return [dataset[i%8][0] for i in range(16)]
        first, second = collect(original), collect(paired)
        for a,b in zip(first,second):
            self.assertTrue(torch.equal(a,b))
        original.close(); paired.close()

    def test_online_and_cached_encoder_gradients_match_without_augmentation(self):
        torch.manual_seed(12)
        online = ResponseModel(Encoder(),4,{"vae":Probe()},["vae"],hidden=8,dropout=0).eval()
        cached = ResponseModel(Encoder(),4,{},["vae"],hidden=8,dropout=0,cached=True).eval()
        cached.encoder.load_state_dict(online.encoder.state_dict())
        cached.head.load_state_dict(online.head.state_dict())
        images = torch.randn(3,3,8,8)
        labels = torch.tensor([0,1,0])
        a = online(images)["logits"]
        b = cached(images,{"vae":images*.5})["logits"]
        self.assertTrue(torch.equal(a,b))
        torch.nn.functional.cross_entropy(a,labels).backward()
        torch.nn.functional.cross_entropy(b,labels).backward()
        for pa,pb in zip(online.encoder.parameters(),cached.encoder.parameters()):
            self.assertTrue(torch.equal(pa.grad,pb.grad))

    def test_partial_mismatch_and_corruption_rejected(self):
        meta_path = self.path / "缓存清单.json"
        original = meta_path.read_text()
        meta = json.loads(original)
        meta["status"] = "building"; meta_path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError,"未完整"):
            read_cache(self.path,self.rows,self.config)
        meta_path.write_text(original)
        with self.assertRaisesRegex(ValueError,"预处理"):
            read_cache(self.path,self.rows,dict(self.config,resolution=16))
        with self.assertRaisesRegex(ValueError,"元数据"):
            read_cache(self.path,[dict(self.rows[0],label=1)],self.config)
        arr = np.load(self.path/"重构张量.npy",mmap_mode="r+")
        arr[0,0,0,0] += 1
        arr.flush(); del arr
        with self.assertRaisesRegex(ValueError,"校验失败"):
            read_cache(self.path,self.rows,self.config,verify_file=True)

    def test_source_changes_rejected(self):
        db = self.root / "DeepfakeBench/lmdb/FaceForensics++_lmdb"
        env = lmdb.open(str(db))
        with env.begin(write=True) as txn:
            txn.put(self.rows[0]["key"].encode(),cv2.imencode(".png",np.zeros((8,8,3),np.uint8))[1].tobytes())
        env.close()
        dataset = LMDBImages(self.rows,self.config)
        with self.assertRaisesRegex(ValueError,"原图内容变化"):
            dataset[0]
        dataset.close()

    def test_cached_model_gradients_and_repeated_slot(self):
        torch.manual_seed(3)
        model = ResponseModel(Encoder(),4,{},["vae","vae"],hidden=8,dropout=0,cached=True)
        images = torch.randn(2,3,8,8)
        with self.assertRaisesRegex(ValueError,"缓存响应"):
            model(images)
        out = model(images,{"vae":images*.5})
        self.assertEqual(model.encoder.calls,2)
        self.assertEqual(len(model.probes),0)
        self.assertTrue(torch.equal(out["features"][:,4:8],out["features"][:,8:]))
        out["logits"].sum().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.encoder.parameters()))

    def test_never_overwrite_cache(self):
        with self.assertRaises(FileExistsError):
            build_cache(self.rows,self.config,nn.Identity(),str(self.path),self.spec,torch.device("cpu"))

    def test_cached_training_endpoints_and_protocol(self):
        manifests = []
        for role,rows in [("source_train",self.rows[:4]),("source_val",self.rows[4:]),("target_dev",self.rows[4:])]:
            file = self.root/(role+".json")
            file.write_text(json.dumps(dict(role=role,rows=rows)))
            manifests.append(str(file))
        protocol = dict(id="synthetic-cache-only",epochs=2,seed=3,max_wall_seconds=60,
                        train_manifest=manifests[0],val_manifest=manifests[1],evaluation_manifests=[manifests[2]],
                        reconstruction_caches={"vae":str(self.path)})
        def fake_git(command,**kwargs):
            return "" if "status" in command else "synthetic-cache-test"
        def model(config,device):
            return ResponseModel(Encoder(),4,{},["vae"],hidden=8,cached=True)
        with patch("prd_r01.cache.operator_spec",return_value=self.spec), patch("subprocess.check_output",side_effect=fake_git), patch("prd_r01.engine.build_model",side_effect=model):
            result = run_training(self.config,protocol,"中文缓存测试",torch.device("cpu"))
        self.assertEqual(result["optimizer_updates"],4)
        meta = json.loads((self.root/"outputs/r01/中文缓存测试/metadata.json").read_text())
        self.assertEqual(meta["cache_identities"]["vae"]["spec"],self.spec)
        self.assertEqual(result["endpoints"]["terminal"][0]["role"],"target_dev")

if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
