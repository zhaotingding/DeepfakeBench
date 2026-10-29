"""传统算子的独立公式、缓存及两槽模型验证。"""
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
from torch import nn
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))
from prd_r01.operators import ClassicalProbe, classical_spec
from prd_r01.model import ResponseModel

class OperatorsTests(unittest.TestCase):
    def setUp(self):
        self.c = json.loads((ROOT / "training/config/r01/common.json").read_text())
        self.c.update(mean=[0,0,0], std=[1,1,1], resolution=8)
        self.c["operators"]["lowrank"]["rank"] = 2

    def test_haar_matches_explicit_orthogonal_transform(self):
        x = np.random.default_rng(12).random((2,3,8,8)).astype(np.float32)
        h = np.array([[1,1],[1,-1]], dtype=np.float64) / np.sqrt(2)
        expected = np.empty_like(x)
        for n in range(2):
            for c in range(3):
                for i in range(0,8,2):
                    for j in range(0,8,2):
                        coef = h @ x[n,c,i:i+2,j:j+2] @ h.T
                        coef[0,1] *= .5; coef[1,0] *= .5; coef[1,1] *= .5
                        expected[n,c,i:i+2,j:j+2] = h.T @ coef @ h
        actual = ClassicalProbe(self.c,"wavelet")(torch.from_numpy(x)).numpy()
        np.testing.assert_allclose(actual,expected,atol=2e-7,rtol=0)

    def test_svd_matches_numpy_rank2_and_preserves_constant(self):
        x = np.random.default_rng(41).random((2,3,8,8)).astype(np.float32)
        u,s,vh = np.linalg.svd(x.astype(np.float64),full_matrices=False)
        expected = (u[...,:2] * s[...,:2][...,None,:]) @ vh[...,:2,:]
        probe = ClassicalProbe(self.c,"lowrank")
        np.testing.assert_allclose(probe(torch.from_numpy(x)).numpy(),expected,atol=3e-6,rtol=0)
        torch.testing.assert_close(probe(torch.ones(2,3,8,8)),torch.ones(2,3,8,8),atol=2e-6,rtol=0)

    def test_two_slots_skip_unused_operator_and_backpropagate(self):
        encoder=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(3,4))
        model=ResponseModel(encoder,4,{},["wavelet"],hidden=8,cached=True,slots=2)
        x=torch.randn(3,3,8,8);out=model(x,{"wavelet":x*.5})
        self.assertEqual(out["features"].shape,(3,8))
        out["logits"].sum().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in encoder.parameters()))
        with self.assertRaises(ValueError):
            ResponseModel(encoder,4,{},["wavelet","lowrank"],cached=True,slots=2)

    def test_recipe_changes_when_rank_changes(self):
        first=classical_spec(self.c,"lowrank")
        changed=dict(self.c,operators={"lowrank":{"rank":3}})
        self.assertNotEqual(first,classical_spec(changed,"lowrank"))

if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
