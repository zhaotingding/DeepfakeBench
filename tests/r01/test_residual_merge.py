"""合并后的残差、辅助梯度和激活重算CPU合同；不加载预训练模型、不更新参数。"""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))
from prd_r01.model import ResponseModel, CLIPEncoder
from prd_r01.auxiliary_supervision import training_loss

torch.set_num_threads(1)


def small_model(residual="signed_diff", auxiliary="none"):
    torch.manual_seed(3407)
    return ResponseModel(nn.Linear(3, 3, bias=False).double(), 3, {}, ["vqgan"],
                         hidden=4, dropout=0, cached=True, slots=2,
                         residual=residual, auxiliary_input=auxiliary).double()


class ResidualMergeTests(unittest.TestCase):
    def test_residual_formulas_gradients_and_zero_vectors(self):
        x = torch.tensor([[1., -2., 4.], [-3., 2., 1.]], dtype=torch.float64)
        xr = torch.tensor([[2., 1., -2.], [1., -1., 3.]], dtype=torch.float64)
        for mode in ["signed_diff", "absolute_diff", "unit_residual", "normalized_feature_diff"]:
            with self.subTest(mode=mode):
                model = small_model(mode).eval()
                z, zr = model.encoder(x), model.encoder(xr)
                delta = z - zr
                expected = {"signed_diff": delta, "absolute_diff": delta.abs(),
                            "unit_residual": delta / (delta.norm(dim=1, keepdim=True) + 1e-6),
                            "normalized_feature_diff": z / (z.norm(dim=1, keepdim=True) + 1e-6)
                            - zr / (zr.norm(dim=1, keepdim=True) + 1e-6)}[mode]
                features = model(x, {"vqgan": xr})["features"]
                torch.testing.assert_close(features[:, :3], z, rtol=0, atol=0)
                torch.testing.assert_close(features[:, 3:], expected, rtol=0, atol=0)
                actual_grad = torch.autograd.grad(features[:, 3:].sum(), model.encoder.weight)[0]
                expected_grad = torch.autograd.grad(expected.sum(), model.encoder.weight)[0]
                torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
                zero = model(torch.zeros_like(x), {"vqgan": torch.zeros_like(xr)})["features"]
                self.assertTrue(torch.isfinite(zero).all())
                self.assertEqual(torch.count_nonzero(zero).item(), 0)

    def test_auxiliary_preserves_main_initialization_rng_and_eval_path(self):
        original = small_model()
        original_rng = torch.get_rng_state().clone()
        auxiliary = small_model(auxiliary="original_aux")
        self.assertTrue(torch.equal(original_rng, torch.get_rng_state()))
        for name, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, auxiliary.state_dict()[name]), name)
        x = torch.tensor([[1., -2., 4.], [-3., 2., 1.]], dtype=torch.float64)
        xr, labels = x * .5, torch.tensor([0, 1])
        auxiliary.train()
        out = auxiliary(x, {"vqgan": xr})
        total, main, aux = training_loss(out, labels, "original_aux", 1.0)
        torch.testing.assert_close(total, main + aux, rtol=0, atol=0)
        named = list(auxiliary.named_parameters())
        gradients = torch.autograd.grad(aux, [p for _, p in named], allow_unused=True)
        self.assertTrue(any(g is not None and torch.count_nonzero(g) for (n, _), g in zip(named, gradients) if n.startswith("encoder.")))
        self.assertTrue(all(g is None for (n, _), g in zip(named, gradients) if n.startswith("head.")))
        self.assertTrue(any(g is not None and torch.count_nonzero(g) for (n, _), g in zip(named, gradients) if n.startswith("auxiliary_head.")))
        auxiliary.eval()
        result = auxiliary(x, {"vqgan": xr})
        self.assertNotIn("auxiliary_logits", result)
        torch.testing.assert_close(result["logits"], auxiliary.head(result["features"]), rtol=0, atol=0)
        none_loss, none_main, no_aux = training_loss(result, labels)
        self.assertIs(none_loss, none_main)
        self.assertIsNone(no_aux)
        torch.testing.assert_close(none_loss, F.cross_entropy(result["logits"], labels), rtol=0, atol=0)

    def test_full_and_even_checkpoint_keep_outputs_gradients_and_rng(self):
        for mode, expected_calls in [("clip_transformer_nonreentrant_preserve_rng_v1", 24),
                                     ("clip_even_layers_nonreentrant_preserve_rng_v1", 12)]:
            with self.subTest(mode=mode):
                torch.manual_seed(14)
                vision = nn.Module()
                vision.config = SimpleNamespace(hidden_size=3)
                vision.encoder = nn.Module()
                vision.encoder.layers = nn.ModuleList([nn.Sequential(nn.Linear(3, 3), nn.Dropout(.1)).double() for _ in range(24)])
                lora = dict(rank=16, alpha=16, dropout=.05, target_modules=["q_proj", "v_proj"])
                with patch("transformers.CLIPModel.from_pretrained", return_value=SimpleNamespace(vision_model=vision)), \
                     patch("peft.get_peft_model", side_effect=lambda model, _: model), \
                     patch("torch.utils.checkpoint.checkpoint", wraps=checkpoint) as checkpoint_call:
                    CLIPEncoder("unused-local-path", lora, mode)
                    apply = vision.encoder._gradient_checkpointing_func
                    inputs = torch.tensor([[.2, .3, -.4]], dtype=torch.float64, requires_grad=True)
                    parameters = [inputs, *vision.parameters()]
                    torch.manual_seed(19)
                    plain = inputs
                    for layer in vision.encoder.layers:
                        plain = layer(plain)
                    plain_rng = torch.get_rng_state().clone()
                    plain_grad = torch.autograd.grad(plain.sum(), parameters)
                    torch.manual_seed(19)
                    actual = inputs
                    for layer in vision.encoder.layers:
                        actual = apply(layer.__call__, actual)
                    actual_grad = torch.autograd.grad(actual.sum(), parameters)
                    torch.testing.assert_close(actual, plain, rtol=0, atol=0)
                    for got, expected in zip(actual_grad, plain_grad):
                        torch.testing.assert_close(got, expected, rtol=0, atol=0)
                    self.assertTrue(torch.equal(torch.get_rng_state(), plain_rng))
                    self.assertEqual(checkpoint_call.call_count, expected_calls)
                    for call in checkpoint_call.call_args_list:
                        self.assertFalse(call.kwargs["use_reentrant"])
                        self.assertTrue(call.kwargs["preserve_rng_state"])


if __name__ == "__main__":
    unittest.main()
