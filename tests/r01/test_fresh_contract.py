"""CPU合同回归：不加载真实模型、不更新参数、不占用GPU。"""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))
import prd_fresh as fresh

torch.set_num_threads(1)


class FreshContractTests(unittest.TestCase):
    def test_common_settings_preserve_method_and_machine_fields(self):
        inherited = dict(response_fusion="mean", clip_path="/assets/clip", seed=3407,
                         operators={"old": True}, train_batch_size=16,
                         reconstruction_mode="cached_paired")
        with patch("prd_r01.config.read_config", return_value=inherited.copy()) as read:
            config = fresh.baseline_settings(ROOT, "vqgan_sr_mean")
        read.assert_called_once_with(ROOT / "machine.local.json", "vqgan_sr_mean")
        self.assertEqual(config["response_fusion"], "mean")
        self.assertEqual(config["clip_path"], "/assets/clip")
        self.assertEqual(config["seed"], 3407)
        self.assertNotIn("operators", config)
        self.assertEqual((config["train_batch_size"], config["test_batch_size"]), (32, 64))
        self.assertEqual(config["reconstruction_mode"], "online_after_augment")
        self.assertEqual(config["lora"]["rank"], 16)
        self.assertEqual(config["head"]["slots"], 2)

    def test_video_auc_averages_probabilities_before_ranking(self):
        # 平均logits会把负视频A排在两个正视频前，错误地得到0.5。
        differences = [10, -1, -1, -2, -2, 1, 1, 1, 2, 2]
        logits = np.column_stack([np.zeros(10), differences]).astype(np.float32)
        groups = [("negative_a", 0, 3), ("negative_b", 0, 2),
                  ("positive_a", 1, 3), ("positive_b", 1, 2)]
        rows = [dict(video_id=name, label=label)
                for name, label, count in groups for _ in range(count)]
        result = fresh.metric(logits, rows)
        self.assertEqual(result["video_auc"], 1.0)
        self.assertEqual((result["frames"], result["videos"]), (10, 4))
        self.assertLessEqual(result["independent_auc_error"], 1e-12)
        rows[1]["label"] = 1
        with self.assertRaisesRegex(ValueError, "同视频标签冲突"):
            fresh.metric(logits, rows)

    def test_tied_validation_keeps_earliest_best_and_counts_patience(self):
        progress = dict(best_auc=-1.0, best_epoch=None, bad_epochs=0)
        for epoch, score in enumerate([.7, .75, .75, .74, .75, .74, .75], 1):
            fresh.update_best(progress, score, epoch)
        self.assertEqual(progress, dict(best_auc=.75, best_epoch=2, bad_epochs=5))
        fresh.update_best(progress, .76, 8)
        self.assertEqual(progress, dict(best_auc=.76, best_epoch=8, bad_epochs=0))


if __name__ == "__main__":
    unittest.main()
