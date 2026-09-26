# PRD 实验操作笔记

## 监控频率与顺序队列

- 用户于 2026-09-26 要求：每次检查依据最近完整 epoch 的训练与评测耗时、当前 epoch、连续下降计数和队列交接时间，预估下次检查间隔；禁止默认固定每小时唤醒对话。预计长时间无事件时延长间隔，临近结束或存在异常时缩短；普通状态保持安静。不要把应用 heartbeat 当成免费的服务器检查，它会触发模型调用。
- 初始尚无本次轮耗时时参考历史同配置运行；第一次正式检查再重估。建议长运行阶段约 4–8 小时、结束窗口约 1–2 个 epoch；早停依赖尚未观测的指标，不承诺精确结束时间。
- 已确定的下一实验由服务器队列在上一组正常退出、权重校验归档、摘要 Git 提交后自动启动，无需等待模型定时检查；异常、归档或提交失败必须停队列，不能跳过或重复启动。
- 适用：4090-48G，远程 `/home/zhaoting.ding/disk/zhaoting.ding/DeepfakeBench`，GPU0 单卡。唯一训练入口为 `scripts/launch_prd_training.sh`，具有 flock 和日志不覆盖保护；不操作其他用户进程、不 push。
- 证据：本次用户要求；远程启动器和 `.state/prd_t2_blur_seed2026_20260925/run.sh`；本地 `.codex/automations/prd/automation.toml`（旧任务每小时、已暂停）。验证日期 2026-09-26：旧队列正常归档和提交已实际验证；自适应监控策略是用户要求，尚需下一次定时触发验证。

## 实验解释约束

- 当前协议用 CDFv2 / DFDCP / DFDC 三集平均 AUC 选择 checkpoint 和早停，属于目标测试集参与选择的探索结果；不能当作独立泛化评测。
- 早停是完整 epoch 末平均 AUC 连续三次逐轮下降，不是三轮未刷新历史最好；epoch 从 0 开始。
- T3 identity 在训练态对同一图像两次 CLIP 前向，LoRA dropout=0.05，残差未必为零；F1/F2 的分类头宽度与 concat 不同，存在容量混杂。
- 证据：远程 `training/train.py`、`training/early_stopping.py`、`training/detectors/lora_detector.py`；2026-09-26 代码核验，T3 零残差解释不成立的判断基于代码。

## 2026-09-26 配对队列

- 有限队列：T3 identity → F1 orig → F2 residual，seed=2026、训练 BS32、测试 BS64；与已完成的 T1/T2 seed2026 配对。每组从原始预训练权重初始化，不继承上一组训练权重。
- 队列入口 `scripts/run_prd_seed2026_queue.sh`；状态 `.state/prd_seed2026_queue_20260926/runner.log`，每组 `.state/prd_<实验>_seed2026_20260926/completion.json`、`result_commit.txt`；队列完成文件仅在全部归档与提交后写出。队列结束不自行添加新实验。
- GPU0 真实 FF++ 小批量验证：三组各两次前向、反向、Adam 更新，loss 与梯度有限；特征宽度 T3=2048，F1/F2=1024。证据 `.state/prd_seed2026_queue_20260926/smoke.log`；2026-09-26 实际验证，仅证明小批量路径可运行，完整训练待完成。
- 初次监控暂估 6 小时后；后续按本次实际 epoch 耗时重估，不把 6 小时设成永久固定频率。修改应用调度文件后须区分“配置已写入”与“实际定时触发已验证”。
