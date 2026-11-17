# PRD 研究代码

本仓库用于重构算子、多响应融合和视觉表征研究。当前训练、恢复、选点和评测共用早停实现；不同研究方向保留各自的方法代码。

## 先确认打开的目录

| 工作树目录 | 分支 | 用途 |
|---|---|---|
| 公共基线 | 公共基线 | 统一的训练、评测实现与原图／VAE对照 |
| 重构算子探索 | 4090-重构算子探索 | 重构算子、残差定义和辅助监督 |
| 多响应融合 | 4090-多响应融合 | 多种响应的融合 |
| 视觉表征探索 | 4090-视觉表征探索 | 编码器与表示方法 |

4090上的这些目录位于~/prd-worktrees/。旧DeepfakeBench目录也是工作树，检出prd-4090；它保存共用Git数据库和历史代码，不是当前新实验入口。打开一个目录不会自动切换其他工作树的分支。

执行前在当前目录核对：
```bash
git branch --show-current
git rev-parse HEAD
git status --short
```

## 当前文件入口

| 文件 | 职责 |
|---|---|
| [scripts/早停基线.py](scripts/早停基线.py) | 仅原图、经典PRD等方案的共同入口；具体实验使用本目录已提供的具名脚本 |
| [training/prd_fresh.py](training/prd_fresh.py) | 公共配置、数据加载、训练、完整状态恢复、最佳点选择与评测 |
| r01.machine.json | 当前机器的数据、预训练权重、输出路径；实际运行仍读取此文件，保持Git忽略 |
| [r01.machine.example.json](r01.machine.example.json) | 不含真实机器路径的配置模板 |
| [requirements-prd.txt](requirements-prd.txt) | 已记录的PRD运行环境依赖；不等于新机器已完成安装验收 |
| tests/r01/ | 公共规则与方法的CPU测试 |
| scripts/experiment_summaries/ | 可发布的必要结果摘要，不存训练权重或原始逐步日志 |

旧scripts/prd_r01.py和training/train.py属于历史实现，当前正式实验不从它们启动。r01.machine.json和training/prd_r01模块仍被调用，旧名称不是删除依据。Dockerfile和install.sh保留为上游环境参考，不自动用于重装当前PRD环境。

## 当前比较协议

- 从对应预训练底座与新LoRA／分类头开始；CLIP基准rank/alpha16、trainBS32/testBS64、两槽主头。改变骨干或配置时须设置匹配对照。
- FF++官方train更新参数，固定官方val帧AUC选点；严格创新高才更新最佳，持平不更新。连续5轮未改善早停，最多30轮。
- 停止后恢复验证集最佳权重，再评测CDFv2、DFDC、DFD、UADFV；主指标为四域帧AUC等权平均，视频指标先平均帧概率。
- 当前默认增强后在线重构。旧common.json的部分值会被正式入口覆盖，旧缓存协议不能当作等价替换。
- 已经参与方法开发的目标域结果属于开发评估，不声明独立最终测试。

## 本方向已有入口

- 仅原图／经典VAE：[scripts/早停基线.py](scripts/早停基线.py)。
- [VQGAN单位方向残差早停.py](scripts/VQGAN单位方向残差早停.py)。
- [VQGAN原图辅助监督早停.py](scripts/VQGAN原图辅助监督早停.py)。
- [VQGAN绝对值残差早停.py](scripts/VQGAN绝对值残差早停.py)。
- [学习型压缩重构早停.py](scripts/学习型压缩重构早停.py)。

单位方向残差、分别归一化后的特征差、绝对值差保持不同公式。原图辅助监督训练时增加系数1辅助CE，终测只用主头；压缩使用全层激活重算，原图辅助使用偶数层重算。旧BM3D/Restormer/扩散实现用于追溯，不自动转为新训练队列。

## 使用边界

同一方向复用工作树，用独立run-id区分运行；只有并发修改不同代码时才增加隔离目录。运行期间不修改运行文件或切换分支。工作树共享环境、数据和GPU，资源检查与任务授权仍须单独完成。

本机r01.machine.json、.state、数据集、权重和原始日志不纳入Git。历史产物迁移关系与完整实验记录在Windows本地dev；已发布摘要中的原始运行片段已经剔除，必要指标、结论及产物哈希保留。历史记录中的原代码提交号通过[发布提交映射](docs/发布提交映射.json)追溯，不改写旧运行身份。

现有环境中的CPU合同检查：
```bash
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover -s tests/r01 -p test_fresh_contract.py -v
```
该检查不构成正式训练或GPU预检许可。模型训练与GPU计算仅在对应实验明确获批后进行。

## 来源与许可

基于[SCLBD/DeepfakeBench](https://github.com/SCLBD/DeepfakeBench)及其[NeurIPS 2023论文](https://arxiv.org/abs/2307.01426)；上游作者为Zhiyuan Yan、Yong Zhang、Xinhang Yuan、Siwei Lyu、Baoyuan Wu。许可见[LICENSE](LICENSE)，第三方算子目录保留各自来源与许可证。
