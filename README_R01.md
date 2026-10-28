# PRD R01 公共运行底座

代码起点：`b5fb097740a0815f468faac7d6feac402b29e518`。
本目录属于新阶段研发工作树；旧服务器仓库、分支、实验记录及权重保持原状。
本文件仅说明运行代码，协作总表、任务认领和研究Skill仍只在Windows本地dev。

## 唯一入口

使用既有PRD环境的Python执行 `scripts/prd_r01.py`。默认只检查，不训练。
历史 `training/train.py` 在本分支直接拒绝启动，避免误入旧目标集选点/自动早停流程。
旧分支的入口没有修改。

```bash
python scripts/prd_r01.py --check --variant original
python scripts/prd_r01.py --check --variant vae
python scripts/prd_r01.py --check --variant vae_diffusion
python -m unittest discover -s tests/r01 -v
```

机器路径写入被Git忽略的 `r01.machine.json`，结构见example。
只有数据根目录、CLIP/VAE权重路径、输出根目录可按机器变化；不在这里改学习率、batch或模型。
不依赖GitHub，不自动下载权重，不回落到NAS或CPU训练。

## 此次实际提供的能力

- 统一三槽输入 `[z,r_a,r_b]`，signed特征差；共同MLP为3072→1024→2（CLIP维度1024）。
- original为 `[z,0,0]`；vae为 `[z,r_V,0]`；repeat_vae为 `[z,r_V,r_V]`。
- original不加载/计算VAE；重复VAE只计算一次，再复用响应作为容量参考。
- CLIP+LoRA rank32/alpha32/dropout .05继承起点；不声称原论文rank16的严格复现。
- 固定算子与输入的预处理均显式实现；VAE使用posterior mode、256往返、CUDA FP16；编码器与头FP32，无AMP。
- 附加视图通过fork_rng保存/恢复Torch随机状态，避免仅因多调用几路而改变后续主路径RNG。
- LMDB→OpenCV→CPU增强/归一化→pin_memory→non_blocking传输。坏键直接失败，不换成第0张图。
- 保留旧基本增强分布，去掉224尺寸下恒等resize及PIL往返；这仍是新协议，不能直接拼接旧训练结果。
- 训练每轮只验证source-val；保存每轮源验证logits。用宏视频balancedCE选点，平局保留较早轮。
- target/source-test仅在source_best与terminal评估；视频用平均logits，宏指标为逐域等权均值。
- 每100步输出loss/耗时，不在每batch计算sklearn AUC/AP/EER。host_data_wait是主线程取数等待，不是GPU分段计时。
- 独立run目录拒绝覆盖，有限墙钟检查、失败留证，不自动重试、追加队列或制作备份。

## 仍未准备的部分

diffusion、restoration、DINOv2和SigLIP只登记明确的配置名，检查时会报缺少实现/权重/工作点。
不会把它们替换为identity/VAE/CLIP。需要在后续R01-P01中具体选定并实现、验证。
当前模板中的数据清单、epochs、seed、墙钟预算、判据未冻结，正式训练默认拒绝。
本次没有下载新模型，也没有完成全量训练或跨硬件GPU等价验收。

## Smoke与正式训练是两件事

```bash
python scripts/prd_r01.py --smoke --variant vae --device cuda:0
```

smoke仅取两张FF++源训练图，执行前向/反向与重复eval：0 optimizer更新、0 checkpoint、0目标集成绩。
无GPU机器只验收源码、CPU单元测试与路径，不自动在CPU跑大模型。
执行smoke前仍须核对当前资源与授权。

正式入口接受明确协议；配置模板为draft，不能直接启动：

```bash
python scripts/prd_r01.py --train --variant vae --protocol /absolute/protocol.json --run-id unique-run-id
```

协议的approved只是执行前记录字段，不替代用户授权、总表认领和现场检查。
新组从预训练初始化，不加载任何旧Q实验最优权重。
正式代码必须干净并先提交；运行失败保留原目录，不覆盖重启。

## 冻结清单格式

```json
{
  "role": "source_val",
  "rows": [
    {"domain": "FaceForensics++", "key": "原JSON中的LMDB键", "label": 0,
     "video_id": "包含数据子类的唯一视频标识", "source_ids": ["源视频编号"]}
  ]
}
```

role为source_train/source_val/source_test/target_dev；必须完整包含真假两类才能评估。
源train/val必须按真实视频来源填写source_ids并无交集；真假/方法可共享来源但不能跨split。
源划分、视频来源关联、清单生成规则及其审核属于执行协议准备，当前代码不会替你猜划分。
manifest保留原始LMDB键，不能用斜杠规范化后的路径替换真实键。
反复使用的CDFv2/DFDCP/DFDC只能称开发域，不因源选点就变成独立最终测试。

## 验证边界

CPU小模型测试验证槽位、复用、梯度、RNG、指标聚合、清单拒绝和合成数据训练闭环。
真实GPU smoke仅覆盖实际执行过的配置与机器；不会将mock或CPU通过写成真实训练验收。
依赖沿用既有环境；已观察的4090版本为torch2.5.1+cu124、transformers4.46.3、
diffusers0.31.0、peft0.13.2、albumentations1.3.1。不自动安装/升级环境。
