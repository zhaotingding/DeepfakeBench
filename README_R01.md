# PRD公共基线：重构缓存与配对增强

本代码由原始基底b5fb097及公共准备版本77d9c18继续演化。协作总表、认领和研究Skill仍只在Windows本地dev；服务器保存实际运行代码与产物。旧实验分支、目录和训练资产保留。

## 入口与中文工作树

现有prd-r01将统一改名为“公共基线”；各服务器的“重构算子探索、多响应融合、视觉表征探索”从同一公共提交建立独立分支和工作树。工作树只隔离代码，不隔离GPU、环境、输出或资源锁。
同类内不同配置和必要seed重复共用工作树，以明确提交、配置和唯一中文run-id区分。运行中不修改或切换该工作树。
公共修复应先统一基线，再在空闲工作树快进或合入并记录；同名分支不自动同步。

使用既有PRD Conda环境，执行目录为当前新阶段工作树：
```bash
python scripts/prd_r01.py --check --variant vae
python -m unittest discover -s tests/r01 -v
python scripts/prd_r01.py --cache-smoke --variant vae --device cpu
```

默认check不训练。cache-smoke仅两张真实FF++源图、临时缓存、读回/梯度检查，0optimizer更新和0checkpoint；需有相应资源与授权。显式--smoke仍保留在线接口检查，不是新正式训练协议。
历史training/train.py在新分支拒绝执行，旧分支没有改动。机器路径用忽略的r01.machine.json，见example。数据和预训练模型不随worktree复制，不自动下载/安装或回落CPU。

## 先重构，再配对增强

默认reconstruction_mode=cached_paired；原图先做固定224预处理，VAE经256往返、posterior mode生成重构。缓存保存与在线算子输出相同的FP32、CLIP归一化、NCHW张量，不做JPEG/PNG编码或FP16二次量化。
VAE/CLIP实际预训练权重、输入分辨率、均值方差、工作点与原图字节身份有记录；同输入/算子可跨epoch和配置复用。原图变化、缓存缺帧、清单身份不符、不完整或内容损坏均拒绝，不偷偷回落在线计算。

训练从(A(x),R(A(x)))改为(A(x),A(R(x)))，这是一项明确的新协议，不声称等价于旧训练。
两图经同样的浮点反归一化和[0,1]裁剪后，通过同一次Albumentations参数抽样，统一翻转、旋转、模糊、亮度对比或色调，以及JPEG增强；然后分别归一化。所有配置的原图分支均使用同一paired_float32_v1流程。
浮点流程去掉仅支持uint8的FancyPCA；JPEG仍是预登记训练增强，会按库实现量化，但缓存文件本身不量化。常规评估不做随机增强或上述训练裁剪，直接读取保存的归一化张量。
变换后的鲁棒性评估需要与实际受损输入匹配的缓存，不能套用干净缓存。随机扩散算子将来须单独固定随机策略，目前未实现。

## 按清单生成缓存

```bash
python scripts/生成重构缓存.py --manifest /absolute/train.json --manifest /absolute/val.json --manifest /absolute/target.json --output /absolute/唯一缓存目录 --device cuda:0 --batch-size 16 --max-seconds 3600
```

命令仅示例，时间/资源预算应依据实测批准。清单合并时相同帧必须元数据一致；已有输出目录拒绝覆盖，失败保留生成状态，不自动重试。
目录包含“重构张量.npy”（mmap）、“缓存清单.json”和“生成状态.json”。整库校验在正式启动前一次进行，worker按需映射，训练不逐epoch执行重构。
224RGB FP32每帧约0.574MiB；28,768训练帧约16.1GiB/算子，165,758旧目标帧约93.0GiB/算子，另有索引开销。先核算空间，正式大缓存未由本次小样本验收自动生成。
生成耗时及存储应计入实验总成本；部署遇到新图片仍需算子，不能把缓存读取时间充当完整推理成本。

## 正式训练协议仍须批准

```bash
python scripts/prd_r01.py --train --variant vae --protocol /absolute/approved.json --run-id 4090-重构算子探索-具体方案-日期
```

协议模板保持draft。必须固定数据、epochs/seed、预算、判据、source-val选点，以及：
```json
{
  "reconstruction_mode": "cached_paired",
  "augmentation_protocol": "paired_float32_v1",
  "reconstruction_caches": {"vae": "/absolute/已完成缓存"}
}
```

original不需要缓存；repeat_vae复用同一重构和同一响应。模型端只保留可训练编码器与三槽共同头，不加载VAE；两路编码器仍前向/反向，不缓存变化中的LoRA特征。
CLIP LoRA rank32/alpha32/dropout.05、Adam和batch保持公共版本设置。每epoch一次源验证选点，source_best和terminal才评估完整目标；逐视频平均logits、域等权宏指标。终态保存预测与cache身份，不覆盖旧run。
approved字段不替代用户授权、认领和现场资源检查。当前源码干净并已提交才能正式启动；不自动追加队列。

## 清单与未完成部分

清单role为source_train/source_val/source_test/target_dev，rows含domain、原始LMDB key、label、video_id；源train/val另需source_ids且不重叠。评估含两类；缺图不替换。
正式清单尚未冻结。新扩散、恢复器、DINOv2/SigLIP仍需具体化，代码明确报缺口，不伪装成已实现。
CPU测试覆盖缓存存取、同步增强、worker、RNG、映射、失效拒绝、编码器梯度和合成训练闭环。缓存训练的实际检测效果及GPU完整提速须后续有限对照，不由CPU通过推出。
