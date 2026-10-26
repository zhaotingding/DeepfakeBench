# DeepfakeBench 项目报告

> **论文**: A Comprehensive Benchmark of Deepfake Detection (NeurIPS 2023 D&B)
> **核心目标**: 提供一个统一的深度伪造检测基准平台，标准化数据管理、模型训练与评估流程。

---

## 1. 项目架构

```
DeepfakeBench-main/
├── preprocessing/          # 数据预处理（人脸检测/对齐/裁剪 + 数据集重排）
│   ├── preprocess.py       # 视频 → 人脸帧 + 关键点 + mask
│   ├── rearrange.py        # 生成 JSON 索引文件（统一数据加载入口）
│   ├── config.yaml         # 预处理/重排的配置文件
│   ├── dataset2lmdb_test.py# RGB → LMDB 格式转换
│   └── dlib_tools/         # dlib 人脸关键点模型
│
├── training/               # 核心训练与评测框架
│   ├── train.py            # 训练主入口
│   ├── test.py             # 评测主入口（加载权重 → 测试集指标）
│   ├── logger.py           # 日志系统
│   ├── vae.py              # VAE 图像重建工具
│   ├── config/             # YAML 配置
│   │   ├── train_config.yaml / test_config.yaml  # 全局配置（路径/标签/数据增强）
│   │   ├── detector/       # 每个检测器的超参配置（~45个 yaml）
│   │   └── backbone/       # 骨干网络配置
│   ├── detectors/          # 检测器实现（~45 个）
│   ├── dataset/            # 数据集类 + 数据增强/融合工具
│   ├── trainer/            # Trainer（训练循环/验证/checkpoint/swa）
│   ├── networks/           # 骨干网络（Xception/ResNet/EfficientNet/MesoNet）
│   ├── loss/               # 损失函数（交叉熵/对比损失/AM-Softmax 等）
│   ├── optimizor/          # 优化器扩展（SAM）
│   ├── metrics/            # 评估指标（AUC/EER/ACC/AP）+ 注册器
│   ├── pretrained/         # ImageNet 预训练权重
│   ├── weights/            # 训练好的检测器权重
│   └── logs/               # 训练日志与输出
│
├── analysis/               # 分析与可视化脚本
│   ├── auc_fromaug.py      # 数据增强消融 AUC 表格
│   ├── auc_table1/2.py     # 跨数据集泛化 AUC 表
│   ├── curve_draw.py       # ROC/PR 曲线
│   ├── tsne.py             # t-SNE 特征可视化
│   ├── GradCAM.py          # 类激活图可视化
│   ├── frequency.py        # 频域分析
│   ├── model_archi.py      # 模型架构分析
│   └── number_frames.py    # 帧数消融实验
│
├── Dockerfile / install.sh # 环境配置
└── train.sh                # 多 GPU 分布式训练启动脚本
```

---

## 2. 运行流程

### 2.1 环境安装

```bash
conda create -n DeepfakeBench python=3.7.2
conda activate DeepfakeBench
sh install.sh          # 安装所有 Python 依赖
# 或者用 Docker:
docker build -t DeepfakeBench .
docker run --gpus all -itd -v /path/to/repo:/app/ --shm-size 64G DeepfakeBench
```

### 2.2 数据准备

1. **下载数据集**: FF++, Celeb-DF, DFDC, DFDCP, DeeperForensics, UADFV 等 9 个数据集
2. **（可选）预处理**: 配置 `preprocessing/config.yaml` → `python preprocessing/preprocess.py`（人脸检测+对齐+裁剪为 256×256，提取 81 点关键点 + mask）
3. **数据重排**: 配置 `preprocessing/config.yaml` → `python preprocessing/rearrange.py` → 生成 `dataset_json/*.json`
4. **（可选）LMDB**: `python preprocessing/dataset2lmdb_test.py` 将 RGB 帧转为 LMDB 加速 IO

### 2.3 训练

```bash
# 单 GPU
python training/train.py --detector_path ./training/config/detector/xception.yaml

# 多 GPU (DDP)
torchrun --nproc_per_node=4 training/train.py --ddp \
    --detector_path ./training/config/detector/xception.yaml \
    --train_dataset "FaceForensics++" \
    --test_dataset "Celeb-DF-v1" "Celeb-DF-v2"
```

### 2.4 评测

```bash
python training/test.py \
    --detector_path ./training/config/detector/xception.yaml \
    --test_dataset "Celeb-DF-v1" "Celeb-DF-v2" \
    --weights_path ./training/weights/xception_best.pth
```

---

## 3. 核心模块说明

### 3.1 `detectors/` — 检测器（~45 个，继承 `AbstractDetector`）

每个检测器需实现：`build_backbone()`, `build_loss()`, `features()`, `classifier()`, `forward()`, `get_losses()`, `get_train_metrics()`

| 类别 | 检测器 |
|------|--------|
| **朴素基线** (5) | Xception, Meso4, MesoInception, ResNet34, EfficientNetB4 |
| **空间域** (19) | Face X-ray, FWA, FFD, CORE, RECCE, UCF, LRL, IID, RFM, SIA, SLADD, UIA-ViT, CLIP, SBI, PCL-I2G, Multi-Attention, LSDA, Capsule, SRM |
| **频域** (3) | F3Net, SPSL, SRM |
| **视频** (8) | TALL, I3D, STIL, FTCN, X-CLIP, TimeSformer, VideoMAE, AltFreezing |
| **自定义** (8+) | LoRA, LoRA1/2, Our, Aour, Aour1, Effort, Aeffort, Aeffort1, CLIP-Frozen, DIRE |

通过 `metrics/registry.py` 中的 `DETECTOR` 注册器统一管理。

### 3.2 `dataset/` — 数据加载

- **`abstract_dataset.py`**: 基类 `DeepfakeAbstractBaseDataset`，支持 RGB/LMDB 两种后端，从 JSON 索引文件读取数据，支持视频级/帧级采样
- **`ff_blend.py`**: Face X-ray 风格混合数据增强
- **`fwa_blend.py`**: FWA 风格混合数据增强
- **`sbi_dataset.py`**: SBI 自混合数据增强
- **`lsda_dataset.py`**: LSDA 潜在空间增广
- **`pair_dataset.py`**: 成对采样（正负样本均衡）
- **`iid_dataset.py`**: IID 数据集
- **`lrl_dataset.py`**: Local-relation 数据集
- **`tall_dataset.py`** / **`I2G_dataset.py`**: 特定检测器数据集

### 3.3 `trainer/trainer.py` — 训练器

- 管理训练/验证循环、梯度更新、学习率调度
- 支持 SWA（随机权重平均）和 checkpoint 保存
- 支持 DDP 分布式训练
- TensorBoard 日志记录
- 自动根据最佳 AUC/EER 保存模型

### 3.4 `networks/` — 骨干网络

| 网络 | 来源 |
|------|------|
| Xception | FaceForensics++ 经典骨干 |
| EfficientNetB4 | 轻量高效 |
| ResNet34 | 朴素 CNN 基线 |
| Meso4 / MesoInception4 | 轻量浅层网络 |
| Xception_SLADD | SLADD 方法改进版 |

通过 `BACKBONE` 注册器统一管理。

### 3.5 `loss/` — 损失函数

| 损失函数 | 用途 |
|----------|------|
| CrossEntropyLoss | 标准分类 |
| BCELoss | 二分类 |
| AMSoftmaxLoss | 大间隔分类 |
| ContrastiveLoss / SupConLoss | 对比学习 |
| L1Loss / VGGLoss / IDLoss | 重建/感知损失 |
| CapsuleLoss / PatchConsistencyLoss / RegionIndependentLoss | 特殊检测器 |
| MLLoss / JS_Loss / ConsistencyCos | 多标签/一致性 |

通过 `LOSSFUNC` 注册器统一管理。

### 3.6 `metrics/` — 评估系统

- `registry.py`: 定义 `BACKBONE`, `DETECTOR`, `TRAINER`, `LOSSFUNC` 四个全局注册器
- `base_metrics_class.py`: 提供 `Metrics_batch`（批次内 AUC/EER/ACC/AP）和 `Recorder`（全局累积）
- `utils.py`: 格式化输出评估指标

### 3.7 `preprocessing/` — 数据预处理管线

- **`preprocess.py`**: 使用 dlib 检测人脸 → 5 点对齐 → 仿射变换裁剪 256×256 → 提取 81 点关键点 → 保存为 PNG/NumPy
- **`rearrange.py`**: 扫描预处理后的帧文件 → 按数据集/类别/视频组织 → 生成 JSON 索引文件
- **`dataset2lmdb_test.py`**: 将 PNG 帧转换为 LMDB 数据库（加速训练 IO）
- **`config.yaml`**: 配置预处理参数（数据集名、路径、压缩等级、采样模式等）

### 3.8 `analysis/` — 分析工具

| 文件 | 功能 |
|------|------|
| `auc_fromaug.py` | 数据增强消融 — 绘制 AUC 热力图 |
| `auc_table1_fromrecord.py` | 跨数据集泛化 AUC 表格（论文 Table 1） |
| `auc_table2_fromrecord.py` | 跨数据集泛化 AUC 表格（论文 Table 2） |
| `curve_draw.py` | ROC / PR 曲线绘制 |
| `tsne.py` | t-SNE 降维可视化 |
| `GradCAM.py` | 类激活图可视化 |
| `frequency.py` | 频域分析 |
| `model_archi.py` | 模型参数量/FLOPs 分析 |
| `number_frames.py` | 帧数对性能影响分析 |

---

## 4. 关键设计模式

1. **注册器模式** (`registry.py`): `DETECTOR`, `BACKBONE`, `LOSSFUNC` 三大注册器 → 通过 YAML 配置名即可动态实例化任意组件
2. **抽象基类** : `AbstractDetector` 和 `DeepfakeAbstractBaseDataset` 定义统一接口，新增检测器只需实现抽象方法
3. **YAML 驱动配置** : 每个检测器一个独立 YAML，包含模型超参、数据路径、训练参数、数据增强策略，实现配置与代码分离
4. **多数据集联合训练** : 通过 JSON 索引 + `label_dict` 统一不同数据集的标签映射

---

## 5. 支持的数据集

| 数据集 | 真实视频 | 伪造视频 | 伪造方法 |
|--------|---------|---------|----------|
| FaceForensics++ | 1000 | 4000 | 4 种 |
| FaceShifter | 1000 | 1000 | 1 种 |
| DeepfakeDetection | 363 | 3000 | 5 种 |
| DFDC (Preview) | 1131 | 4119 | 2 种 |
| DFDC | 23654 | 104500 | 8 种 |
| Celeb-DF-v1/v2 | 998 | 6434 | 1 种 |
| DFDCP | — | — | 2 种 |
| DeeperForensics-1.0 | 50000 | 10000 | 1 种 |
| UADFV | 49 | 49 | 1 种 |

---

## 6. 常用命令速查

```bash
# 数据预处理
cd preprocessing && python preprocess.py          # 人脸裁剪
cd preprocessing && python rearrange.py           # 生成 JSON 索引
cd preprocessing && python dataset2lmdb_test.py   # 转 LMDB 格式

# 训练
python training/train.py --detector_path ./training/config/detector/xception.yaml
torchrun --nproc_per_node=4 training/train.py --ddp --detector_path ./training/config/detector/xception.yaml

# 评测
python training/test.py --detector_path ./training/config/detector/xception.yaml --test_dataset "Celeb-DF-v1" --weights_path path/to/best.pth

# GradCAM 可视化
python training/grad_cam.py

# 特征分析
python training/tsne.py
```
