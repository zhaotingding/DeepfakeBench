if __name__ == "__main__":
    raise SystemExit("prd-r01 uses scripts/prd_r01.py; legacy target-selected training is disabled in this branch")

# region authorinfo

# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: training code.
# 这是John添加的注释
# 这是笔记本加的注释

# endregion authorinfo

import os
import argparse    # 用于解析命令行参数
from os.path import join
import cv2
import random
import datetime   # 处理日期和时间
import time       # 提供时间相关操作
import yaml       # 处理YAML格式的配置文件
from tqdm import tqdm     # 一个进度条工具
import numpy as np
from datetime import timedelta       # 处理时间间隔
from copy import deepcopy        # 深拷贝数据结构
from PIL import Image as pil_image     # PIL库用于图像处理

import torch
import torch.nn as nn
import torch.nn.parallel     # 用于多GPU并行训练
import torch.backends.cudnn as cudnn      # 提供对NVIDIA cuDNN的支持，加速训练
import torch.utils.data                 # 数据加载相关模块
import torch.optim as optim           # 优化器
from torch.utils.data.distributed import DistributedSampler     # 分布式采样器
import torch.distributed as dist        # 分布式训练相关模块


# 自定义模块
from optimizor.SAM import SAM    # 引入SAM优化器（Sharpness-Aware Minimization）
from optimizor.LinearLR import LinearDecayLR       # 自定义的线性学习率衰减器

from trainer.trainer import Trainer    # 训练器模块
from detectors import DETECTOR         # 检测器模块
from dataset import *                 # 数据集相关模块
from metrics.utils import parse_metric_for_print     # 工具函数，用于格式化评价指标输出
from logger import create_logger, RankFilter      # 日志记录模块wat
from path_config import TRAINING_ROOT, project_path, resolve_data_paths, resolve_output_path
from experiment_metadata import write_run_metadata
from early_stopping import ConsecutiveDeclineStopper
from experiment_summary import write_experiment_summary

# 命令行参数解析器，用于接收训练相关配置
parser = argparse.ArgumentParser(description='Process some paths.') #创建一个 ArgumentParser 类型的对象
parser.add_argument('--detector_path', type=str,#它是在告诉 ArgumentParser：我的程序支持一个叫 --detector_path 的命令行选项。
                    default=str(TRAINING_ROOT / 'config/detector/lora.yaml'),
                    # 这里应该全部使用相对路径，方便后来人复现  ！！！！！
                    # 想要换backbone，先配置config下的detector的配置文件，然后在这里指定路径
                    help='path to detector YAML file')
parser.add_argument("--train_dataset", nargs="+")    # nargs="+"支持多训练数据集 例python train.py --train_dataset FF++ CelebDF DFDC
parser.add_argument("--test_dataset", nargs="+")    # 支持多测试数据集
parser.add_argument('--train_batchSize', type=int, default=None, help='override training batch size')
parser.add_argument('--no-save_ckpt', dest='save_ckpt', action='store_false', default=True) # 是否保存模型检查点 默认保存
parser.add_argument('--no-save_feat', dest='save_feat', action='store_false', default=True) # 是否保存特征 输入--no-save_feat 不保存
parser.add_argument("--ddp", action='store_true', default=False)     # 是否启用分布式数据并行
parser.add_argument('--local_rank', type=int, default=0)         # 分布式训练中的本地进程编号，设备编号
parser.add_argument('--task_target', type=str, default="", help='specify the target of current training task')
args = parser.parse_args()   # 解析命令行参数
torch.cuda.set_device(args.local_rank)     # 设置当前进程的CUDA设备


def init_seed(config):# 初始化随机种子，确保结果的可复现性
    config['manualSeed'] = random.randint(1, 10000) if config['manualSeed'] is None else config['manualSeed']   #如果配置中没有配置随机函数，则随机生成一个种子；随机生成一个种子
    random.seed(config['manualSeed'])    # 使每次运行都输出相同的数
    np.random.seed(config['manualSeed'])
    os.environ['PYTHONHASHSEED'] = str(config['manualSeed'])
    (torch.manual_seed(config['manualSeed']),torch.cuda.manual_seed_all(config['manualSeed'])) if config['cuda'] else None  # 如果启用CUDA，则设置CPU/GPU随机种子
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # train speed is slower after enabling this opts. # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html

def prepare_training_data(config):# 准备训练数据加载器，根据配置文件选择不同的数据集类# Only use the blending dataset class in training
    if 'dataset_type' in config and config['dataset_type'] == 'blend':
        if config['model_name'] == 'facexray':
            train_set = FFBlendDataset(config)   # 使用Face X-ray数据集
        elif config['model_name'] == 'fwa':
            train_set = FWABlendDataset(config)    # 使用FWA数据集
        elif config['model_name'] == 'sbi':
            train_set = SBIDataset(config, mode='train')    # 使用SBI数据集
        elif config['model_name'] == 'lsda':
            train_set = LSDADataset(config, mode='train')     # 使用LSDA数据集这里调用得到了所有数的路径
        else:
            raise NotImplementedError('Only facexray, fwa, sbi, and lsda are currently supported for blending dataset')  # 抛出未实现错误，仅支持facexray/fwa/sbi/lsda混合数据集
    elif 'dataset_type' in config and config['dataset_type'] == 'pair': # 一对有关联的图像，通常用于视频帧对或图像对的训练。
        train_set = pairDataset(config, mode='train')  # Only use the pair dataset class in training
    elif 'dataset_type' in config and config['dataset_type'] == 'iid': # independent / individual sample 风格：每张图自己作为一个训练样本。
        train_set = IIDDataset(config, mode='train')
    elif 'dataset_type' in config and config['dataset_type'] == 'I2G':
        train_set = I2GDataset(config, mode='train')
    elif 'dataset_type' in config and config['dataset_type'] == 'lrl': # Local Relation Learning
        train_set = LRLDataset(config, mode='train')
    else:
        train_set = DeepfakeAbstractBaseDataset(config=config, mode='train')    # 默认数据集

    if config['model_name'] == 'lsda':# 根据模型名称或配置决定是否使用自定义采样器或分布式采样器
        from dataset.lsda_dataset import CustomSampler
        custom_sampler = CustomSampler(num_groups=2*360, n_frame_per_vid=config['frame_num']['train'], batch_size=config['train_batchSize'], videos_per_group=5) # 实例化一个自定义采样器，CustomSampler 可以控制如何在每个批次中选择视频和帧
        train_data_loader = \
            torch.utils.data.DataLoader( # 使用DataLoader加载数据
                dataset = train_set,
                batch_size = config['train_batchSize'],
                num_workers = int(config['workers']),
                sampler = custom_sampler,
                collate_fn = train_set.collate_fn,
            )
    elif config['ddp']:    # 如果启用分布式数据并行
        sampler = DistributedSampler(train_set)    # 使用分布式采样器
        train_data_loader = \
            torch.utils.data.DataLoader(
                dataset = train_set,
                batch_size = config['train_batchSize'],
                num_workers = int(config['workers']),
                collate_fn = train_set.collate_fn,
                sampler = sampler
            )
    else:  # 单GPU或普通训练
        train_data_loader = \
            torch.utils.data.DataLoader(
                dataset = train_set,
                batch_size = config['train_batchSize'],
                shuffle = True,
                num_workers = int(config['workers']), #控制多少个子进程并行准备数据
                collate_fn = train_set.collate_fn, # 这时默认拼接可能失败，因此 Dataset 自己提供：collate_fn 告诉 DataLoader 应该怎样把单样本合并成 batch。
                )
    return train_data_loader #  注意这里返回的不是全部图像 Tensor。它返回的是一个可迭代的数据供应器。此创建 DataLoader 时，一般没有立刻把所有图像读入内存。

def prepare_testing_data(config):# 准备测试数据加载器
    def get_test_data_loader(config, test_name):
        config = config.copy()  # 创建配置的副本，防止修改原始配置 create a copy of config to avoid altering the original one
        config['test_dataset'] = test_name  # 原始config保留完整测试集列表 局部config只保存当前测试集名字 specify the current test dataset
        test_set = LRLDataset(config=config, mode='test') if config.get('dataset_type', None) == 'lrl' else DeepfakeAbstractBaseDataset(config=config, mode='test',) # 三元表达式选择测试集：'lrl' 用LRL数据集，否则用默认数据集；mode='test' 很重要
        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset = test_set,
                batch_size = config['test_batchSize'],
                shuffle = False,      # 测试时不打乱数据
                num_workers = int(config['workers']),
                collate_fn = test_set.collate_fn,
                # drop_last = (test_name=='DeepFakeDetection'),
                drop_last = False,   # 保留所有批次
            )
        return test_data_loader
    test_data_loaders = {} #{}为空字典 []为空列表 set()为空集合 ()为空元组
    for one_test_name in config['test_dataset']:# 创建多个测试数据加载器
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders

def choose_optimizer(model, config): # 选择优化器 梯度告诉你往哪个方向走，优化器决定具体怎样走、走多远
    opt_name = config['optimizer']['type']
    if opt_name == 'sgd': 
        optimizer = optim.SGD(
            params = model.parameters(),
            lr = config['optimizer'][opt_name]['lr'],
            momentum = config['optimizer'][opt_name]['momentum'],
            weight_decay = config['optimizer'][opt_name]['weight_decay']
        )
        return optimizer
    elif opt_name == 'adam':
        optimizer = optim.Adam(
            params = model.parameters(),
            lr = config['optimizer'][opt_name]['lr'],
            weight_decay = config['optimizer'][opt_name]['weight_decay'],
            betas = (config['optimizer'][opt_name]['beta1'], config['optimizer'][opt_name]['beta2']),
            eps = config['optimizer'][opt_name]['eps'],
            amsgrad = config['optimizer'][opt_name]['amsgrad'], # AMSGrad 就加了一条规则：二阶矩只允许保留“历史最大值”，不允许往回变小 的主要是改善 Adam 在某些理论场景下的收敛问题，使有效学习率不会因为二阶矩下降而再次异常增大
        )
        return optimizer
    elif opt_name == 'sam':
        optimizer = SAM(model.parameters(), optim.SGD, lr=config['optimizer'][opt_name]['lr'], momentum=config['optimizer'][opt_name]['momentum'])
    else:raise NotImplementedError(f"Optimizer {config['optimizer']} is not implemented")
    return optimizer

def choose_scheduler(config, optimizer):# 学习率调度器 优化器决定怎么更新，scheduler决定学习率如何随训练变化。Scheduler 并不直接更新模型参数 scheduler → optimizer.param_groups[i]['lr'] → 影响下一次 optimizer.step() 的步长
    if config['lr_scheduler'] is None:   # 如果使none，说明不需要学习率调度器
        return None
    elif config['lr_scheduler'] == 'step': # 说明要使用StepLR调度器 作用：每隔固定的 step_size 个 epoch，将学习率乘以一个因子 gamma，用于控制的优化器，每隔多少个 epoch 调整一次学习率，学习率缩放因子
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_step'], gamma=config['lr_gamma'])  
        return scheduler  
    elif config['lr_scheduler'] == 'cosine':     # 如果配置中 lr_scheduler 被设置为 'cosine'，说明要使用CosineAnnealingLR调度器。
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['lr_T_max'], eta_min=config['lr_eta_min'])  # 余弦退火：T_max为周期最大epoch数，eta_min为最小学习率
        return scheduler
    elif config['lr_scheduler'] == 'linear':   # 如果配置中 lr_scheduler 被设置为 'linear'，说明要使用LinearDecayLR调度器。
        scheduler = LinearDecayLR(optimizer, config['nEpochs'], int(config['nEpochs']/4))  # 线性递减学习率：总共的训练 epoch 数，最后四分之一阶段开始线性衰减
        return scheduler
    else:
        raise NotImplementedError(f"Scheduler {config['lr_scheduler']} is not implemented")

def choose_metric(config):# 选择评价指标
    metric_scoring = config['metric_scoring'] #selection metric 用哪个指标挑选最佳 checkpoint
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:raise NotImplementedError(f"metric {metric_scoring} is not implemented")
    return metric_scoring


def main():
    with open(project_path(args.detector_path), 'r', encoding='utf-8') as f:   # 配置优先级：CLI > train_config.yaml > detector YAML
        config = yaml.safe_load(f)
    with open(TRAINING_ROOT / 'config/train_config.yaml', 'r', encoding='utf-8') as f:  # 打开训练配置文件，也就是训练集
        config2 = yaml.safe_load(f)
    if 'label_dict' in config:config2['label_dict']=config['label_dict']
    config.update(config2)# 存在，则 config 中该键的值会被 config2 中对应的值替换。不存在，则会将该键值对添加到 config 中。
    config['local_rank']=args.local_rank  # 配置训练设备
    config['nEpochs'], config['save_feat'] = (0, False) if config['dry_run'] else (config['nEpochs'], config['save_feat'])  # 如果是 dry_run 模式（试运行/测试流程，不进行实际训练）//设置训练轮数为 0//不保存特征数据
    config['train_dataset'] = args.train_dataset if args.train_dataset else config['train_dataset']# 如果从命令行提供了数据集路径参数，则覆盖配置文件中的路径设置 
    config['test_dataset'] = args.test_dataset if args.test_dataset else config['test_dataset']
    config['train_batchSize'] = args.train_batchSize if args.train_batchSize is not None else config['train_batchSize']
    config['task_target'] = args.task_target if args.task_target else config.get('task_target')
    config['save_ckpt'] = args.save_ckpt   # 配置模型保存路径
    config['save_feat'] = args.save_feat    # 配置是否保存训练特征
    config['ddp'] = args.ddp    # 设置分布式训练参数
    resolve_data_paths(config)
    config['log_dir'] = str(resolve_output_path(config['log_dir']))
    if config['ddp']:
        dist.init_process_group(backend='nccl', timeout=timedelta(minutes=30))
  
    # 创建日志文件夹并初始化日志记录器  # create logger
    timenow = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
    if config['ddp']:
        run_time = [timenow]
        dist.broadcast_object_list(run_time, src=0)
        timenow = run_time[0]
    task_str = f"_{config['task_target']}" if config.get('task_target', None) is not None else ""  # dict.get()：安全读取 dict['key']：强制读取
    logger_path = os.path.join(config['log_dir'], config['model_name'] + task_str + '_' + timenow)  # 日志存储目录/日志文件夹名
    if not config['ddp'] or dist.get_rank() == 0:
        os.makedirs(logger_path, exist_ok=False)    # 拒绝复用已有实验输出目录
    if config['ddp']:
        dist.barrier()
    logger = create_logger(os.path.join(logger_path, 'training.log'))   # 创建日志记录器 之后交给Trainer使用
    logger.info(f"Save log to {logger_path}")    # 记录日志文件存储路径
    logger.info("--------------- Configuration ---------------")  # 打印完整的配置信息 # print configuration
    params_string = "Parameters: \n"
    for key, value in config.items():  #items()它里面每个元素是一个二元组 tuple (key, value)
        params_string += f"{key}: {value}" + "\n"
    logger.info(params_string)

    init_seed(config)# 初始化随机种子，确保实验可重复性  # init seed set_seed(1024)# 如果启用 cudnn 加速，设置 benchmark 模式以提升性能
    if config['cudnn']:cudnn.benchmark = True   # set cudnn benchmark if needed
    if config['ddp']:# 如果启用分布式数据并行（DDP），初始化通信进程组
        # dist.init_process_group(backend='gloo')
        logger.addFilter(RankFilter(0))    # 仅记录主进程日志

    train_data_loader = prepare_training_data(config)    # prepare the training data loader
    test_data_loaders = prepare_testing_data(config)  # prepare the testing data loader
    model_class = DETECTOR[config['model_name']] # prepare the model (detector)
    model = model_class(config)   # 实例化模型  这就是大型框架常见的“插件化”设计：
    if not config['ddp'] or dist.get_rank() == 0:
        write_run_metadata(logger_path, config)  # 记录实际 seed 与模型已解析的权重路径
    optimizer = choose_optimizer(model, config)    # prepare the optimizer
    scheduler = choose_scheduler(config, optimizer)  # prepare the scheduler
    metric_scoring = choose_metric(config)  # prepare the metric

    # 开始训练  # start training
    trainer = Trainer(config, model, optimizer, scheduler, logger, metric_scoring, time_now=timenow)# 初始化训练器 # prepare the trainer
    early_stop = config.get('early_stopping', {})
    stopper = ConsecutiveDeclineStopper(
        max_declines=early_stop.get('max_declines', 3),
        lower_is_better=metric_scoring == 'eer',
    ) if early_stop.get('enabled', False) else None
    stop_reason = '正常完成训练计划'
    for epoch in range(config['start_epoch'], config['nEpochs'] + 1):
        trainer.model.epoch = epoch   # 更新模型当前训练的 epoch
        best_metric = trainer.train_epoch(epoch=epoch, train_data_loader=train_data_loader, test_data_loaders=test_data_loaders)  # 这句虽然只有几行，但它大概率触发了绝大多数实际工作 每个epoch训练并测试模型，返回最佳评估指标
        logger.info(f"===> Epoch[{epoch}] end with testing {metric_scoring}: {parse_metric_for_print(best_metric)}!") if best_metric is not None else None  # 如果存在最佳评估指标，记录日志
        # Count only the last complete evaluation of this epoch.
        should_stop = False
        latest = getattr(trainer, 'latest_epoch_metric', None)
        if not config['ddp'] or dist.get_rank() == 0:
            if stopper is not None and latest is not None and latest[0] == epoch:
                should_stop = stopper.update(latest[1])
                logger.info(f'[EarlyStop] epoch={epoch} current_avg_{metric_scoring}={latest[1]:.8f} consecutive_declines={stopper.declines} stop={should_stop}')
        if config['ddp']:
            # All ranks exit together to avoid a distributed training hang.
            stop_flag = torch.tensor(int(should_stop), device=f"cuda:{args.local_rank}")
            dist.broadcast(stop_flag, src=0)
            should_stop = bool(stop_flag.item())
        if should_stop:
            stop_reason = f'自动早停：连续{stopper.max_declines}个epoch末平均指标恶化'
            logger.info(f'[EarlyStop] {stopper.max_declines} consecutive declining epochs; retaining saved best checkpoints.')
            break
        if scheduler is not None:
            scheduler.step()  # 每个 epoch 结束后更新学习率，供下一个 epoch 使用
    logger.info(f"Stop Training on best Testing metric {parse_metric_for_print(best_metric)}")
    if not config['ddp'] or dist.get_rank() == 0:
        for handler in logger.handlers:
            handler.flush()
        summary = write_experiment_summary(os.path.join(logger_path, 'training.log'), stop_reason)
        logger.info(f'Experiment summary saved to {summary}')

    if 'svdd' in config['model_name']:model.update_R(epoch)  # 如果模型为 'svdd' 类型，更新 R 参数    
    for writer in trainer.writers.values():writer.close()   # 关闭 TensorBoard 写入器（释放资源）# close the tensorboard writers
        

if __name__ == '__main__':
    main()
