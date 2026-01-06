"""
UnTrack 模型的训练入口脚本
配置加载 -> 数据构建 -> 模型初始化 -> 损失函数定义 -> 优化器设置 -> 训练启动
"""
import os
# loss function related
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss
# train pipeline related
from lib.train.trainers import LTRTrainer # 训练器类，封装了训练循环、验证、保存、日志、AMP（自动混合精度）等逻辑。
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP # PyTorch官方分布式训练包装器, 将模型包装为 DDP 后，可在多 GPU 上自动同步梯度
# some more advanced functions
from .base_functions import *
# network related
from lib.models.untrack import build_ostrack
from lib.models.untrack import build_untrack
# forward propagation related
from lib.train.actors import UntrackActor
# for import modules
import importlib

from ..utils.focal_loss import FocalLoss

import warnings
from pydantic._internal._generate_schema import UnsupportedFieldAttributeWarning

warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)
warnings.filterwarnings(
    "ignore",
    message=".*An output with one or more elements was resized.*"
)

def run(settings):
    settings.description = 'Training script for untrack'

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    # update settings based on cfg
    update_settings(settings, cfg)

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    # Build dataloaders
    loader_train, loader_val = build_dataloaders(cfg, settings)
    # Create network
    if settings.script_name == "untrack":
        net = build_untrack(cfg)
    else:
        raise ValueError("illegal script name")

    # wrap networks to distributed one
    net.cuda()
    if settings.local_rank != -1:
        # net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)  # add syncBN converter
        """
        如果是多卡训练， 使用DDP包装模型
        """
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")

    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)
    settings.distill = getattr(cfg.TRAIN, "DISTILL", False)
    settings.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "KL")
    # Loss functions and Actors
    if settings.script_name == "untrack":
        # here cls loss and cls weight are not use
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': 1.0, 'nce': 1.0}
        """
        UntrackActor:
        1. 封装 forward 逻辑
        2. 计算加权总损失
        3. 处理 AMP （自动混合精度）、梯度裁剪等
        """
        actor = UntrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    else:
        raise ValueError("illegal script name")

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    use_amp = getattr(cfg.TRAIN, "AMP", False)
    settings.save_epoch_interval = getattr(cfg.TRAIN, "SAVE_EPOCH_INTERVAL", 1)
    settings.save_last_n_epoch = getattr(cfg.TRAIN, "SAVE_LAST_N_EPOCH", 1)

    if loader_val is None:
        trainer = LTRTrainer(actor, [loader_train], optimizer, settings, lr_scheduler, use_amp=use_amp)
    else:
        trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler, use_amp=use_amp)
    
    # train process
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)
