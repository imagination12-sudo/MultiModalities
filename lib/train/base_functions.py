import torch
from torch.utils.data.distributed import DistributedSampler
# datasets related
from lib.train.dataset import DepthTrack
from lib.train.data import sampler, opencv_loader, processing, LTRLoader
import lib.train.data.transforms as tfm
from lib.utils.misc import is_main_process


def update_settings(settings, cfg):
    settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
    settings.search_area_factor = {'template': cfg.DATA.TEMPLATE.FACTOR,
                                   'search': cfg.DATA.SEARCH.FACTOR}
    settings.output_sz = {'template': cfg.DATA.TEMPLATE.SIZE,
                          'search': cfg.DATA.SEARCH.SIZE}
    settings.center_jitter_factor = {'template': cfg.DATA.TEMPLATE.CENTER_JITTER,
                                     'search': cfg.DATA.SEARCH.CENTER_JITTER}
    settings.scale_jitter_factor = {'template': cfg.DATA.TEMPLATE.SCALE_JITTER,
                                    'search': cfg.DATA.SEARCH.SCALE_JITTER}
    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
    settings.print_stats = None
    settings.batchsize = cfg.TRAIN.BATCH_SIZE
    settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE
    settings.fix_bn = getattr(cfg.TRAIN, "FIX_BN", False) # add for fixing base model bn layer


def names2datasets(name_list: list, settings, image_loader):
    assert isinstance(name_list, list)
    datasets = []
    for name in name_list:
        assert name in ["LASOT", "GOT10K_vottrain", "GOT10K_votval", "GOT10K_train_full", "GOT10K_official_val", "COCO17", "VID", "TRACKINGNET",
                        "DepthTrack_train", "DepthTrack_LasHeR", "DepthTrack_LasHeR_VisEvent", "DepthTrack_val", "LasHeR_all", "LasHeR_train", "LasHeR_val", "DepthTrack_LasHeR_val","VisEvent", "LasHeR_VisEvent", "DepthTrack_VisEvent"]
        if name == "DepthTrack_train":
            datasets.append(DepthTrack(settings.env.depthtrack_dir_train, dtype='rgbcolormap', split='train'))
        if name == "DepthTrack_val":
            datasets.append(DepthTrack(settings.env.depthtrack_dir_val, dtype='rgbcolormap', split='val'))
    print('dataset list', len(datasets))
    return datasets


def build_dataloaders(cfg, settings):
    # Data transform
    # Note: for multimodal data, ToGrayscale and Normalize need modify
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                    tfm.RandomHorizontalFlip(probability=0.5))

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                    tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    # The tracking pairs processing module
    output_sz = settings.output_sz
    search_area_factor = settings.search_area_factor

    data_processing_train = processing.UntrackProcessing(search_area_factor=search_area_factor,
                                                       output_sz=output_sz,
                                                       center_jitter_factor=settings.center_jitter_factor,
                                                       scale_jitter_factor=settings.scale_jitter_factor,
                                                       mode='sequence',
                                                       transform=transform_train,
                                                       joint_transform=transform_joint,
                                                       settings=settings)

    data_processing_val = processing.UntrackProcessing(search_area_factor=search_area_factor,
                                                     output_sz=output_sz,
                                                     center_jitter_factor=settings.center_jitter_factor,
                                                     scale_jitter_factor=settings.scale_jitter_factor,
                                                     mode='sequence',
                                                     transform=transform_val,
                                                     joint_transform=transform_joint,
                                                     settings=settings)

    # Train sampler and loader
    settings.num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)
    settings.num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 1)
    sampler_mode = getattr(cfg.DATA, "SAMPLER_MODE", "causal")
    train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", False)
    print("sampler_mode", sampler_mode)

    dataset_train = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, opencv_loader),
                                            p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
                                            samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
                                            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                            num_template_frames=settings.num_template, processing=data_processing_train,
                                            frame_sample_mode=sampler_mode, train_cls=train_cls, max_seq_len = cfg.MODEL.LANGUAGE.BERT.MAX_QUERY_LEN, bert_path=cfg.MODEL.LANGUAGE.PATH)
    print('datasetname', cfg.DATA.TRAIN.DATASETS_NAME)
    train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None
    shuffle = False if settings.local_rank != -1 else True

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=shuffle,
                             num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=train_sampler)

    # Validation samplers and loaders(visevent no val split)
    if cfg.DATA.VAL.DATASETS_NAME[0] is None:
        loader_val = None
    else:
        dataset_val = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.VAL.DATASETS_NAME, settings, opencv_loader),
                                            p_datasets=cfg.DATA.VAL.DATASETS_RATIO,
                                            samples_per_epoch=cfg.DATA.VAL.SAMPLE_PER_EPOCH,
                                            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                            num_template_frames=settings.num_template, processing=data_processing_val,
                                            frame_sample_mode=sampler_mode, train_cls=train_cls, max_seq_len = cfg.MODEL.LANGUAGE.BERT.MAX_QUERY_LEN, bert_path=cfg.MODEL.LANGUAGE.PATH)
        # print(f"[DEBUG] Val dataset length: {len(dataset_val)}")
        # if len(dataset_val) == 0:
        #     raise RuntimeError("Validation dataset is EMPTY!")
        # try:
        #     sample = dataset_val[0]
        #     print("[DEBUG] Successfully loaded first val sample!")
        #     print("Sample keys:", list(sample.keys()) if isinstance(sample, dict) else "Not dict")
        # except Exception as e:
        #     print(f"[ERROR] Failed to load val sample 0: {e}")
        #     raise
        val_sampler = DistributedSampler(dataset_val) if settings.local_rank != -1 else None
        loader_val = LTRLoader('val', dataset_val, training=False, batch_size=cfg.TRAIN.BATCH_SIZE,
                            num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=val_sampler,
                            epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL)

    return loader_train, loader_val


def get_optimizer_scheduler(net, cfg):
    train_type = getattr(cfg.TRAIN.PROMPT, "TYPE", "")
    if 'deep' in train_type:
        print("Only training prompt parameters. They are: ")

        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "prompt" in n and p.requires_grad]}
        ]
        print('######  require grad true ######')
        for n, p in net.named_parameters():
            if "prompt" not in n:
                p.requires_grad = False
            else:
                print(n)
    else:
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in net.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
            },
        ]
        if is_main_process():
            print("Learnable parameters are shown below.")
            for n, p in net.named_parameters():
                if p.requires_grad:
                    print(n)

    if cfg.TRAIN.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR,
                                      weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        raise ValueError("Unsupported Optimizer")
    if cfg.TRAIN.SCHEDULER.TYPE == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP_EPOCH)
    elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
                                                            gamma=cfg.TRAIN.SCHEDULER.GAMMA)
    else:
        # lr_scheduler = None
        raise ValueError("Unsupported scheduler")
    return optimizer, lr_scheduler
