import numpy as np
import multiprocessing
import os
import sys
from itertools import product
from collections import OrderedDict
from lib.test.evaluation import Sequence, Tracker
import torch


def _save_tracker_output(seq: Sequence, tracker: Tracker, output: dict):
    """Saves only the target bounding boxes in the format: x,y,w,h per line, comma-separated, no spaces."""

    # 确定结果根目录
    if not os.path.exists(tracker.results_dir):
        os.makedirs(tracker.results_dir)

    # 构建 dataset 子目录（如 vtuav_st）
    if seq.dataset in ['trackingnet', 'got10k', 'vtuav_st', 'vtuav_lt']:
        dataset_dir = os.path.join(tracker.results_dir, seq.dataset)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

    # 提取纯序列名（去掉路径前缀，如 'vtuav_st/001' → '001'）
    pure_seq_name = seq.name.split('/')[-1]

    # 构建序列专属文件夹路径：results/vtuav_st/001/
    seq_dir = os.path.join(tracker.results_dir, seq.dataset, pure_seq_name)
    os.makedirs(seq_dir, exist_ok=True)

    # 定义保存 bbox 的函数（逗号分隔，无空格，保留一位小数）
    def save_bb(file, data):
        data = np.array(data).astype(float)
        np.savetxt(file, data, delimiter=',', fmt='%.1f')

    # 只处理 'target_bbox'
    if 'target_bbox' in output and output['target_bbox']:
        data = output['target_bbox']

        # 多目标情况（一般不会出现在 VTUAV 单目标？但保留兼容性）
        if isinstance(data[0], (dict, OrderedDict)):
            # 如果是多目标，只取第一个目标（或按需处理），但 VTUAV 是单目标
            # 这里我们按你的要求：只输出一个文件，所以假设单目标
            # 或者你可以选择跳过多目标情况
            print(f"Warning: Multi-object detected in {seq.name}, but only saving first object.")
            # 提取第一个 object 的轨迹（假设 object_ids 有序）
            obj_id = list(data[0].keys())[0]
            single_data = [frame[obj_id] for frame in data]
            bbox_file = os.path.join(seq_dir, f"{pure_seq_name}.txt")
            save_bb(bbox_file, single_data)
        else:
            # 单目标：直接保存
            bbox_file = os.path.join(seq_dir, f"{pure_seq_name}.txt")
            save_bb(bbox_file, data)
    else:
        # 如果没有 target_bbox，创建空文件或跳过
        bbox_file = os.path.join(seq_dir, f"{pure_seq_name}.txt")
        open(bbox_file, 'w').close()  # 创建空文件
        print(f"Warning: No target_bbox for {seq.name}, created empty file.")


def run_sequence(seq: Sequence, tracker: Tracker, debug=False, num_gpu=8):
    """Runs a tracker on a sequence."""
    '''2021.1.2 Add multiple gpu support'''
    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        torch.cuda.set_device(gpu_id)
    except:
        pass

    def _results_exist():
        if seq.object_ids is None:
            if seq.dataset in ['trackingnet', 'got10k']:
                base_results_path = os.path.join(tracker.results_dir, seq.dataset, seq.name)
                bbox_file = '{}.txt'.format(base_results_path)
            else:
                bbox_file = '{}/{}.txt'.format(tracker.results_dir, seq.name)
            return os.path.isfile(bbox_file)
        else:
            bbox_files = ['{}/{}_{}.txt'.format(tracker.results_dir, seq.name, obj_id) for obj_id in seq.object_ids]
            missing = [not os.path.isfile(f) for f in bbox_files]
            return sum(missing) == 0

    if _results_exist() and not debug:
        print('FPS: {}'.format(-1))
        return

    print('Tracker: {} {} {} ,  Sequence: {}'.format(tracker.name, tracker.parameter_name, tracker.run_id, seq.name))

    output = tracker.run_sequence(seq, debug=debug)
    sys.stdout.flush()
    if isinstance(output['time'][0], (dict, OrderedDict)):
        exec_time = sum([sum(times.values()) for times in output['time']])
        num_frames = len(output['time'])
    else:
        exec_time = sum(output['time'])
        num_frames = len(output['time'])

    print('FPS: {}'.format(num_frames / exec_time))

    if not debug:
        _save_tracker_output(seq, tracker, output)


def run_dataset(dataset, trackers, debug=False, threads=0, num_gpus=8):
    """Runs a list of trackers on a dataset.
    args:
        dataset: List of Sequence instances, forming a dataset.
        trackers: List of Tracker instances.
        debug: Debug level.
        threads: Number of threads to use (default 0).
    """
    multiprocessing.set_start_method('spawn', force=True)

    print('Evaluating {:4d} trackers on {:5d} sequences'.format(len(trackers), len(dataset)))

    multiprocessing.set_start_method('spawn', force=True)

    if threads == 0:
        mode = 'sequential'
    else:
        mode = 'parallel'

    if mode == 'sequential':
        for seq in dataset:
            for tracker_info in trackers:
                run_sequence(seq, tracker_info, debug=debug)
    elif mode == 'parallel':
        param_list = [(seq, tracker_info, debug, num_gpus) for seq, tracker_info in product(dataset, trackers)]
        with multiprocessing.Pool(processes=threads) as pool:
            pool.starmap(run_sequence, param_list)
    print('Done')
