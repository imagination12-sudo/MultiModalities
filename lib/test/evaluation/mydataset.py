import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os


class MyDataset(BaseDataset):
    """
    自定义多模态数据集：RGB + Depth + Language Description
    数据结构：
        your_dataset/
        ├── 001/
        │   ├── color/      # RGB frames
        │   ├── depth/      # Depth frames
        │   ├── groundtruth.txt  # [x,y,w,h] per line (or just first line)
        │   └── nlp.txt     # Single sentence description
        └── ...
    """

    def __init__(self, load_all_gt=False):
        """
        Args:
            load_all_gt (bool): 若为 True，加载完整 groundtruth；否则只加载第一帧（用于初始化）
        """
        super().__init__()
        self.base_path = self.env_settings.my_data_path
        self.load_all_gt = load_all_gt
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        seq_dir = os.path.join(self.base_path, sequence_name)

        # === 1. Load groundtruth ===
        gt_path = os.path.join(seq_dir, 'groundtruth.txt')
        if not os.path.isfile(gt_path):
            raise FileNotFoundError(f"groundtruth.txt not found in {seq_dir}")

        gt_rects = load_text(gt_path, delimiter=',', dtype=np.float64)
        if gt_rects.ndim == 1:
            gt_rects = gt_rects[np.newaxis, :]

        if not self.load_all_gt:
            ground_truth_rect = gt_rects[0:1]  # (1, 4)
        else:
            ground_truth_rect = gt_rects

        # === 2. Load language description ===
        nlp_path = os.path.join(seq_dir, 'nlp.txt')
        if not os.path.isfile(nlp_path):
            raise FileNotFoundError(f"nlp.txt not found in {seq_dir}")

        with open(nlp_path, 'r', encoding='utf-8') as f:
            language_description = f.read().strip()

        # === 3. Load RGB and Depth frames ===
        color_dir = os.path.join(seq_dir, 'color')
        depth_dir = os.path.join(seq_dir, 'depth')

        if not os.path.isdir(color_dir):
            raise NotADirectoryError(f"'color' folder missing in {seq_dir}")
        if not os.path.isdir(depth_dir):
            raise NotADirectoryError(f"'depth' folder missing in {seq_dir}")

        color_frames = sorted([f for f in os.listdir(color_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        depth_frames = sorted([f for f in os.listdir(depth_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        if len(color_frames) != len(depth_frames):
            raise ValueError(
                f"Frame count mismatch in {sequence_name}: RGB={len(color_frames)}, Depth={len(depth_frames)}")

        color_paths = [os.path.join(color_dir, f) for f in color_frames]
        depth_paths = [os.path.join(depth_dir, f) for f in depth_frames]
        frames_list = list(zip(color_paths, depth_paths))

        # ✅ 直接构造 init_info 字典（不再用函数！）
        init_info_dict = {
            'init_bbox': ground_truth_rect[0].tolist(),  # 第一帧 bbox
            'language': language_description
        }

        # === 4. Construct Sequence ===
        seq = Sequence(
            name=sequence_name,
            frames=frames_list,
            dataset='mydataset',
            ground_truth_rect=ground_truth_rect,
            language_description=language_description
        )

        # ✅ 关键修改：存储为普通字典属性，而非函数
        seq.init_info = init_info_dict  # ← 直接赋值 dict，可 pickle！

        return seq

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        """自动发现所有合法序列目录（支持 001, 01, 1, test_01 等）"""
        all_items = os.listdir(self.base_path)
        sequence_dirs = []
        for item in all_items:
            full_path = os.path.join(self.base_path, item)
            if os.path.isdir(full_path):
                # 检查是否包含必要子目录和文件
                if (os.path.isdir(os.path.join(full_path, 'color')) and
                    os.path.isdir(os.path.join(full_path, 'depth')) and
                    os.path.isfile(os.path.join(full_path, 'groundtruth.txt')) and
                    os.path.isfile(os.path.join(full_path, 'nlp.txt'))):
                    sequence_dirs.append(item)
        return sorted(sequence_dirs)