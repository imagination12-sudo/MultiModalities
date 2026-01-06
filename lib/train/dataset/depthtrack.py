import os
import os.path
import torch
import numpy as np
import pandas
import csv
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader_w_failsafe
from lib.train.admin import env_settings
from lib.train.dataset.depth_utils import get_x_frame
import cv2


class DepthTrack(BaseVideoDataset):
    """ DepthTrack dataset.
    """

    def __init__(self, root=None, dtype='rgbcolormap', split='train', image_loader=jpeg4py_loader_w_failsafe): #  vid_ids=None, split=None, data_fraction=None
        """
        args:

            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            vid_ids - List containing the ids of the videos (1 - 20) used for training. If vid_ids = [1, 3, 5], then the
                    videos with subscripts -1, -3, and -5 from each class will be used for training.
            # split - If split='train', the official train split (protocol-II) is used for training. Note: Only one of
            #         vid_ids or split option can be used at a time.
            # data_fraction - Fraction of dataset to be used. The complete dataset is used by default

            root     - path to the lasot depth dataset.
            dtype    - colormap or depth,, colormap + depth
                        if colormap, it returns the colormap by cv2,
                        if depth, it returns [depth, depth, depth]
        """
        if split == 'train':
            root = env_settings().depthtrack_dir_train if root is None else root
        else:
            root = env_settings().depthtrack_dir_val if root is None else root
        super().__init__('DepthTrack', root, image_loader)

        self.dtype = dtype  # colormap or depth
        self.split = split
        self.sequence_list = self._build_sequence_list()

        self.seq_per_class, self.class_list = self._build_class_list()
        self.class_list.sort()
        self.class_to_id = {cls_name: cls_id for cls_id, cls_name in enumerate(self.class_list)}

    def _build_sequence_list(self):

        ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
        if self.split == 'train':
            file_path = os.path.join(ltr_path, 'data_specs', 'TrainSet_list.txt')
        else:
            file_path = os.path.join(ltr_path, 'data_specs', 'ValidationSet_list.txt')
        # sequence_list = pandas.read_csv(file_path, header=None, squeeze=True).values.tolist()
        sequence_list = pandas.read_csv(file_path, header=None).iloc[:, 0].tolist()
        return sequence_list

    def _build_class_list(self):
        seq_per_class = {}
        class_list = []
        for seq_id, seq_name in enumerate(self.sequence_list):
            class_name = seq_name.split('_')[0]

            if class_name not in class_list:
                class_list.append(class_name)

            if class_name in seq_per_class:
                seq_per_class[class_name].append(seq_id)
            else:
                seq_per_class[class_name] = [seq_id]

        return seq_per_class, class_list

    def get_name(self):
        return 'depthtrack'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_num_classes(self):
        return len(self.class_list)

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    # def _read_bb_anno(self, seq_path):
    #     bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
    #     gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=True, low_memory=False).values
    #     return torch.tensor(gt)
    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth_rect.txt")  # ← 修改这里！
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=True, low_memory=False).values
        return torch.tensor(gt)

    def _get_sequence_path(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        return os.path.join(self.root, seq_name)

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)  # xywh just one kind label
        '''
        if the box is too small, it will be ignored
        '''
        # valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        valid = (bbox[:, 2] > 10.0) & (bbox[:, 3] > 10.0)
        visible = valid.clone().byte()

        """ 新增自然语言读取 """
        nlp = self._read_nlp(seq_path)
        return {'bbox': bbox, 'valid': valid, 'visible': visible, 'nlp': nlp}

    # def _get_frame_path(self, seq_path, frame_id):
    #     '''
    #     return depth image path
    #     '''
    #     return os.path.join(seq_path, 'color', '{:08}.jpg'.format(frame_id+1)) , os.path.join(seq_path, 'depth', '{:08}.png'.format(frame_id+1)) # frames start from 1
    def _get_frame_path(self, seq_path, frame_id):
        # frames start from 1
        return (
            os.path.join(seq_path, 'color', '{:08}.jpg'.format(frame_id + 1)),
            os.path.join(seq_path, 'depth', '{:08}.png'.format(frame_id + 1))
        )

    def _get_frame(self, seq_path, frame_id):
        '''
        Return :
            - colormap from depth image
            - 3xD = [depth, depth, depth], 255
            - rgbcolormap
            - rgb3d
            - color
            - raw_depth
        '''
        color_path, depth_path = self._get_frame_path(seq_path, frame_id)
        img = get_x_frame(color_path, depth_path, dtype=self.dtype, depth_clip=True)
        modality = img[:, :, 3:]
        dummy = modality[:, :, 0:1].copy() * 0 + 1
        img = cv2.merge((img[:, :, :3], modality, dummy))

        return img

    def _get_class(self, seq_path):
        # raw_class = seq_path.split('/')[-2]
        # return raw_class
        return self.split

    """ 新增自然语言读取函数 """
    def _read_nlp(self, seq_path):
        nlp_file = os.path.join(seq_path, "nlp.txt")
        with open(nlp_file, 'r', encoding='utf-8') as file:
            nlp_text = file.read().strip()  # 读取整个文件内容并去除首尾空白字符
        return nlp_text  # 返回整个文本内容

    def get_class_name(self, seq_id):
        depth_path = self._get_sequence_path(seq_id)
        obj_class = self._get_class(depth_path)

        return obj_class

    def get_sequence_nlp(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        nlp = self._read_nlp(seq_path)
        return nlp

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)

        obj_class = self._get_class(seq_path)

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            if key == 'nlp':
                anno_frames[key] = value
            else:
                anno_frames[key] = [value[f_id, ...].clone() for ii, f_id in enumerate(frame_ids)]

        frame_list = [self._get_frame(seq_path, f_id) for ii, f_id in enumerate(frame_ids)]

        object_meta = OrderedDict({'object_class_name': obj_class,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})
        return frame_list, anno_frames, object_meta


# class DepthTrackTrain(BaseVideoDataset):
#     """ DepthTrack dataset.
#     """
#
#     def __init__(self, root=None, dtype='rgbcolormap', split='train', image_loader=jpeg4py_loader_w_failsafe):
#         """
#         args:
#             ...
#         """
#         root = env_settings().depthtrack_dir if root is None else root
#         super().__init__('DepthTrack', root, image_loader)
#
#         self.dtype = dtype  # colormap or depth
#         self.split = split
#
#         # 修改1：从TrainSet_list或ValidationSet_list加载序列列表
#         list_file = f"{self.split.capitalize()}Set_list"  # 根据split选择文件名
#         list_path = os.path.join(root, list_file)
#         with open(list_path, 'r') as f:
#             self.sequence_list = [line.strip() for line in f.readlines()]
#
#         self.seq_per_class, self.class_list = self._build_class_list()
#         self.class_list.sort()
#         self.class_to_id = {cls_name: cls_id for cls_id, cls_name in enumerate(self.class_list)}
#
#     def _build_sequence_list(self):
#         # 已在__init__中处理，此处不再需要
#         pass
#
#     def _build_class_list(self):
#         seq_per_class = {}
#         class_list = []
#         for seq_id, seq_name in enumerate(self.sequence_list):
#             # 修改2：根据实际路径格式获取类别名称
#             class_name = seq_name.split('/')[0]
#
#             if class_name not in class_list:
#                 class_list.append(class_name)
#
#             if class_name in seq_per_class:
#                 seq_per_class[class_name].append(seq_id)
#             else:
#                 seq_per_class[class_name] = [seq_id]
#
#         return seq_per_class, class_list
#
#     # 其他方法保持不变...
#
#     def _get_sequence_path(self, seq_id):
#         # 修改3：修正路径拼接方式
#         seq_name = self.sequence_list[seq_id]
#         return os.path.join(self.root, seq_name)  # 注意：这里不需要添加split，因为序列列表已经包含了完整路径
#
#     def _read_bb_anno(self, seq_path):
#         # 修改4：尝试多种可能的标注文件名
#         bb_anno_file = os.path.join(seq_path, "groundtruth_rect.txt")
#         alt_files = ["groundtruth.txt", "gt.txt"]
#         if not os.path.exists(bb_anno_file):
#             for alt in alt_files:
#                 alt_path = os.path.join(seq_path, alt)
#                 if os.path.exists(alt_path):
#                     bb_anno_file = alt_path
#                     break
#             else:
#                 raise FileNotFoundError(f"No annotation file found in {seq_path}")
#
#         gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=True).values
#         return torch.tensor(gt)
#
#     def _get_frame_path(self, seq_path, frame_id):
#         color_path = os.path.join(seq_path, 'color', '{:08}.jpg'.format(frame_id + 1))
#         depth_path = os.path.join(seq_path, 'depth', '{:08}.png'.format(frame_id + 1))
#
#         # 修改5：检查文件是否存在
#         if not os.path.exists(color_path):
#             raise FileNotFoundError(f"Color image not found: {color_path}")
#         if not os.path.exists(depth_path):
#             raise FileNotFoundError(f"Depth image not found: {depth_path}")
#
#         return color_path, depth_path
#
#     def _get_class(self, seq_path):
#         # 修改6：返回正确的类别名称
#         return seq_path.split('/')[-2]  # 获取父目录作为类别名称