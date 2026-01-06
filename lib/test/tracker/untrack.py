import math
from lib.models.untrack import build_untrack
from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os
import vot
import numpy as np
from transformers import BertTokenizer
from lib.test.tracker.data_utils import PreprocessorMM
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
from lib.utils_NLT.misc import NestedTensor


class UnTrack(BaseTracker):
    def __init__(self, params):
        super(UnTrack, self).__init__(params)
        network = build_untrack(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu', weights_only=False)['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = PreprocessorMM()
        self.tokenizer = BertTokenizer.from_pretrained(self.cfg.MODEL.LANGUAGE.PATH)
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        if getattr(params, 'debug', None) is None:
            setattr(params, 'debug', 0)
        self.use_visdom = False #params.debug
        self.debug = params.debug
        self.frame_id = 0
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes

    def _load_7ch_image(self, image_input):
        """
        Load 7-channel image from:
          - (rgb_path, ir_path) tuple → load both and stack
          - np.ndarray (H, W, 3) → assume RGB only (for single-modality datasets)
        Returns:
            np.ndarray of shape (H, W, 7), dtype=np.float32
        """
        if isinstance(image_input, tuple) and len(image_input) == 2:
            rgb_path, ir_path = image_input
            # Load RGB
            rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)  # BGR
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32)
            # Load IR (grayscale)
            ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            # Expand IR to 3 channels
            ir_3ch = np.stack([ir, ir, ir], axis=-1)  # (H, W, 3)
            # Create ones channel
            ones = np.ones((rgb.shape[0], rgb.shape[1], 1), dtype=np.float32)
            # Concatenate to 7 channels
            image_7ch = np.concatenate([rgb, ir_3ch, ones], axis=-1)  # (H, W, 7)
        elif isinstance(image_input, np.ndarray) and image_input.ndim == 3:
            # Assume single-modality (e.g., RGB only)
            H, W = image_input.shape[:2]
            rgb = image_input.astype(np.float32)
            if rgb.shape[2] == 3:
                # Fake IR as grayscale of RGB, or zeros? Here we use grayscale
                ir_gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
                ir_3ch = np.stack([ir_gray, ir_gray, ir_gray], axis=-1)
                ones = np.ones((H, W, 1), dtype=np.float32)
                image_7ch = np.concatenate([rgb, ir_3ch, ones], axis=-1)
            else:
                raise ValueError(f"Unexpected image channel: {rgb.shape}")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        return image_7ch

    def initialize(self, image, info: dict):
        # print("===================================== image", image.shape)

        self.language = info.get('language', None)

        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr  = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr)
        with torch.no_grad():
            self.z_tensor = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.device, template_bbox)

        # ✅ 构造 template_masks（全零）
        H_t, W_t = self.params.template_size, self.params.template_size
        self.template_masks = torch.zeros((1, H_t, W_t), dtype=torch.bool).cuda().unsqueeze(0)  # [1, H, W]
        self.template_patch = NestedTensor(self.z_tensor, self.template_masks)

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def extract_token_from_nlp(self, nlp, seq_length):
        """ use tokenizer to convert nlp to tokens
        param:
            nlp:  a sentence of natural language
            seq_length: the max token length, if token length larger than seq_len then cut it,
            elif less than, append '0' token at the reef.
        return:
            token_ids and token_marks
        """
        nlp_token = self.tokenizer.tokenize(nlp)
        if len(nlp_token) > seq_length - 2:
            nlp_token = nlp_token[0:(seq_length - 2)]
        # build tokens and token_ids
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in nlp_token:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)
        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        return input_ids, input_mask

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1

        # 获取当前帧的语言
        language = getattr(self, 'language', None)


        nl_tokens, nl_masks = self.extract_token_from_nlp(language, self.cfg.MODEL.LANGUAGE.BERT.MAX_QUERY_LEN)

        # 转为 tensor 并增加 batch 维度
        nl_tokens = torch.tensor(nl_tokens).unsqueeze(0).cuda()  # [1, seq_len]
        nl_masks = torch.tensor(nl_masks).unsqueeze(0).cuda()  # [1, seq_len]

        text_data = NestedTensor(nl_tokens, nl_masks)

        # --- 构造 search_masks（全零）---
        H_s, W_s = self.params.search_size, self.params.search_size
        search_masks = torch.zeros((1, H_s, W_s), dtype=torch.bool).cuda().unsqueeze(0) # [1, H, W]


        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)

        with torch.no_grad():
            x_tensor = search
            # merge the template and the search
            # run the transformer
            search_patch = NestedTensor(x_tensor, search_masks)
            out_dict = self.network.forward(
                template=self.z_tensor, search=x_tensor, ce_template_mask=self.box_mask_z, text_data=text_data, template_patch=self.template_patch, search_patch=search_patch)

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes, best_score = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'], return_score=True)
        max_score = best_score[0][0].item()
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # for debug
        if self.debug == 1:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image[:,:,:3], cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
            cv2.putText(image_BGR, 'max_score:' + str(round(max_score, 3)), (40, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 255), 2)
            cv2.imshow('debug_vis', image_BGR)
            cv2.waitKey(1)

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state}
                    # "all_boxes": all_boxes_save,
                    # "best_score": max_score}
        else:
            return {"target_bbox": self.state}
                    # "best_score": max_score}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_tracker_class():
    return UnTrack
