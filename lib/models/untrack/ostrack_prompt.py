"""
Basic UnTrack model.
"""
import math
import os
from typing import List
from timm.models.layers import to_2tuple    # 将标量转为(x, x)元组, 主要用于统一处理图像尺寸,如224 到 (224, 224)
import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones # 用于复制多个模块，（如多个box_head，用于辅助损失）
from lib.models.layers.head import build_box_head   # 构建回归头
from lib.models.language_model import build_bert
from lib.models.visual_model.vl_transformer import build_vl_transformer # 多源关系建模模块
from lib.models.visual_model.swin_transformer import build_swin_transformer_backbone  # 构建视觉编码器
# 支持 P
from lib.models.untrack.vit_prompt import vit_base_patch16_224_prompt # Prompt tokens
from lib.models.untrack.vit_ce_prompt import vit_base_patch16_224_ce_prompt # Prompt tokens + class elimination
from lib.utils.box_ops import box_xyxy_to_cxcywh


class UnTrack(nn.Module):
    """ This is the base class for UnTrack """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: 修改版的ViT, 不是标准Transformer
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                text_data=None,
                template_patch = None,
                search_patch = None
                ):
        """

        :param template: 模板图像，目标初始状态
        :param search:  搜索区域图像
        :param ce_template_mask: CE 机制中模板 token 的 mask
        :param ce_keep_rate: CE 保留率
        :param return_last_attn: 是否返回最后一层注意力权重
        :return:
        """

        # 融合后的token序列, aux_dict: 辅助信息
        x, aux_dict = self.backbone(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn,
                                    text_data = text_data,
                                    template_patch = template_patch,
                                    search_patch = search_patch)

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        # 提取最后一层信息
        # 检测头预测
        out = self.forward_head(feat_last, None)

        # 合并输出
        out.update(aux_dict)
        out['backbone_feat'] = x     # 保存原始特征（用于可视化或后续处理）
        return out

    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        从 ViT 输出的 token 序列中，提取搜索区域对应的特征，并预测边界框
        """
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        # 重塑为特征图
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_untrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, 'pretrained')  # use pretrained OSTrack as initialization
    # print(pretrained_path) # lib/models/vipt/pretrained_models
    # print(cfg.MODEL.PRETRAIN_FILE) #./pretrained/OSTrack_ep300.pth.tar
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''
        print('pretained:', pretrained)

    language_backbone = build_bert(cfg)
    if "swin" in cfg.MODEL.VISUAL.BACKBONE:
        visual_backbone = build_swin_transformer_backbone(cfg.MODEL.VISUAL.BACKBONE,
                                                          output_layers=(0, 1, 2), pretrained_path = cfg.MODEL.VISUAL.PRETRAINED_PATH)
        for parameter in visual_backbone.parameters():
            parameter.requires_grad_(True)
    else:
        raise NotImplementedError("VISUAL BACKBONE method not implemented")

    vl_joint_trans = build_vl_transformer(cfg)

    # 构建骨干网络
    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_prompt':
        backbone = vit_base_patch16_224_prompt(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                               search_size=to_2tuple(cfg.DATA.SEARCH.SIZE),
                                               template_size=to_2tuple(cfg.DATA.TEMPLATE.SIZE),
                                               new_patch_size=cfg.MODEL.BACKBONE.STRIDE,
                                               prompt_type=cfg.TRAIN.PROMPT.TYPE,
                                               language_backbone=language_backbone,
                                               vision_backbone=visual_backbone,
                                               vl_joint_trans = vl_joint_trans,
                                               max_query_len = cfg.MODEL.LANGUAGE.BERT.MAX_QUERY_LEN
                                               )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce_prompt':
        backbone = vit_base_patch16_224_ce_prompt(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           search_size=to_2tuple(cfg.DATA.SEARCH.SIZE),
                                           template_size=to_2tuple(cfg.DATA.TEMPLATE.SIZE),
                                           new_patch_size=cfg.MODEL.BACKBONE.STRIDE,
                                           prompt_type=cfg.TRAIN.PROMPT.TYPE,
                                           language_backbone=language_backbone,
                                           hidden_dim=cfg.MODEL.LANGUAGE.BERT.HIDDEN_DIM,
                                           visual_backbone=visual_backbone,
                                           vl_joint_trans = vl_joint_trans,
                                           max_query_len=cfg.MODEL.LANGUAGE.BERT.MAX_QUERY_LEN
                                           )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError
    """For prompt no need, because we have OSTrack as initialization"""
    # backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)
    # 构建检测头
    box_head = build_box_head(cfg, hidden_dim)

    model = UnTrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )
    print(cfg.MODEL.BACKBONE.TYPE)
    if 'OSTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu", weights_only=False)
        pretrained_dict = {key.replace(".attn.", ".attn.lora_attn."): value for key, value in checkpoint['net'].items()}
        missing_keys, unexpected_keys = model.load_state_dict(pretrained_dict, strict=False)

        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)
        #print(f"missing_keys: {missing_keys}")
        print(f"unexpected_keys: {unexpected_keys}")

    return model
