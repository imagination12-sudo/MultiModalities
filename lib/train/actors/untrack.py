import pdb

from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
from lib.train.admin import multigpu
from lib.utils_NLT.misc import NestedTensor


class UntrackActor(BaseActor):
    """ Actor for training UnTrack models """
    """
    1. 数据预处理与组织
    2.模型前向传播(含有CE机制)
    3.多任务损失计算（GIoU + L1 + Focal）
    4. 训练状态统计(用于日志/可视化)
    """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        """

        :param net: UnTrack model (ViT + Head)
        :param objective: 损失函数字典 （giou, l1, focal 等）
        :param loss_weight: 各项损失权重
        :param settings:
        :param cfg: 完整训练配置(用于CE、 heatmap等)
        """
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    def fix_bns(self):
        net = self.net.module if multigpu.is_multi_gpu(self.net) else self.net
        net.box_head.apply(self.fix_bn)

    def fix_bn(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()

    # 主入口: 每次训练迭代的实际执行函数
    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # print("nl_token_ids type:", type(data['nl_token_ids']))
        # print("nl_token_ids shape:", data['nl_token_ids'].shape if torch.is_tensor(data['nl_token_ids']) else "N/A")

        if isinstance(data['nl_token_ids'], list):
            data['nl_token_ids'] = torch.tensor(data['nl_token_ids'], dtype=torch.long)
        if isinstance(data['nl_token_masks'], list):
            data['nl_token_masks'] = torch.tensor(data['nl_token_masks'], dtype=torch.long)
        data['nl_token_ids'] = data['nl_token_ids'].permute(1, 0)
        data['nl_token_masks'] = data['nl_token_masks'].permute(1, 0)
        # forward pass
        out_dict = self.forward_pass(data)
        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        """

        :param data包含:
                 - template_images: (N_t, batch, C, H, W)
                 - search_images:   (N_s, batch, C, H, W)
                 - template_anno:   (N_t, batch, 4)  # [x, y, w, h]
                 - search_anno:     (N_s, batch, 4)
                 - epoch:           int
        :return:
        """
        # currently only support 1 template and 1 search region
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        # Text Input
        text_data = NestedTensor(data['nl_token_ids'], data['nl_token_masks'])
        # grounding_path = NestedTensor(data['grounding_images'], data['grounding_att'])

        # 数据重组， 用于适配模型输入
        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 6, 128, 128)
            template_list.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 6, 320, 320)

        box_mask_z = None
        ce_keep_rate = None

        # # 添加template_patch 和 search_patch
        template_patch = NestedTensor(data['template_images'], data['template_att'])
        search_patch = NestedTensor(data['search_images'], data['search_att'])
        # Class Elimination（CE）机制
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            # 根据模板目标框，生成二值掩码，标记哪些template tokens属于前景
            box_mask_z = generate_mask_cond(self.cfg, template_list[0].shape[0], template_list[0].device,
                                            data['template_anno'][0])

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            # 随训练进行，逐步降低保留率
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])
            # ce_keep_rate = 0.7

        if len(template_list) == 1:
            template_list = template_list[0]

        out_dict = self.net(template=template_list,
                            search=search_img,
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False,
                            text_data=text_data,
                            template_patch=template_patch,
                            search_patch=search_patch)

        """
        返回字典结构：
            {
            'pred_boxes': (B, N, 4),   # 预测框（cxcywh 归一化）
            'score_map': (B, H, W),    # 响应图（可选）
            'aux_dict': {...}          # 其他中间输出
            }
        """

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        #infonce = pred_dict['infonce']

        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)  # (B,1,H,W)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "IoU": mean_iou.item(),
                      }
            return loss, status
        else:
            return loss