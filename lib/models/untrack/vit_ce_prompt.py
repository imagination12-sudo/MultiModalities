"""
候选消除(Candidate Elimination) + 多模态提示(Prompt) 的 vision Transformer(ViT)模型:
    1.多模态输入支持(RGB + Depth + Event等)
    2.LoRA(Low-Rank Adaptation) 微调机制
    3.候选消除(CE)模块：在transformer的某些层动态剪枝搜索区域token
    4.Prompt模块: 通过额外的prompt token 或特征融合提升模型对多模态信息的建模能力
    5.Token 选择与交换机制:用于融合不同模态或者不同分支的信息
"""

import math
import logging
import pdb
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple

from lib.utils_NLT.misc import NestedTensor
from lib.models.layers.patch_embed import PatchEmbed
from .utils import combine_tokens, recover_tokens, token2feature, feature2token
from .vit import VisionTransformer
from ..layers.attn_blocks import CEBlock, candidate_elimination_prompt
import functools
from torch import nn, Tensor

_logger = logging.getLogger(__name__)

class Prompt_block_VIPT(nn.Module, ):
    def __init__(self, inplanes=None, hide_channel=None, smooth=False):
        super(Prompt_block_VIPT, self).__init__()
        self.conv0_0 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv0_1 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv1x1 = nn.Conv2d(in_channels=hide_channel, out_channels=inplanes, kernel_size=1, stride=1, padding=0)
        self.fovea = Fovea(smooth=smooth)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """ Forward pass with input x. """
        B, C, W, H = x.shape
        x0 = x[:, 0:int(C/2), :, :].contiguous()
        x0 = self.conv0_0(x0)
        x1 = x[:, int(C/2):, :, :].contiguous()
        x1 = self.conv0_1(x1)
        x0 = self.fovea(x0) + x1

        return self.conv1x1(x0)


class PredictorConv(nn.Module):
    def __init__(self, embed_dim=384, num_modals=2):
        """
        作用:为每个模态生成空间注意力权重
        输入: x是一个list, 每个元素是[B, C, H,W]
        输出：每个模态对应的[B, 1, H, W]权重图
        :param embed_dim:
        :param num_modals:
        """
        super().__init__()
        self.num_modals = num_modals
        self.score_nets = nn.ModuleList([nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, groups=(embed_dim)),
            nn.Conv2d(embed_dim, 1, 1),
            nn.Sigmoid()
        ) for _ in range(num_modals)])

    def forward(self, x):
        B, C, H, W = x[0].shape

        x_ = [torch.zeros((B, 1, H, W)) for _ in range(self.num_modals)]

        for i in range(self.num_modals):
            x_[i] = self.score_nets[i](x[i])
        return x_


class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]


class ConvLayerNorm(nn.Module):
    """Channel first layer norm
    """

    def __init__(self, normalized_shape, eps=1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class LayerNormParallel(nn.Module):
    def __init__(self, num_features, num_modals=4):
        super(LayerNormParallel, self).__init__()
        # self.num_modals = num_modals
        for i in range(num_modals):
            setattr(self, 'ln_' + str(i), ConvLayerNorm(num_features, eps=1e-6))

    def forward(self, x_parallel):
        return [getattr(self, 'ln_' + str(i))(x) for i, x in enumerate(x_parallel)]


class PatchEmbedParallel(nn.Module):
    def __init__(self, c1=3, c2=32, patch_size=7, stride=4, padding=0, num_modals=4):
        """
        并行对多个模态图像做 patch embedding
        :param c1:
        :param c2:
        :param patch_size:
        :param stride:
        :param padding:
        :param num_modals:
        """
        super().__init__()
        self.proj = ModuleParallel(nn.Conv2d(c1, c2, 3, 1, 1))  # padding=(ps[0]//2, ps[1]//2)
        self.norm = LayerNormParallel(c2, num_modals)

    def forward(self, x: list) -> list:
        x = self.proj(x)
        _, _, H, W = x[0].shape
        x = self.norm(x)
        return x, H, W


class Fovea(nn.Module):

    def __init__(self, smooth=False):
        """
        模拟"中央凹"注意力，对高响应区域增强
        :param smooth:
        """
        super().__init__()

        # 对每个位置做softmax
        self.softmax = nn.Softmax(dim=-1)

        self.smooth = smooth
        if smooth:
            self.smooth = nn.Parameter(torch.zeros(1) + 10.0)

    def forward(self, x):
        '''
            x: [batch_size, features, k]
        '''
        b, c, h, w = x.shape
        x = x.contiguous().view(b, c, h * w)

        if self.smooth:
            mask = self.softmax(x * self.smooth)
        else:
            mask = self.softmax(x)
        output = mask * x
        output = output.contiguous().view(b, c, h, w)

        return output


class _LoRALayer(nn.Module):
    def __init__(self, w: nn.Module, w_a: nn.Module, w_b: nn.Module):
        """
        在ViT的attention模块上插入LoRA适配器,是的模型可以在冻结主干的同时，通过少量参数微调适配新任务
        :param w:
        :param w_a:
        :param w_b:
        """
        super().__init__()
        self.lora_attn = w
        self.prompt_w_a = w_a
        self.prompt_w_b = w_b

    def forward(self, x, mask_x=None, **kwargs):
        x_attn, attn = self.lora_attn(x, mask_x, True)
        x = x_attn + self.prompt_w_b(self.prompt_w_a(x))
        return x, attn


class PredictorLG(nn.Module):
    """ Image to Patch Embedding from DydamicVit
    """

    def __init__(self, embed_dim=384):
        """
        基于 token 的重要性打分(用于token selection)
        :param embed_dim:
        """
        super().__init__()
        self.num_parallel = 2
        self.score_nets = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1)
        )

    def forward(self, x):
        x = self.score_nets(x)
        return x


class TokenExchange(nn.Module):
    def __init__(self):
        """
        作用：在两个模态之间交换token (基于 mask 阈值)
        逻辑:
            1.若模态1的mask >= 阈值 -> 保留自身
            2.否则 -> 用模态0的token替代
        """
        super(TokenExchange, self).__init__()

    def forward(self, x, mask, mask_threshold):
        # x: [B, N, C], mask: [B, N, 1]
        x0, x1 = torch.zeros_like(x[0]), torch.zeros_like(x[1])
        x1[mask[1] >= mask_threshold] = x[1][mask[1] >= mask_threshold]
        x1[mask[1] < mask_threshold] = x[0][mask[1] < mask_threshold]
        return x1


class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]


class LayerNormParallel(nn.Module):
    def __init__(self, num_features):
        super(LayerNormParallel, self).__init__()
        self.num_parallel = 2
        for i in range(self.num_parallel):
            setattr(self, 'ln_' + str(i), nn.LayerNorm(num_features, eps=1e-6))

    def forward(self, x_parallel):
        return [getattr(self, 'ln_' + str(i))(x) for i, x in enumerate(x_parallel)]

"""
SelectDillDeepFuse加入vl_token条件，增加文本语义
"""
class SelectDillDeepFuse(nn.Module):
    def  __init__(self, embed_dim, use_text_cond=True):
        """
        核心创新模块：基于 Gumbel-Softmax 的动态 token 选择与融合
        :param embed_dim:
        """
        super(SelectDillDeepFuse, self).__init__()
        self.use_text_cond = use_text_cond
        if self.use_text_cond:
            self.text_gate = nn.Linear(embed_dim, embed_dim)
        self.gumbel_x = nn.Linear(embed_dim, 1)
        self.gumbel_mod = nn.Linear(embed_dim, 1)
        rank = 8
        self.proj_x = nn.Linear(embed_dim, rank)
        self.proj_mod = nn.Linear(embed_dim, rank)
        self.proj_res_x = nn.Linear(embed_dim, rank)
        self.proj_res_mod = nn.Linear(embed_dim, rank)

        self.proj_fuse_lvl_x = nn.Linear(2*rank, rank)
        self.proj_fuse_lvl_mod = nn.Linear(2*rank, rank)
        self.proj_back = nn.Linear(rank, embed_dim, True)

    def forward(self, x, vl_token=None):
        x, mod = x.chunk(2,dim=1)
        int = x
        x = feature2token(x)
        mod = feature2token(mod)
        B, N, emb_dim = x.shape[0], x.shape[1], x.shape[2]
        number = N//4
        tau = 1

        token_scores = self.gumbel_x(x) #[64 256 1]
        token_scores = token_scores.reshape(B, -1) #[64 256]
        token_mask, rej_mask = gumbel_softmax(F.log_softmax(token_scores, dim=-1), k=number, tau=tau, hard=True) #[64 256]  sum[token==1] = 64*number, from each batch select the n most tokens
        #token_mask[:, 0] = 1.
        token_mask_x = token_mask.expand(emb_dim, -1, -1).permute(1, 2, 0)
        rej_mask_x = rej_mask.expand(emb_dim, -1, -1).permute(1, 2, 0)
        low_x = self.proj_x(x * token_mask_x + mod * rej_mask_x)

        low_res_x = self.proj_res_x((x + mod) * (1 - rej_mask_x - token_mask_x))

        fuse_x = self.proj_fuse_lvl_x(torch.cat((low_x, low_res_x), dim=2) )

        token_scores = self.gumbel_mod(mod) #[64 256 1]
        token_scores = token_scores.reshape(B, -1) #[64 256]
        token_mask, rej_mask = gumbel_softmax(F.log_softmax(token_scores, dim=-1), k=number, tau=tau, hard=True) #[64 256]  sum[token==1] = 64*number, from each batch select the n most tokens
        #token_mask[:, 0] = 1.
        token_mask_mod = token_mask.expand(emb_dim, -1, -1).permute(1, 2, 0)
        rej_mask_mod = rej_mask.expand(emb_dim, -1, -1).permute(1, 2, 0)

        low_mod = self.proj_mod(mod * token_mask_mod + x * rej_mask_mod)
        low_res_mod = self.proj_res_mod((x + mod) * (1 - rej_mask_mod - token_mask_mod))

        fuse_mod = self.proj_fuse_lvl_mod(torch.cat((low_mod, low_res_mod), dim=2) )

        fuse = self.proj_back(fuse_x + fuse_mod)
        if self.use_text_cond and vl_token is not None:
            vl_token = vl_token.mean(dim=1)  # [48, 768]
            # vl_token: [B, C]
            gate = torch.sigmoid(self.text_gate(vl_token))  # [B, C]
            fuse = fuse * gate.unsqueeze(1)     # [B, N, C]
        return token2feature(fuse)

class EdgeLora(nn.Module):
    def  __init__(self, embed_dim):
        """
        基于语义索引(sem_idx)对不同区域(depth/texture/edge)做LoRA适配
        用途:增强边缘/纹理/深度区域的表示
        :param embed_dim:
        """
        super(EdgeLora, self).__init__()
        rank = 4
        self.proj_d = nn.Linear(embed_dim, rank)
        self.proj_t = nn.Linear(embed_dim, rank)
        self.proj_e = nn.Linear(embed_dim, rank)

        self.proj_grad = nn.Linear(embed_dim, rank)

        self.proj_fuse = nn.Linear(3*rank, rank)

        self.fuse = nn.Linear(rank, rank)
        self.proj_back = nn.Linear(rank, embed_dim, True)

        # 新增：vl_token 调制路径
        self.vl_proj = nn.Linear(embed_dim, rank)  # 将 vl_token 映射到 rank 空间

    def forward(self, x, grad, sem_idx=None, vl_token=None):
        """

        :param x: 主干特征
        :param grad: 边缘/梯度增强特征
        :param sem_idx: 语义索引
        :param vl_token: 多源关系建模输出
        :return:
        """
        B, N, C = x.shape[0], x.shape[1], x.shape[2]

        d = torch.zeros_like(x)
        t = torch.zeros_like(x)
        e = torch.zeros_like(x)

        d[sem_idx == 1, ...] = x[sem_idx == 1, ...]
        t[sem_idx == 2, ...] = x[sem_idx == 2, ...]
        e[sem_idx == 3, ...] = x[sem_idx == 3, ...]

        rank_d = self.proj_d(d)
        rank_t = self.proj_t(t)
        rank_e = self.proj_e(e)
        rank_grad = self.proj_grad(grad)
        fused = self.proj_fuse(torch.cat((rank_d , rank_t , rank_e), dim=2))
        guide = self.fuse(rank_grad)
        rank =  fused + guide
        # print("rank shape", rank.shape)
        # === 新增：vl_token 调制 ===
        if vl_token is not None:
            # vl_token: [L_vl, B, C] -> 取平均或特定 token 作为全局语义
            # 假设 vl_token 包含 [text; template; search]，我们只关心与当前样本相关的语义
            # 简单做法：对所有 token 平均池化
            vl_global = vl_token.mean(dim=1)  # [B, C]
            # print("vl_global shape", vl_global.shape)

            # 映射到 rank 空间
            vl_rank = self.vl_proj(vl_global)  # [B, r]
            # print("vl_rank shape", vl_rank.shape)
            # 广播到每个 patch
            vl_rank = vl_rank.unsqueeze(1).expand(-1, N, -1)  # [B, N, r]
            # print("vl_rank shape", vl_rank.shape)
            # 融合：加法 or 门控（这里用加法，简单有效）
            rank = rank + vl_rank

        # 重建
        recon = self.proj_back(rank)  # [B, N, C]
        return recon + grad


def gradient(depth_tmp, step =7):
    """
    作用：计算图像的多方向梯度图（模拟边缘检测）
    步骤：
        1. 对输入做滑动窗口 max pooling（step=7）
        2.在四个方向（上下左右）做仿射变换
        3.计算差值 → 得到梯度响应
        4.输出 4 通道梯度 + 最大梯度图
    :param depth_tmp:
    :param step:
    :return:
    """
    B, C, H, W = depth_tmp.size()
    pad = (step - 1) // 2
    depth_tmp = F.pad(depth_tmp, [pad, pad, pad, pad], mode='constant', value=0)
    patches = depth_tmp.unfold(dimension=2, size=step, step=1)
    patches = patches.unfold(dimension=3, size=step, step=1)
    max_depth, _ = patches.reshape(B, C, H, W, -1).max(dim=-1)

    step = float(step)
    shift_list = [[step / H, 0.0 / W], [-step / H, 0.0 / W], [0.0 / H, step / W], [0.0 / H, -step / W]]
    output_list = []
    for shift in shift_list:
        transform_matrix = torch.tensor([[1, 0, shift[0]], [0, 1, shift[1]]]).unsqueeze(0).repeat(B, 1, 1).to(depth_tmp.device)
        grid = F.affine_grid(transform_matrix, max_depth.shape).float()
        output = F.grid_sample(max_depth, grid, mode='nearest', align_corners=True)
        output = max_depth - output
        output_mask = ((output == max_depth) == False)
        output = output * output_mask
        output_list.append(output)
    grad = torch.cat(output_list, dim=1)
    max_grad = torch.abs(grad).max(dim=1)[0].unsqueeze(1)

    return grad, max_grad

def scatter(logits, index, k):
    bs = logits.shape[0]
   #print('bs = {}'.format(bs))

    x_index = torch.arange(bs).reshape(-1, 1).expand(bs,k)
    x_index = x_index.reshape(-1).tolist()
    y_index = index.reshape(-1).tolist()

    output = torch.zeros_like(logits).cuda()
    output[x_index, y_index] = 1.0
   #print(output.sum(dim=1))

    return output

def gumbel_softmax(logits, k, tau=1, hard=False, eps=1e-10, dim=-1):
    """
    作用：实现可微分的 top-k 采样
    与标准 Gumbel 不同：
        没有加 Gumbel 噪声（直接用 logits）
        hard=True 时：
        ret_top: top-k 位置为 1
        ret_bot: bottom-k 位置为 1（用于 reject mask）
        配合 scatter 实现 one-hot 硬采样
⚠️ 注意：这里其实不是标准 Gumbel-Softmax，而是可微分 top-k + bottom-k mask 生成器。
    :param logits:
    :param k:
    :param tau:
    :param hard:
    :param eps:
    :param dim:
    :return:
    """
    # type: # (torch.Tensor, float, bool, float, int) -> torch.Tensor
    #gumbels = (
    #    -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    #)  # ~Gumbel(0,1)
    #gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    gumbels = logits
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.topk(k, dim=dim)[1]
        y_hard = scatter(logits, index, k)
        ret_top = y_hard - y_soft.detach() + y_soft

        index = (-y_soft).topk(k, dim=dim)[1]
        y_hard = scatter(logits, index, k)
        ret_bot = y_hard - y_soft.detach() + y_soft


    else:
        # Reparametrization trick.
        ret_top = y_soft
        ret_bot = y_soft

    if torch.isnan(ret_top).sum():
        raise OverflowError(f'gumbel softmax output: {ret_top}')
    return ret_top, ret_bot


class VisionTransformerCE(VisionTransformer):
    """ Vision Transformer with candidate elimination (CE) module

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', ce_loc=None, ce_keep_ratio=None, search_size=None, template_size=None,
                 new_patch_size=None, prompt_type=None, language_backbone = None, hidden_dim=None, visual_backbone=None,
                 vl_joint_trans = None, max_query_len=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
            new_patch_size: backbone stride
        """
        super().__init__()
        self.divisor = 16
        self.USE_VIS_SEP = True
        self.num_visu_template_token = int((template_size[0] // self.divisor) ** 2)
        self.num_visu_search_token = int((search_size[0] // self.divisor) ** 2)
        self.num_text_token = max_query_len
        self.num_total = self.num_visu_template_token + self.num_visu_search_token + self.num_text_token
        if self.USE_VIS_SEP:
            self.num_total += 1
            self.sep_embed = nn.Embedding(1, hidden_dim)
        self.vl_pos_embed = nn.Embedding(self.num_total, hidden_dim)
        self.local_rank = torch.cuda.current_device()
        """ 新增: vl_joint_trans 视觉语义信息建模模块"""
        self.vl_joint_trans = vl_joint_trans
        """ 新增: swin_transformer 提取多尺度特征 (visual_backbone)"""
        self.visual_backbone = visual_backbone

        """ 新增自然语言处理 """
        self.language_backbone = language_backbone

        self.hidden_dim = hidden_dim

        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.shallow_prompt = EdgeLora(embed_dim)

        """
        下述两行代码作用:
            将视觉和语言特征投影（映射）到同一个共享的隐空间(hidden_space)中，
            便于后续进行跨模态融合
        """
        self.visu_proj = nn.Linear(visual_backbone.num_channels_output[-1], hidden_dim)
        self.text_proj = nn.Linear(self.language_backbone.num_channels, hidden_dim)

        # 多模态Patch Embedding
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim) # RGB

        self.patch_embed_prompt_dte = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim) # Depth/Texture

        self.patch_embed_prompt_grad = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans*9, embed_dim=embed_dim) # Gradient (9通道:RGB + 4 grad + 4 grad)

        self.patch_embed_prompt_edge = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=1, embed_dim=embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_drop = nn.Dropout(p=drop_rate)

        '''
        prompt parameters
        '''
        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_search = new_P_H * new_P_W
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_template = new_P_H * new_P_W
        """add here, no need use backbone.finetune_track """
        self.pos_embed_z = nn.Parameter(torch.zeros(1, self.num_patches_template, embed_dim))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, self.num_patches_search, embed_dim))

        self.prompt_type = prompt_type
        # various architecture
        if self.prompt_type in ['shaw', 'deep']:
            prompt_blocks = []
            block_nums = depth if self.prompt_type == 'deep' else 1
            for i in range(block_nums):
                prompt_blocks.append(SelectDillDeepFuse(embed_dim))

            self.prompt_blocks = nn.Sequential(*prompt_blocks)
            prompt_norms = []
            for i in range(block_nums):
                prompt_norms.append(norm_layer(embed_dim))
            self.prompt_norms = nn.Sequential(*prompt_norms)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        ce_index = 0
        self.ce_loc = ce_loc
        for i in range(depth):
            ce_keep_ratio_i = 1.0
            if ce_loc is not None and i in ce_loc:
                ce_keep_ratio_i = ce_keep_ratio[ce_index]
                ce_index += 1

            blocks.append(
                CEBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                    keep_ratio_search=ce_keep_ratio_i)
            )

        self.blocks = nn.Sequential(*blocks)

        self.w_As = []  # These are linear layers
        self.w_Bs = []

        rank = 4
        # Here, we do the surgery
        for t_layer_idx, blk in enumerate(self.blocks):
            # If we only want few lora layer instead of all]
            w_a_linear_qkv = nn.Linear(embed_dim, rank, bias=False)
            w_b_linear_qkv = nn.Linear(rank, embed_dim, bias=False)
            self.w_As.append(w_a_linear_qkv)
            self.w_Bs.append(w_b_linear_qkv)
            # blk.prev_attn = blk.attn
            blk.attn = _LoRALayer(blk.attn, w_a_linear_qkv, w_b_linear_qkv)
        self.reset_parameters()

        self.prompt_model = self.blocks

        self.norm = norm_layer(embed_dim)

        self.init_weights(weight_init)


    def reset_parameters(self):
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def tokenselect(self, x_ext, module):
        x_scores = module(x_ext)
        for i in range(len(x_ext)):
            x_ext[i] = x_scores[i] * x_ext[i] + x_ext[i]
        x_f = functools.reduce(torch.max, x_ext)
        return x_f

    def forward_text(self, text_data: NestedTensor):
        # language bert
        text_fea = self.language_backbone(text_data)
        text_src, text_mask = text_fea.decompose()  # seq_len * b * HIDDEN_DIM , seq_len * b
        text_src = self.text_proj(text_src)
        return text_src, text_mask

    def forward_vision_backbone(self, img):
        # img, mask = images.decompose()
        img_sz = img.shape[-1]
        img_src = self.visual_backbone(img)
        img_src = self.visu_proj(img_src[-1].flatten(2).permute(0, 2, 1).contiguous())
        return img_src

    def process_mask_to_tokens(self, mask):
        """
        Args:
            mask: Tensor of shape [1, B, H, W] (bool or int)
            img_sz: int, original image size used for reference (e.g., 128 or 256)
            divisor: int, patch size (e.g., 16)

        Returns:
            token_mask: Tensor of shape [B, token_len], dtype=torch.bool
        """
        # Step 1: Remove the leading singleton dim -> [B, H, W]
        mask = mask.squeeze(0)  # [B, H, W]

        B, H, W = mask.shape

        # Step 2: Interpolate to token grid size: [B, H//divisor, W//divisor]
        target_size = (H // self.divisor, W // self.divisor)
        mask_float = mask.float().unsqueeze(1)  # [B, 1, H, W]
        mask_resized = F.interpolate(mask_float, size=target_size, mode='nearest')  # [B, 1, h, w]
        mask_resized = mask_resized.squeeze(1)  # [B, h, w]

        # Step 3: Flatten spatial dims -> [B, h*w]
        token_mask = mask_resized.to(torch.bool).flatten(1)  # [B, token_len]

        return token_mask


    def forward_joint(self, text_src, text_mask, template_src, template_mask, search_src, search_mask, temporal):
        srcs = [text_src, template_src]
        masks = [text_mask, template_mask]
        bs = text_src.shape[0]

        self.vl_proj = nn.Linear(768, 256).to(text_src.device)


        if self.USE_VIS_SEP:
            sep_mask = torch.zeros((bs, 1), requires_grad=True).cuda().to(torch.bool)
            sep_src = self.sep_embed.weight.unsqueeze(1).repeat(bs, 1, 1)
            srcs.append(sep_src)
            masks.append(sep_mask)

        srcs.append(search_src)
        masks.append(search_mask)
        vl_src = torch.cat(srcs, dim=1).permute(1, 0, 2).contiguous()
        vl_src = self.vl_proj(vl_src)
        vl_mask = torch.cat(masks, dim=1).contiguous()
        vl_pos = self.vl_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        # vl_src: (L, B, C)
        # vl_mask: (B, L)
        # vl_pos: (L, B, C)
        output = self.vl_joint_trans(vl_src, vl_mask, vl_pos)

        return output

    def get_vl_fused_search_tokens(self, text_src, text_mask, template_src, template_mask, search_src, search_mask):
        srcs = [text_src, template_src]
        masks = [text_mask, template_mask]
        bs = text_src.shape[0]

        if self.USE_VIS_SEP:
            sep_mask = torch.zeros((bs, 1), dtype=torch.bool, device=text_src.device)
            sep_src = self.sep_embed.weight.unsqueeze(0).expand(bs, -1, -1)  # (B, 1, C)
            srcs.append(sep_src)
            masks.append(sep_mask)

        srcs.append(search_src)
        masks.append(search_mask)

        vl_src = torch.cat(srcs, dim=1).permute(1, 0, 2).contiguous()  # (L, B, C)
        vl_mask = torch.cat(masks, dim=1).contiguous()  # (B, L)
        vl_pos = self.vl_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)  # (L, B, C)

        output = self.vl_joint_trans(vl_src, vl_mask, vl_pos)  # (L, B, C)

        # 提取 search tokens 部分
        num_search = search_src.shape[1]  # e.g., H*W of search region
        search_tokens = output[-num_search:, :, :]  # (num_search, B, C)
        return search_tokens.permute(1, 0, 2).contiguous()  # (B, num_search, C)


    def forward_features(self, z, x, mask_z=None, mask_x=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False, text_data=None, template_patch=None, search_patch=None):
        """

        :param z: template
        :param x: search region
        :param mask_z:
        :param mask_x:
        :param ce_template_mask:
        :param ce_keep_rate:
        :param return_last_attn:
        :return:
        """
        # 获取文本数据
        """
        text_src含义:
            文本经过语言模型(如BERT)编码后的 “上下文感知的token表示” （embeddings）
        形状:
            （B, LANGUAGE_LENGTH, hidden_dim）
            B: batch_size
            LANGUAGE_LENGTH: 文本序列长度
            hidden_dim: 语言模型的隐藏层维度
        内容:
            每个位置是对应 token 的语义向量，已融合上下文信息
        用途:
            作为多模态的文本特征输入到后续的跨模态模块
            
        text_mask含义:
            与text_src对应的“有效 token 掩码”(key padding mask)
        形状:
            (B, LANGUAGE_LENGTH)
        数据类型:
            torch.BoolTensor(或可转换为 bool)
        值含义：
            True：表示该位置是 padding（无效），不应参与 attention 计算
            False：表示该位置是 真实 token（有效）
        """
        text_src, text_mask = self.forward_text(text_data)  # 初始即为 768 维
        # print("text_mask", text_mask.shape)
        bs = text_src.shape[0]      # 当前批次中样本的数量

        text_len = text_src.shape[1]  # e.g., 30

        # x = 32, 9, 256, 256; z = 32, 9, 128, 128
        loss_mod = []

        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        # print("x shape before indexing:", x.shape)
        sem_idx = x[:, 6, 0, 0]
        # rgb_img·
        _, template_mask = template_patch.decompose()
        _, search_mask = search_patch.decompose()

        x_rgb = x[:, :3, :, :]
        print("x_rgb", x_rgb.shape)
        search_src = self.forward_vision_backbone(x_rgb)
        search_masks = self.process_mask_to_tokens(search_mask)


        z_rgb = z[:, :3, :, :]
        template_src = self.forward_vision_backbone(z_rgb)
        template_masks = self.process_mask_to_tokens(template_mask)

        # 确保所有 mask 的 batch 维度为 B
        text_mask = text_mask.expand(B, -1)  # 虽然已经是 [48, 41]，但保险起见
        template_masks = template_masks.expand(B, -1)  # [1, 512] → [48, 512]
        search_masks = search_masks.expand(B, -1)

        # 多源关系建模模块
        vl_search_token = self.get_vl_fused_search_tokens(text_src, text_mask, template_src, template_masks, search_src,
                                                          search_masks)  # temporal作用：启用时序建模模块
        print("vl_search_token", vl_search_token.shape)

        x_dte = x[:, 3:6, ...]
        # print("x_dte shape", x_dte.shape)
        z_dte = z[:, 3:6, ...]
        # print("z_dte shape", z_dte.shape)

        x, z = x_rgb, z_rgb
        z = self.patch_embed(z)  # 32, 64, 768
        x = self.patch_embed(x)  # 32, 256, 768

        z_rgb_4edge, max_rgb_z_edge = gradient(z_dte)
        x_rgb_4edge, max_rgb_x_edge = gradient(x_dte)

        z_4edge, max_z_edge = gradient(z_dte)
        x_4edge, max_x_edge = gradient(x_dte)


        z_dte_mod = self.patch_embed_prompt_grad(torch.cat((z_rgb, z_rgb_4edge, z_4edge), dim=1))
        x_dte_mod = self.patch_embed_prompt_grad(torch.cat((x_rgb, x_rgb_4edge, x_4edge), dim=1))

        z_dte = self.patch_embed_prompt_dte(z_dte)  # 32, 64, 768
        x_dte = self.patch_embed_prompt_dte(x_dte)

        z_dte = self.shallow_prompt(z_dte, z_dte_mod, sem_idx, vl_search_token)
        x_dte = self.shallow_prompt(x_dte, x_dte_mod, sem_idx, vl_search_token)

        ze = self.patch_embed_prompt_edge(max_z_edge)
        xe = self.patch_embed_prompt_edge(max_x_edge)


        if self.prompt_type in ['shaw', 'deep']:
            z_feat = token2feature(self.prompt_norms[0](z))
            x_feat = token2feature(self.prompt_norms[0](x))

            z_dte_feat = token2feature(self.prompt_norms[0](z_dte))
            x_dte_feat = token2feature(self.prompt_norms[0](x_dte))

            z_feat = torch.cat([z_feat, z_dte_feat], dim=1)
            x_feat = torch.cat([x_feat, x_dte_feat], dim=1)
            z_feat = self.prompt_blocks[0](z_feat)
            x_feat = self.prompt_blocks[0](x_feat)

            z_dte = feature2token(z_feat)
            x_dte = feature2token(x_feat)
            z_prompted, x_prompted = z_dte, x_dte

            z = z + z_dte
            x = x + x_dte
        else:
            z = z + z_dte
            x = x + x_dte

        # attention mask handling
        # B, H, W
        if mask_z is not None and mask_x is not None:
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)

            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)

            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        z += self.pos_embed_z
        x += self.pos_embed_x

        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed

        x = combine_tokens(z, x, mode=self.cat_mode)
        # 将文本 token 拼接进入, 将与文本结合的token与视觉分开
        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_drop(x)

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]

        # 记录文本token的起始位置
        total_vis_tokens = lens_z + lens_x

        text_start = total_vis_tokens

        global_index_t = torch.linspace(0, lens_z - 1, lens_z, dtype=torch.int64).to(x.device)
        global_index_t = global_index_t.repeat(B, 1)

        global_index_s = torch.linspace(0, lens_x - 1, lens_x, dtype=torch.int64).to(x.device)
        global_index_s = global_index_s.repeat(B, 1)

        removed_indexes_s = []
        removed_flag = False
        for i, blk in enumerate(self.prompt_model):
            '''
            add parameters prompt from 1th layer
            '''
            if i >= 1:
                if self.prompt_type in ['deep']:
                    x_ori = x
                    # recover x to go through prompt blocks
                    lens_z_new = global_index_t.shape[1]
                    lens_x_new = global_index_s.shape[1]
                    z = x[:, :lens_z_new]
                    x = x[:, lens_z_new:]
                    if removed_indexes_s and removed_indexes_s[0] is not None:
                        removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)
                        pruned_lens_x = lens_x - lens_x_new
                        pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
                        x = torch.cat([x, pad_x], dim=1)
                        index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
                        C = x.shape[-1]
                        x = torch.zeros_like(x).scatter_(dim=1,
                                                         index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64),
                                                         src=x)
                    x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)
                    x = torch.cat([z, x], dim=1)

                    # prompt
                    x = self.prompt_norms[i - 1](x)
                    z_tokens = x[:, :lens_z, :]
                    x_tokens = x[:, lens_z:text_start, :]

                    z_feat = token2feature(z_tokens)
                    x_feat = token2feature(x_tokens)
                    #import pdb; pdb.set_trace()

                    z_prompted = self.prompt_norms[i](z_prompted) + ze
                    x_prompted = self.prompt_norms[i](x_prompted) + xe

                    z_prompt_feat = token2feature(z_prompted)
                    x_prompt_feat = token2feature(x_prompted)

                    z_feat = torch.cat([z_feat, z_prompt_feat], dim=1)
                    x_feat = torch.cat([x_feat, x_prompt_feat], dim=1)

                    z_feat = self.prompt_blocks[i](z_feat)
                    x_feat = self.prompt_blocks[i](x_feat)

                    z = feature2token(z_feat)
                    x = feature2token(x_feat)
                    z_prompted, x_prompted = z, x

                    x = combine_tokens(z, x, mode=self.cat_mode)
                    # re-conduct CE
                    x = candidate_elimination_prompt(x, global_index_t.shape[1], global_index_s)
                    x = x_ori + x

            x, global_index_t, global_index_s, removed_index_s, attn = \
                blk(x, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate)



            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_s.append(removed_index_s)

        x = self.norm(x)
        lens_x_new = global_index_s.shape[1]
        lens_z_new = global_index_t.shape[1]

        z = x[:, :lens_z_new]
        x = x[:, lens_z_new:]

        if removed_indexes_s and removed_indexes_s[0] is not None:
            removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)

            pruned_lens_x = lens_x - lens_x_new
            pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
            x = torch.cat([x, pad_x], dim=1)
            index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
            # recover original token order
            C = x.shape[-1]
            x = torch.zeros_like(x).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64),
                                             src=x)

        x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)

        # re-concatenate with the template, which may be further used by other modules
        x = torch.cat([z, x], dim=1)

        aux_dict = {
            "attn": attn,
            "removed_indexes_s": removed_indexes_s,  # used for visualization
        }

        return x, aux_dict

    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None,
                return_last_attn=False,
                language_backbone=None,
                visual_backbone=None,
                text_data = None, template_patch = None, search_patch = None):

        x, aux_dict = self.forward_features(z, x, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate, text_data=text_data,
                                            template_patch=template_patch, search_patch=search_patch)

        return x, aux_dict


def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerCE(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
            print('Load pretrained OSTrack from: ' + pretrained)
            print(f"missing_keys: {missing_keys}")
            print(f"unexpected_keys: {unexpected_keys}")

    return model


def vit_base_patch16_224_ce_prompt(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


def vit_large_patch16_224_ce_prompt(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model
