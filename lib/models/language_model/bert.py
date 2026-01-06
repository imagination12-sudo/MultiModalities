# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
import torch
from torch import nn
from transformers import BertModel, BertConfig, logging
from lib.utils_NLT.misc import NestedTensor

# 可选：关闭 transformers 的警告
logging.set_verbosity_error()

"""
语言编码器（Language Encoder）
关键类：BERT(封装了 BertModel)

功能支持：
    1. 支持冻结参数 (train_bert = False)
    2. 输入为 NestedTensor (含有token IDS 和 attention mask)
    3. 输出为词嵌入或某层transformer的输出 (由enc_num控制)
"""


class BERT(nn.Module):
    def __init__(self, name: str, path: str, train_bert: bool, hidden_dim: int, max_len: int, enc_num: int):
        super().__init__()
        # 自动推断通道数
        if "base" in name or (path and "base" in path):
            self.num_channels = 768
        elif "large" in name or (path and "large" in path):
            self.num_channels = 1024
        else:
            # 尝试从 config 获取
            config = BertConfig.from_pretrained(path if path else name)
            self.num_channels = config.hidden_size

        self.enc_num = enc_num

        # 加载预训练模型
        if path is not None and path != "":
            self.bert = BertModel.from_pretrained(path, add_pooling_layer=False)
        else:
            self.bert = BertModel.from_pretrained(name, add_pooling_layer=False)

        # 冻结参数
        if not train_bert:
            print('Language Model BERT has been frozen!')
            for param in self.bert.parameters():
                param.requires_grad_(False)

    def forward(self, tensor_list: NestedTensor):
        input_ids = tensor_list.tensors  # [B, L]
        attention_mask = tensor_list.mask  # [B, L], 1 for real tokens, 0 for padding

        # 注意：transformers 的 attention_mask 是 1 表示保留，0 表示 mask
        # 所以直接使用即可，无需取反

        if self.enc_num > 0:
            # 获取所有层输出
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            # hidden_states 是 tuple，索引 0 是 embedding，1~N 是各层
            # enc_num=1 表示第一层 transformer 输出（即 hidden_states[1]）
            all_hidden_states = outputs.hidden_states  # tuple of [B, L, H]
            xs = all_hidden_states[self.enc_num]  # 注意：embedding 是第 0 层
        else:
            # 仅词嵌入（不经过任何 transformer 层）
            xs = self.bert.embeddings.word_embeddings(input_ids)

        # mask: 1 表示有效 token，0 表示 padding
        # NestedTensor 的 mask 是 1 表示 padding（原始代码中如此）
        # 所以我们保持一致性：out.mask = ~attention_mask
        mask = ~attention_mask.bool()  # 转为 bool 并取反，与原始逻辑一致
        out = NestedTensor(xs, mask)

        return out


def build_bert(cfg):
    train_bert = cfg.MODEL.LANGUAGE.BERT.LR > 0
    bert = BERT(
        name=cfg.MODEL.LANGUAGE.TYPE,
        path=cfg.MODEL.LANGUAGE.PATH,
        train_bert=train_bert,
        hidden_dim=cfg.MODEL.LANGUAGE.BERT.HIDDEN_DIM,
        max_len=cfg.MODEL.LANGUAGE.BERT.MAX_QUERY_LEN,
        enc_num=cfg.MODEL.LANGUAGE.BERT.ENC_NUM
    )
    return bert