import numpy as np
import torch
import os
import json
from torch import nn
from torch.nn import functional as F
from torchvision.ops import nms
from utils.basic_anchors import enumerate_shifted_anchor, generate_basic_anchor
from utils.loc_bbox_iou import loc2bbox

config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs/config.json")
with open(config_path, "r") as f:
    config = json.load(f)

device = config['device']

class ProposalCreator():
    def __init__(
        self, 
        mode, 
        nms_iou = 0.7,
        n_train_pre_nms = 12000,
        n_train_post_nms = 600,
        n_test_pre_nms = 3000,
        n_test_post_nms = 300,
        min_size = 16
    ):
        self.mode = mode
        self.nms_iou = nms_iou
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc, score, anchor, img_size, scale=1.):
        if self.mode == "train":
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        # 使用预测偏移量迭代一次anchors, 得到新的anchors, 格式为 (x_min, y_min, x_max, y_max)
        roi = loc2bbox(anchor, loc)

        roi[:, [0, 2]] = torch.clamp(roi[:, [0, 2]], min = 0, max = img_size[1])
        roi[:, [1, 3]] = torch.clamp(roi[:, [1, 3]], min = 0, max = img_size[2])
        
        min_size = self.min_size * scale
        keep = torch.where(((roi[:, 2] - roi[:, 0]) >= min_size) & ((roi[:, 3] - roi[:, 1]) >= min_size))[0]

        roi = roi[keep, :]
        score = score[keep]

        #   根据得分进行排序后直接取前n_pre_nms个建议框
        order = torch.argsort(score, descending=True)
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order, :]
        score = score[order]

        keep = nms(roi, score, self.nms_iou)
        # 若小于 600 个, 则补全
        if len(keep) < n_post_nms:
            index_extra = torch.arange((n_post_nms - len(keep))).type_as(keep)
            keep = torch.cat((keep, index_extra))
        keep = keep[:n_post_nms]
        roi = roi[keep]
        return roi

class RegionProposalNetwork(nn.Module):
    def __init__(
        self, 
        in_channels = 512, 
        ratios = [0.5, 1, 2],
        anchor_scales = [8, 16, 32], 
        feat_stride = 16,
        mode = "training",
    ):
        super(RegionProposalNetwork, self).__init__()
        #   生成基础先验框，shape为[9, 4]
        self.anchor_base    = generate_basic_anchor(anchor_scales = anchor_scales, ratios = ratios)
        n_anchor            = self.anchor_base.shape[0] # 9

        self.score  = nn.Conv2d(in_channels, n_anchor * 2, 1, 1, 0)
        # [b, 512, 37, 37] -> [b, 18, 37, 37]
        self.loc    = nn.Conv2d(in_channels, n_anchor * 4, 1, 1, 0)
        # [b, 512, 37, 37] -> [b, 36, 37, 37]

        #   特征点间距步长
        self.feat_stride    = feat_stride
        #   用于对建议框解码并进行非极大抑制
        self.proposal_layer = ProposalCreator(mode)
        #   对FPN的网络部分进行权值初始化

    def forward(self, x, img_size, scale=1.):
        """
        output:
            rpn_locs: [b, 12321, 4] 偏移量预测，格式为 (dx, dy, dw, dh)
            rpn_scores: [b, 12321, 2] - 分类概率预测，格式为 (background, object)
            rois: [b, 600, 4] - 所有建议框位置，格式为 (x_min, y_min, x_max, y_max)
            roi_indices: [b, 600] - 所有建议框的索引
            anchor: [1, 12996, 4] - 所有先验框，格式为 (x_min, y_min, x_max, y_max)
        """
        n, _, h, w = x.shape
        rpn_locs = self.loc(x)
        # [b, 512, 37, 37] -> [b, 36, 37, 37]
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        # [b, 36, 37, 37] -> [b, 37, 37, 36] -> [b, 37 * 37 * 9, 4] -> [b, 12321, 4]
        rpn_scores = self.score(x).to(device)
        # [b, 512, 37, 37] -> [b, 18, 37, 37]
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(n, -1, 2)
        # [b, 18, 37, 37] -> [b, 37, 37, 18] -> [b, 37 * 37 * 9, 2] -> [b, 12321, 2]
        rpn_softmax_scores = F.softmax(rpn_scores, dim=-1)
        rpn_fg_scores = rpn_softmax_scores[:, :, 1].contiguous()
        # [b, 37 * 37 * 9, 2] -> [b, 37 * 37 * 9]
        rpn_fg_scores = rpn_fg_scores.view(n, -1)
        # [b, 37 * 37 * 9] -> [b, 12321] (假设输入图片为600,600,3)
        # 生成移动后的先验框
        anchor = enumerate_shifted_anchor(
            torch.tensor(self.anchor_base).to(device), # [9, 4]
            self.feat_stride, # 16
            h, # 37
            w  # 37
        ).to(device) # anchor = [37 * 37 * 9, 4]
        rois = list()
        roi_indices = list()
        for i in range(n):
            roi = self.proposal_layer(
                rpn_locs[i], # [12321, 4] 这一批次中的第i个图片的所有anchors回归预测结果
                rpn_fg_scores[i], # [12321] 这一批次中的第i个图片的所有anchors预测为有物体的概率
                anchor, # [12321, 4] 所有的先验anchors
                img_size, # [b, 3, 600, 600] 输入图片的大小
                scale=scale
            ) # roi = [n, 4] 这一批次中的第i个图片的所有建议框, training时n为600，测试时n为300
            rois.append(roi.unsqueeze(0)) # [1, n, 4] 将这一批次中的第i个图片的所有建议框放入列表

        rois = torch.cat(rois, dim=0).type_as(x) # [b, n, 4] 所有建议框
        # [1, n, 4] -> [b, n, 4]
        anchor = anchor.unsqueeze(0).float().to(x.device) # [1, 12321, 4] 所有的先验anchors, 并转换anchor的type和device与x一致

        return rpn_locs, rpn_scores, rois, anchor