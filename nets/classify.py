import warnings
import torch
from torch import nn
from torchvision.ops import RoIPool

warnings.filterwarnings("ignore")

class HarNetRoIHead(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        super().__init__()
        self.classifier = classifier
        #   对ROIPooling后的的结果进行回归预测
        self.cls_loc    = nn.Linear(512, n_class * 4)
        #   对ROIPooling后的的结果进行分类
        self.score      = nn.Linear(512, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        # 池化到固定尺寸
        self.roi = RoIPool((roi_size, roi_size), spatial_scale)
        
    def forward(self, x, rois, roi_indices, img_size):
        """
        output
            roi_cls_locs 是建议框的回归预测结果, 格式为 (dx, dy, dw, dh), shape为: [b, n_sample, n_class*4]
            roi_scores 是建议框的分类预测结果, shape为: [b, n_sample, n_class]
        """
        n, _, _, _ = x.shape
        # x = [b, 512, 37, 37]
        # rois = [b, 600, 4]
        # roi_indices = [b, 600]
        if x.is_cuda:
            roi_indices = roi_indices.cuda()
            rois = rois.cuda()
        rois = torch.flatten(rois, 0, 1)
        # rois = [b*600, 4]
        roi_indices = torch.flatten(roi_indices, 0, 1)
        # roi_indices = [b*600]

        # 将原图尺寸的建议框坐标映射到特征图上
        rois_feature_map = torch.zeros_like(rois)
        # rois_feature_map = [b*600, 4]
        rois_feature_map[:, [0,2]] = rois[:, [0,2]] / img_size[1] * x.size()[3]
        rois_feature_map[:, [1,3]] = rois[:, [1,3]] / img_size[0] * x.size()[2]

        indices_and_rois = torch.cat([roi_indices[:, None], rois_feature_map], dim=1)
        # indices_and_rois = [b, 1, 5] + [b*600, 4] = [b*600, 5]
        # indices_and_rois 的第一列是这个批次所有建议框索引，后面是建议框在特征图上的坐标

        #   利用建议框对公用特征层进行截取, 类似于根据建议框坐标, 在特征图指定位置上截取 b * 600 个大小为 7 * 7 的子特征图
        pool = self.roi(x, indices_and_rois)
        # pool = ([b, 512, 37, 37], [b*600, 5]) -> [b*600, 512, 7, 7]
        # 获取分类结果
        fc7 = self.classifier(pool)
        # fc7 = [b * 600, 512, 7, 7] -> [b * 600, 512]
        roi_cls_locs    = self.cls_loc(fc7)
        # roi_cls_locs = [b*600, 512] -> [b*600, n_class*4]
        roi_scores      = self.score(fc7)
        # roi_scores = [b*600, 512] -> [b*600, n_class]
        roi_cls_locs    = roi_cls_locs.view(n, -1, roi_cls_locs.size(1))
        # roi_cls_locs = [b*600, n_class*4] -> [b, 600, n_class*4]
        roi_scores      = roi_scores.view(n, -1, roi_scores.size(1))
        # roi_scores = [b*600, n_class] -> [b, 600, n_class]
        return roi_cls_locs, roi_scores

def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
