import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.loc_bbox_iou import bbox2loc, bbox_iou, loc2bbox
from nets.classify import HarNetRoIHead
from nets.rpn import RegionProposalNetwork
from models.hardnet import HarDNetFeatureExtraction, HarNetClassifier

class AnchorTargetCreator():
    """
        创建 RPN 网络所需的先验框标签 (正样本, 负样本, 忽略), 确定每个先验框的回归目标
    """
    def __init__(self, n_sample=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3, pos_ratio=0.5):
        self.n_sample       = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio      = pos_ratio

    def __call__(self, bbox, anchor):
        # 创建标签
        # argmax_ious：先验框i -> 最大IOU的真实框索引
        # label：(先验框数量,) 1 为正样本，0 为负样本，-1 为忽略
        argmax_ious, label = self._create_label(anchor, bbox)
        if (label > 0).any():
            # 获取先验框到真实框的偏移量 gt
            loc = bbox2loc(anchor, bbox[argmax_ious])
            # loc: 先验框到真实框的偏移量 gt
            # label: [先验框数量, ] 1 为正样本，0 为负样本，-1 为忽略
            return loc, label
        else:
            return np.zeros_like(anchor), label

    def _calc_ious(self, anchor, bbox):
        #   anchor和bbox的iou
        #   获得的ious的shape为[num_anchors, num_gt]
        ious = bbox_iou(anchor, bbox)
        if len(bbox)==0:
            return np.zeros(len(anchor), np.int32), np.zeros(len(anchor)), np.zeros(len(bbox))
        #   获得每一个先验框最对应的真实框  [num_anchors, ]
        #   先验框i -> 最大IOU的真实框索引
        argmax_ious = ious.argmax(axis=1)
        #   找出每一个先验框最对应的真实框的iou  [num_anchors, ]
        #   先验框i -> 最大IOU的真实框IOU
        max_ious = np.max(ious, axis=1)
        #   获得每一个真实框最对应的先验框  [num_gt, ]
        #   真实框i -> 最大IOU的先验框索引
        gt_argmax_ious = ious.argmax(axis=0)
        #   保证每一个真实框都存在对应的先验框
        #   每个真实框 -> 每个真实框对应的最大IOU先验框索引 -> 每个先验框对应的最大IOU真实框索引换成第i个真实框
        #   即每个真实框都至少有一个先验框对应，多个先验框对应同一个真实框
        for i in range(len(gt_argmax_ious)):
            argmax_ious[gt_argmax_ious[i]] = i
        return argmax_ious, max_ious, gt_argmax_ious
        # argmax_ious：先验框i -> 最大IOU的真实框索引
        # max_ious：先验框i -> 最大IOU的真实框IOU
        # gt_argmax_ious：真实框i -> 最大IOU的先验框索引
        
    def _create_label(self, anchor, bbox):
        #   1是正样本，0是负样本，-1忽略
        #   初始化的时候全部设置为-1
        label = np.empty((len(anchor),), dtype=np.int32)
        label.fill(-1)
        #   argmax_ious为每个先验框对应的最大的真实框的序号         [num_anchors, ]
        #   max_ious为每个先验框对应的最大的真实框的iou             [num_anchors, ]
        #   gt_argmax_ious为每一个真实框对应的最大的先验框的序号    [num_gt, ]
        argmax_ious, max_ious, gt_argmax_ious = self._calc_ious(anchor, bbox)
        #   如果小于门限值则设置为负样本
        #   如果大于门限值则设置为正样本
        #   每个真实框至少对应一个先验框
        label[max_ious < self.neg_iou_thresh] = 0
        label[max_ious >= self.pos_iou_thresh] = 1
        if len(gt_argmax_ious) > 0:
            label[gt_argmax_ious] = 1

        #   判断正样本数量是否大于128，如果大于则限制在128
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1

        #   平衡正负样本，保持总数量为256
        n_neg = self.n_sample - np.sum(label == 1)
        # n_neg = (1 - self.pos_ratio) * self.n_sample
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return argmax_ious, label
        # argmax_ious：先验框i -> 最大IOU的真实框索引
        # label：(先验框数量,) 1 为正样本，0 为负样本，-1 为忽略

class ProposalTargetCreator(object):
    """
        创建建议框的标签(正样本, 负样本, 忽略)
    outputs:
        --------
        sample_roi: [n_sample, 4] 包含正样本和负样本的所有建议框集合
        gt_roi_loc: [n_sample, 4] 偏移量 gt
        gt_roi_label: [n_sample, ] 所有真实框标签, 背景为 0 类别索引 > 0
    """
    def __init__(self, n_sample=128, pos_ratio=0.5, pos_iou_thresh=0.5, neg_iou_thresh_high=0.5, neg_iou_thresh_low=0):
        self.n_sample = n_sample # 总样本数
        self.pos_ratio = pos_ratio # 正样本比例
        self.pos_roi_per_image = np.round(self.n_sample * self.pos_ratio) # 每张图片的正样本数
        self.pos_iou_thresh = pos_iou_thresh # 正样本IOU阈值
        self.neg_iou_thresh_high = neg_iou_thresh_high # 负样本IOU上限
        self.neg_iou_thresh_low = neg_iou_thresh_low # 负样本IOU下限

    def __call__(self, roi, bbox, label, loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        """
        Inputs:
            ------------
            roi: [num_rpn_rois, 4], 格式为 (x_min, y_min, x_max, y_max)
            bbox: [num_gt, 4], 格式为 (x_min, y_min, x_max, y_max)
            label: [num_gt, ]
        """
        # 通过将GT框加入建议框来保证建议框的高质量
        roi = np.concatenate((roi.detach().cpu().numpy(), bbox), axis=0)
        #   计算建议框和真实框的重合程度
        iou = bbox_iou(roi, bbox)
        
        if len(bbox)==0:
            gt_assignment = np.zeros(len(roi), np.int32)
            max_iou = np.zeros(len(roi))
            gt_roi_label = np.zeros(len(roi))
        else:
            #   建议框 -> 对应最大IOU真实框的索引
            gt_assignment = iou.argmax(axis=1)
            #   建议框 -> 对应的最大IOU
            max_iou = iou.max(axis=1)
            #   整体向右 +1, 使背景为 0, 类别索引 > 1
            gt_roi_label = label[gt_assignment] + 1
        #   满足建议框和真实框重合程度大于neg_iou_thresh_high的作为正样本
        #   将正样本的数量限制在self.pos_roi_per_image以内
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(self.pos_roi_per_image, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(pos_index, size=pos_roi_per_this_image, replace=False)
        #   满足建议框和真实框重合程度小于neg_iou_thresh_high大于neg_iou_thresh_low作为负样本
        #   将正样本的数量和负样本的数量的总和固定成self.n_sample
        neg_index = np.where((max_iou < self.neg_iou_thresh_high) & (max_iou >= self.neg_iou_thresh_low))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(neg_index, size=neg_roi_per_this_image, replace=False)
        #   sample_roi      [n_sample, ]
        #   gt_roi_loc      [n_sample, 4]
        #   gt_roi_label    [n_sample, ]
        keep_index = np.append(pos_index, neg_index)

        sample_roi = roi[keep_index]
        if len(bbox)==0:
            return sample_roi, np.zeros_like(sample_roi), gt_roi_label[keep_index]

        # 获取偏移量 gt ,并做归一化
        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_loc = (gt_roi_loc / np.array(loc_normalize_std, np.float32))

        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0

        return sample_roi, gt_roi_loc, gt_roi_label

class FasterRCNNTrainer(nn.Module):
    """
    ouput:
        losses: List[rpn_loc_loss_all, rpn_cls_loss_all, roi_loc_loss_all, roi_cls_loss_all, total_loss]
        anchors_pred: [batch_size, n_sample, 4] (x_min, y_min, x_max, y_max)
        classes_pred: [batch_size, n_sample]
        anchors_gt: [batch_size, n_sample, 4] (x_min, y_min, x_max, y_max)
        classes_gt: [batch_size, n_sample]
    """
    def __init__(
            self, 
            mode,
            optimizer,
            num_classes,
            feat_stride = 16,
            anchor_scales = [8, 16, 32],
            ratios = [0.5, 1, 2],
        ):
        super(FasterRCNNTrainer, self).__init__()
        self.optimizer = optimizer
        self.feat_stride = feat_stride
        self.rpn_sigma = 1
        self.roi_sigma = 1
        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()
        self.feat_extra = HarDNetFeatureExtraction()
        self.classifier = HarNetClassifier()
        self.rpn = RegionProposalNetwork(
            512,
            ratios          = ratios,
            anchor_scales   = anchor_scales,
            feat_stride     = self.feat_stride,
            mode            = mode
        )
        self.head = HarNetRoIHead(
            n_class         = num_classes + 1,
            roi_size        = 7,
            spatial_scale   = 1,
            classifier      = self.classifier
        )
        self.loc_normalize_std = [0.1, 0.1, 0.2, 0.2]

    def _fast_rcnn_loc_loss(self, pred_loc, gt_loc, gt_label, sigma):
        # 提取pred和gt中正样本的结果
        pred_loc = pred_loc[gt_label > 0]
        gt_loc = gt_loc[gt_label > 0]
        num_pos = (gt_label > 0).sum().float()

        sigma_squared = sigma ** 2
        regression_diff = (gt_loc - pred_loc).abs().float()

        # 误差小于 (1. / sigma_squared) 时使用L2损失, 否则使用L2损失
        regression_loss = torch.where(
                regression_diff < (1. / sigma_squared),
                0.5 * sigma_squared * regression_diff ** 2,
                regression_diff - 0.5 / sigma_squared
            ).sum()

        # 得到平均回归 loss
        regression_loss /= torch.max(num_pos, torch.ones_like(num_pos))
        return regression_loss
        
    def forward(self, imgs, bboxes, labels, scale = 1):
        anchors_pred = []
        classes_pred = []
        classes_score_pred = []
        anchors_gt = []
        n           = imgs.shape[0]
        img_size    = imgs.shape[2:]
        #   获取公用特征层
        base_feature = self.feat_extra(imgs)
        #   利用rpn网络获得: 偏移量预测, 分类概率预测, 所有建议框, 所有建议框索引, 所有先验框
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn.forward(
            x = base_feature, 
            img_size = img_size, 
            scale = scale
        )
        rpn_loc_loss_all, rpn_cls_loss_all, roi_loc_loss_all, roi_cls_loss_all = 0, 0, 0, 0
        sample_rois, sample_indexes, gt_roi_locs, gt_roi_labels = [], [], [], []
        for i in range(n):
            bbox        = bboxes[i] # 真实框的坐标
            label       = labels[i] # 真实框的类别
            rpn_loc     = rpn_locs[i] # 偏移量预测
            rpn_score   = rpn_scores[i] # 分类概率预测
            roi         = rois[i] # 建议框坐标
            #   gt_rpn_loc: [num_anchors, 4] 先验框到真实框的偏移量 gt1
            #   gt_rpn_label: [num_anchors, ] 1 为正样本，0 为负样本，-1 为忽略
            gt_rpn_loc, gt_rpn_label    = self.anchor_target_creator(bbox, anchor[0].cpu().numpy())
            gt_rpn_loc                  = torch.Tensor(gt_rpn_loc).type_as(rpn_locs)
            gt_rpn_label                = torch.Tensor(gt_rpn_label).type_as(rpn_locs).long()
            #   分别计算建议框网络的回归损失和分类损失
            rpn_loc_loss = self._fast_rcnn_loc_loss(rpn_loc, gt_rpn_loc, gt_rpn_label, self.rpn_sigma)
            rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label, ignore_index=-1)

            rpn_loc_loss_all += rpn_loc_loss
            rpn_cls_loss_all += rpn_cls_loss
            #   利用真实框和建议框获得classifier网络应该有的预测结果
            #   sample_roi: [n_sample, 4] 包含正样本和负样本的所有建议框集合
            #   gt_roi_loc: [n_sample, 4] 偏移量 gt
            #   gt_roi_label: [n_sample, 1] 所有真实框标签, 背景为 0 类别索引 > 0
            sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(roi, bbox, label, self.loc_normalize_std)
            sample_rois.append(torch.Tensor(sample_roi).type_as(rpn_locs))
            sample_indexes.append(torch.ones(len(sample_roi)).type_as(rpn_locs) * roi_indices[i][0])
            gt_roi_locs.append(torch.Tensor(gt_roi_loc).type_as(rpn_locs))
            gt_roi_labels.append(torch.Tensor(gt_roi_label).type_as(rpn_locs).long())
            
        # 将 sample_rois 和 sample_indexes 创建一个新维度并叠加元素
        sample_rois     = torch.stack(sample_rois, dim=0)
        sample_indexes  = torch.stack(sample_indexes, dim=0)
        # 获取分类的 pred 和 classify 结果, 包含: 回归预测, 分类类别
        roi_cls_locs, roi_scores = self.head.forward(
            x = base_feature, 
            rois = sample_rois, 
            roi_indices = sample_indexes, 
            img_size = img_size
        )
        for i in range(n):
            # 建议框坐标
            roi = sample_rois[i]
            #   根据建议框的种类，取出对应的回归预测结果
            n_sample = roi_cls_locs.size()[1]
            
            roi_cls_loc     = roi_cls_locs[i] # 回归预测
            roi_score       = roi_scores[i] # 分类类别
            gt_roi_loc      = gt_roi_locs[i] # 偏移量 gt
            gt_roi_label    = gt_roi_labels[i] # 真实类别
            
            # [n_sample, num_classes * 4] -> [n_sample, num_classes, 4]
            roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
            # 取出第 i 个建议框第 gt 类的回归预测
            # [n_sample, num_classes, 4] -> [n_sample, 4]
            roi_loc     = roi_cls_loc[torch.arange(0, n_sample), gt_roi_label]

            # anchors_pred: List[torch.tensor(n_sample, 4)]
            # classes_pred: List[torch.tensor(n_sample, 1)]
            anchors_pred.append(loc2bbox(roi, roi_loc))
            classes_pred.append(torch.argmax(roi_score, dim=1))
            classes_score_pred.append(torch.max(roi_score, dim=1))
            anchors_gt.append(loc2bbox(roi, gt_roi_loc))

            #   分别计算Classifier网络的回归损失和分类损失
            roi_loc_loss = self._fast_rcnn_loc_loss(roi_loc, gt_roi_loc, gt_roi_label.data, self.roi_sigma)
            roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label)

            roi_loc_loss_all += roi_loc_loss
            roi_cls_loss_all += roi_cls_loss

        anchors_pred = torch.stack(anchors_pred)
        classes_pred = torch.stack(classes_pred)
        anchors_gt = torch.stack(anchors_gt)
        classes_gt = torch.stack(gt_roi_labels)
            
        losses = [
            rpn_loc_loss_all / n, 
            rpn_cls_loss_all / n,
            roi_loc_loss_all / n, 
            roi_cls_loss_all / n
        ]
        losses = losses + [sum(losses)]

        return losses, anchors_pred, classes_pred, classes_score_pred, anchors_gt, classes_gt
    
    def eval(self, imgs, bboxes, labels, scale=1):
        eval_loss, anchors_pred, classes_pred, classes_score_pred, anchors_gt, classes_gt = self.forward(imgs, bboxes, labels, scale)
        eval_loss = eval_loss[-1]
        metrics = self.calculate_metrics(
            anchors_pred, 
            classes_pred, 
            classes_score_pred, 
            anchors_gt, 
            classes_gt
        )
    
    def calculate_metrics(
            anchors_pred, 
            classes_pred, 
            classes_score_pred, 
            anchors_gt, 
            classes_gt
        ):
        """
        ---
            anchors_pred: [batch_size, n_sample, 4] (x_min, y_min, x_max, y_max)
            classes_pred: [batch_size, n_sample]
            classes_score_pred: [batch_size, n_sample]
            anchors_gt: [batch_size, n_sample, 4] (x_min, y_min, x_max, y_max)
            classes_gt: [batch_size, n_sample]
        """
        classes_score_pred, classes_score_pred_indices = torch.sort(
            classes_score_pred, 
            dim=1
        )
        
        anchors_pred = anchors_pred.view(-1, 4)
        anchors_gt = anchors_gt.view(-1, 4)
        anchors_pred_area = torch.zeros_like(classes_pred, )
        # IoU: [batch_size * n_sample, batch_size * n_sample]
        iou = bbox_iou(anchors_pred, anchors_gt)

