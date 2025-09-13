import os
import json
import torch
import torch.nn as nn
from torchvision.ops import nms
from torch.nn import functional as F
from utils.utils import filter_pr, compute_ap_from_pr
from utils.loc_bbox_iou import bbox2loc, bbox_iou, loc2bbox
from nets.classify import HarNetRoIHead
from nets.rpn import RegionProposalNetwork
from models.hardnet import HarDNetFeatureExtraction, HarNetClassifier

config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs/config.json")
with open(config_path, "r") as f:
    config = json.load(f)

device = config['device']

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
            return torch.zeros_like(anchor).type_as(anchor), label

    def _calc_ious(self, anchor, bbox):
        #   anchor和bbox的iou
        #   ious.shape: [n_anchors, n_gt]
        ious = bbox_iou(anchor, bbox)
        if len(bbox)==0:
            return torch.zeros(len(anchor), dtype=torch.int32).to(device), torch.zeros(len(anchor)).type_as(anchor), torch.zeros(len(bbox)).type_as(bbox)
        #   获得每一个先验框最对应的真实框  [num_anchors, ]
        #   先验框i -> 最大IOU的真实框索引
        #   找出每一个先验框最对应的真实框的iou  [num_anchors, ]
        #   先验框i -> 最大IOU的真实框IOU
        max_ious, argmax_ious = torch.max(ious, dim=1)
        #   获得每一个真实框最对应的先验框  [num_gt, ]
        #   真实框i -> 最大IOU的先验框索引
        gt_argmax_ious = ious.argmax(dim=0)
        #   保证每一个真实框都存在对应的先验框
        #   每个真实框 -> 每个真实框对应的最大IOU先验框索引 -> 每个先验框对应的最大IOU真实框索引换成第i个真实框
        #   即每个真实框都至少有一个先验框对应，多个先验框对应同一个真实框
        for i in range(len(gt_argmax_ious)):
            # 即真实框去对应先验框更优先
            argmax_ious[gt_argmax_ious[i]] = i
        return argmax_ious, max_ious, gt_argmax_ious
        # argmax_ious：先验框i -> 最大IOU的真实框索引
        # max_ious：先验框i -> 最大IOU的真实框IOU
        # gt_argmax_ious：真实框i -> 最大IOU的先验框索引
        
    def _create_label(self, anchor, bbox):
        #   1是正样本，0是负样本，-1忽略
        #   初始化的时候全部设置为-1
        label = torch.empty((len(anchor),), dtype=torch.int64).to(device).fill_(-1)
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
        n_pos = int(self.pos_ratio * self.n_sample) # 128
        pos_index = torch.where(label == 1)
        pos_length = pos_index[0].numel()
        if pos_length > n_pos:
            disable_index = tuple(i[n_pos:] for i in pos_index)
            label[disable_index] = -1
            pos_length = n_pos

        #   平衡正负样本，保持总数量为256
        n_neg = self.n_sample - pos_length
        # n_neg = (1 - self.pos_ratio) * self.n_sample
        neg_index = torch.where(label == 0)
        if len(neg_index) > n_neg:
            disable_index = tuple(i[n_neg:] for i in neg_index)
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
        self.pos_roi_per_image = int(self.n_sample * self.pos_ratio) # 每张图片的正样本数
        self.pos_iou_thresh = pos_iou_thresh # 正样本IOU阈值
        self.neg_iou_thresh_high = neg_iou_thresh_high # 负样本IOU上限
        self.neg_iou_thresh_low = neg_iou_thresh_low # 负样本IOU下限

    def __call__(self, roi, bbox, label, loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        """
        Inputs:
            ------------
            roi: [num_rpn_rois, 4], 格式为 (x_min, y_min, x_max, y_max)
            bbox: [num_gt, 4], 格式为 (x_min, y_min, x_max, y_max)
            label: [num_gt]
        """
        # 通过将GT框加入建议框来保证建议框的高质量
        roi = torch.cat((roi, bbox), dim=0)
        #   计算建议框和真实框的重合程度
        iou = bbox_iou(roi, bbox)
        
        if len(bbox)==0:
            gt_assignment = torch.zeros(len(roi), dtype=torch.int32).to(device)
            max_iou = torch.zeros(len(roi)).type_as(roi)
            gt_roi_label = torch.zeros(len(roi)).type_as(label)
        else:
            #   建议框 -> 对应最大IOU真实框的索引
            #   建议框 -> 对应的最大IOU
            max_iou, gt_assignment = torch.max(iou, dim=1)
            #   整体向右 +1, 使背景为 0, 类别索引 > 1
            gt_roi_label = label[gt_assignment] + 1
        #   满足建议框和真实框重合程度大于neg_iou_thresh_high的作为正样本
        #   将正样本的数量限制在self.pos_roi_per_image以内
        pos_index = torch.where(max_iou >= self.pos_iou_thresh)
        pos_length = pos_index[0].numel()
        if pos_length > self.pos_roi_per_image:
            pos_index = tuple(i[:self.pos_roi_per_image] for i in pos_index)
            pos_length = pos_index[0].numel()
        #   满足建议框和真实框重合程度小于neg_iou_thresh_high大于neg_iou_thresh_low作为负样本
        #   将正样本的数量和负样本的数量的总和固定成self.n_sample
        neg_index = torch.where((max_iou < self.neg_iou_thresh_high) & (max_iou >= self.neg_iou_thresh_low))
        neg_roi_per_this_image = self.n_sample - pos_length
        neg_length = neg_index[0].numel()
        if neg_length > neg_roi_per_this_image:
            neg_index = tuple(i[:neg_roi_per_this_image] for i in neg_index)
        #   sample_roi      [n_sample, ]
        #   gt_roi_loc      [n_sample, 4]
        #   gt_roi_label    [n_sample, ]
        keep_index = tuple(torch.cat((a, b), dim=0) for a, b in zip(pos_index, neg_index))

        sample_roi = roi[keep_index]
        if len(bbox)==0:
            return sample_roi, torch.zeros_like(sample_roi).type_as(sample_roi), gt_roi_label[keep_index]

        # 获取偏移量 gt ,并做归一化
        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        # gt_roi_loc = (gt_roi_loc / torch.array(loc_normalize_std, torch.float32))

        # 留下包含正负样本的 gt_roi_label
        gt_roi_label = gt_roi_label[keep_index]
        # 负样本标签设为 0 背景类
        gt_roi_label[neg_index] = 0

        return sample_roi, gt_roi_loc, gt_roi_label

class FasterRCNNTrainer(nn.Module):
    """
    ouput:
        losses: List[rpn_loc_loss_all, rpn_cls_loss_all, roi_loc_loss_all, roi_cls_loss_all, total_loss]
        anchors_pred: [batch_size, n_sample, 4] (x_min, y_min, x_max, y_max)
        classes_pred: [batch_size, n_sample]
        anchors_gt: [batch_size, n_gt, 4] (x_min, y_min, x_max, y_max)
        classes_gt: [batch_size, n_gt]
    """
    def __init__(
            self, 
            mode,
            num_classes,
            feat_stride = 16,
            anchor_scales = [8, 16, 32],
            ratios = [0.5, 1, 2],
        ):
        super(FasterRCNNTrainer, self).__init__()
        self.feat_stride = feat_stride
        self.rpn_sigma = 1
        self.roi_sigma = 1
        self.n_classes = num_classes
        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()
        self.feat_extra = HarDNetFeatureExtraction()
        self.classifier = HarNetClassifier()
        self.rpn = RegionProposalNetwork(
            512,
            ratios = ratios,
            anchor_scales= anchor_scales,
            feat_stride = self.feat_stride,
            mode = mode
        )
        self.head = HarNetRoIHead(
            n_class = num_classes + 1,
            roi_size = 7,
            spatial_scale = 1,
            classifier = self.classifier
        )
        self.loc_normalize_std = [0.1, 0.1, 0.2, 0.2]

    def _fast_rcnn_loc_loss(self, pred_loc, gt_loc, gt_label, sigma):
        # 提取pred和gt中正样本的结果
        pred_loc = pred_loc[gt_label > 0]
        gt_loc = gt_loc[gt_label > 0]

        sigma_squared = sigma ** 2
        # [n_pos, 4]
        regression_diff = torch.abs(gt_loc - pred_loc).float()

        # 误差小于 (1. / sigma_squared) 时使用L2损失, 否则使用L2损失
        regression_loss = torch.where(
                regression_diff < (1. / sigma_squared),
                0.5 * sigma_squared * regression_diff ** 2,
                regression_diff - 0.5 / sigma_squared
            ).sum()

        # 得到平均回归 loss
        regression_loss /= regression_diff.numel()
        return regression_loss
        
    def forward(self, imgs, bboxes, labels, scale = 1):
        """
        数据结构：
            img: List[torch.tensor: [channel, width, height]]
            bboxes: List[torch.tensor: [n_gt, 4]], (x_min, y_min, x_max, y_max)
            labels: List[torch.tensor: [n_gt, 1]]
        """
        anchors_pred = []
        classes_pred = []
        classes_score_pred = []
        n = len(imgs)
        imgs = torch.unsqueeze(imgs[0], dim=0).to(device)
        img_size = imgs.shape[1:]
        #   获取公用特征层
        base_feature = self.feat_extra(imgs)
        #   利用rpn网络获得: 偏移量预测, 分类概率预测, 所有600个建议框, 所有建议框索引, 所有先验框
        rpn_locs, rpn_scores, rois, anchor = self.rpn.forward(
            x = base_feature, 
            img_size = img_size, 
            scale = scale
        )
        rpn_loc_loss_all, rpn_cls_loss_all, roi_loc_loss_all, roi_cls_loss_all = 0, 0, 0, 0
        sample_rois, sample_indexes, gt_roi_locs, gt_roi_labels = [], [], [], []
        for i in range(n):
            bbox = bboxes[i].to(device) # 真实框的坐标
            label = labels[i].to(device) # 真实框的类别
            rpn_loc = rpn_locs[i] # 偏移量预测
            rpn_score = rpn_scores[i] # 分类概率预测
            roi = rois[i] # 建议框坐标
            #   gt_rpn_loc: [num_anchors, 4] 先验框到真实框的偏移量 gt1
            #   gt_rpn_label: [num_anchors, ] 1 为正样本，0 为负样本，-1 为忽略
            gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(bbox, anchor[0])
            #   分别计算建议框网络的回归损失和分类损失
            rpn_loc_loss = self._fast_rcnn_loc_loss(rpn_loc, gt_rpn_loc, gt_rpn_label, self.rpn_sigma)
            rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label, ignore_index=-1)

            rpn_loc_loss_all += rpn_loc_loss
            rpn_cls_loss_all += rpn_cls_loss
            #   利用真实框和建议框获得classifier网络应该有的预测结果
            #   sample_roi: [128, 4] 包含正样本和负样本的所有建议框集合
            #   gt_roi_loc: [128, 4] 偏移量 gt
            #   gt_roi_label: [128] 所有真实框标签, 背景为 0 类别索引 > 0
            sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(roi, bbox, label, self.loc_normalize_std)
            sample_rois.append(sample_roi)
            sample_indexes.append(i)
            gt_roi_locs.append(gt_roi_loc)
            gt_roi_labels.append(gt_roi_label)
            
        # 将 sample_rois 和 sample_indexes 创建一个新维度并叠加元素
        sample_rois = torch.stack(sample_rois, dim=0)
        # sample_indexes: [b]
        sample_indexes = torch.tensor(sample_indexes, dtype=torch.int32).to(device)
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
            n_sample = roi_cls_locs.size(1)
            
            roi_cls_loc = roi_cls_locs[i] # 回归预测
            roi_score = roi_scores[i] # 分类类别
            gt_roi_loc = gt_roi_locs[i] # 偏移量 gt
            gt_roi_label = gt_roi_labels[i] # 真实类别
            
            # [n_sample, num_classes * 4] -> [n_sample, num_classes, 4]
            roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
            # 取出第 i 个建议框第 gt 类的回归预测
            # [n_sample, num_classes, 4] -> [n_sample, 4]
            roi_loc = roi_cls_loc[torch.arange(0, n_sample).type_as(gt_roi_label), gt_roi_label]

            # anchors_pred: List[torch.tensor(n_sample, 4)]
            # classes_pred: List[torch.tensor(n_sample)]
            anchors_pred.append(loc2bbox(roi, roi_loc))
            cls_score_pred, cls_index_pred = torch.max(roi_score, dim=1)
            classes_pred.append(cls_index_pred)
            classes_score_pred.append(cls_score_pred)

            #   分别计算Classifier网络的回归损失和分类损失
            roi_loc_loss = self._fast_rcnn_loc_loss(roi_loc, gt_roi_loc, gt_roi_label, self.roi_sigma)
            roi_cls_loss = F.cross_entropy(roi_score, gt_roi_label)

            roi_loc_loss_all += roi_loc_loss
            roi_cls_loss_all += roi_cls_loss

        anchors_pred = torch.stack(anchors_pred)
        classes_pred = torch.stack(classes_pred)
        classes_score_pred = torch.stack(classes_score_pred)

        losses = [
            rpn_loc_loss_all / n, 
            rpn_cls_loss_all / n,
            roi_loc_loss_all / n, 
            roi_cls_loss_all / n
        ]
        losses = losses + [sum(losses)]

        return losses, anchors_pred, classes_pred, classes_score_pred, torch.unsqueeze(bboxes[0], dim=0), torch.unsqueeze(labels[0] + 1, dim=0)
        # anchors_pred: [b, 128, 4] 所有建议框预测
        # classes_pred: [b, 128] 所有标签最大类别预测 ()
        # classes_score_pred: [b, 128] 所有标签最大类别预测分数
    
    def eval_fn(self, eval_dataloader, scale=1, nms_iou_threshold=0.7, map_iou_threshold=0.7):
        self.eval()
        batch_num, mAP_total, eval_loss_total = 0, 0, 0
        for _, (imgs, bboxes, labels) in enumerate(eval_dataloader):
            eval_loss, anchors_pred, classes_pred, classes_score_pred, anchors_gt, classes_gt = self.forward(imgs, bboxes, labels, scale)
            eval_loss = eval_loss[-1]
            eval_loss_total += eval_loss.item() if hasattr(eval_loss, 'item') else eval_loss
            
            mAP = self.calculate_metrics(
                anchors_pred.cpu(), 
                classes_pred.cpu(), 
                classes_score_pred.cpu(),
                anchors_gt,
                classes_gt,
                nms_iou_threshold=nms_iou_threshold,
                map_iou_threshold = map_iou_threshold
            )
            if mAP:
                mAP_total += mAP
            batch_num += 1
        
        avg_mAP = mAP_total / batch_num if batch_num > 0 else 0
        avg_eval_loss = eval_loss_total / batch_num if batch_num > 0 else 0
        return avg_eval_loss, avg_mAP
    
    def calculate_metrics(
            self,
            anchors_pred: torch.tensor, 
            classes_pred: torch.tensor, 
            classes_score_pred: torch.tensor, 
            anchors_gt: torch.tensor, 
            classes_gt: torch.tensor,
            nms_iou_threshold: float = 0.7,
            map_iou_threshold: float = 0.7
        ):
        """
        ---
            anchors_pred: [batch_size, n_sample, 4] (x_min, y_min, x_max, y_max)
            classes_pred: [batch_size, n_sample]
            classes_score_pred: [batch_size, n_sample]
            anchors_gt: [batch_size, n_gt, 4] (x_min, y_min, x_max, y_max)
            classes_gt: [batch_size, n_gt]
        """
        # 初始化存储结构
        results = {
            'mAP': 0.0,
            'class_metrics': {}
        }
        class_list = list(range(1, self.n_classes + 1))
        # 为每个类别初始化存储
        for class_id in class_list:
            results['class_metrics'][class_id] = {
                'AP': 0.0,             # 平均精度
                'Recall': 0.0,         # 检出率
                'Precision': 0.0,      # 精确率
                'TP': 0,               # 真正例
                'FP': 0,               # 假正例
                'FN': 0,               # 假负例
            }
        pr_data, data_dict = {class_id: None for class_id in class_list}, {}

        for i in range(anchors_pred.size(0)):
            # image -> info
            data_dict[i] = {
                "anchors_pred": anchors_pred[i], # anchors_pred: [b, 128, 4] 所有建议框预测
                "classes_score_pred": classes_score_pred[i], # classes_score_pred: [b, 128] 所有标签最大类别预测置信度
                "classes_pred": classes_pred[i], # classes_pred: [b, 128] 所有标签最大类别预测
                "anchors_gt": anchors_gt[i], # anchors_gt [b, n_gt, 4] 所有真实框坐标
                "classes_gt": classes_gt[i] # classes_gt [b, n_gt] 所有真实框标签, 已 + 1
            }

        # 第一步：遍历所有图像，收集匹配结果
        all_image_ids = list(data_dict.keys())
        
        for img_id in all_image_ids:
            data = data_dict[img_id]
            # 按类别组织真实框, label -> info
            """
                label_2_info [class_id]:{
                    "boxes_pred": torch.tensor[n_cls_pred, 4] 此类别的所有预测框坐标
                    "scores_pred": torch.tensor[n_cls_pred] 此类别的所有预测类别置信度
                    "boxes_gt": torch.tensor[n_cls_gt, 4] 此类别的所有真实框坐标
                }
            """
            label_2_info = {class_id: {} for class_id in class_list}
            for class_id in class_list:
                # 得到 classes_pred 中的 class_id 类索引
                class_id_index = torch.where(data["classes_pred"] == class_id)
                # 用索引提取出 anchors_pred 中的 class_id 类预测框坐标, 并存进 boxes_pred 字段
                label_2_info[class_id]["boxes_pred"] = data["anchors_pred"][class_id_index]
                # 用索引提取出 classes_score_pred 中的 class_id 类预测类别置信度, 并存进 scores_pred 字段
                label_2_info[class_id]["scores_pred"] = data["classes_score_pred"][class_id_index]
                # 得到 classes_gt 中的 class_id 类索引
                class_id_index = torch.where(data["classes_gt"] == class_id)
                # 用索引提取出 anchors_gt 中的 class_id 类真实框坐标, 并存进 boxes_gt 字段
                label_2_info[class_id]["boxes_gt"] = data["anchors_gt"][class_id_index]

            # 对每个类别单独处理
            for class_id in class_list:
                """
                    pr_data:{
                        class_id: torch.tensor(score, bool) tensor.shape: [n_cls_pred, 2]
                        ...
                    }
                """
                class_id_info = label_2_info[class_id]
                # 使用 nms 再筛选一次重叠预测框, 以保证送去计算指标的预测框数量不会太多而导致指标计算不准确
                keep = nms(class_id_info["boxes_pred"], class_id_info["scores_pred"], iou_threshold=nms_iou_threshold)
                # 更新预测框
                class_id_info["boxes_pred"] = class_id_info["boxes_pred"][keep]
                # 更新置信度
                class_id_info["scores_pred"] = class_id_info["scores_pred"][keep]
                
                # 如果没有预测框，所有真实框都是FN
                if len(class_id_info["boxes_pred"]) == 0:
                    results['class_metrics'][class_id]['FN'] += len(class_id_info["boxes_gt"])
                    continue
                
                # 如果没有真实框，所有预测框都是FP
                if len(class_id_info["boxes_gt"]) == 0:
                    results['class_metrics'][class_id]['FP'] += len(class_id_info["boxes_pred"])
                    # 记录FP用于AP计算 shape: [N, 2]
                    pr_data[class_id] = torch.stack([class_id_info["scores_pred"], torch.zeros_like(class_id_info["scores_pred"])], dim=1)
                    continue
                
                # 按置信度降序排序预测框
                scores_pred_sorted, scores_pred_sorted_indices = torch.sort(class_id_info["scores_pred"], dim=0, descending=True)
                # 更新 scores_pred
                class_id_info["scores_pred"] = scores_pred_sorted
                # 计算按照 scores_pred 排序后的 boxes_pred 与 boxes_gt 的 IoU
                # iou: [n_cls_boxes_pred, n_cls_boxes_gt]
                iou = bbox_iou(class_id_info["boxes_pred"][scores_pred_sorted_indices], class_id_info["boxes_gt"])
                # 记录匹配数据 shape: [N, 2]
                pr_data[class_id] = torch.cat(
                                        (
                                            torch.unsqueeze(class_id_info["scores_pred"], dim=1),
                                            torch.any(
                                                torch.max(
                                                    iou, 
                                                    dim=1, 
                                                    keepdim=True
                                                ).values > map_iou_threshold, 
                                                dim=1, 
                                                keepdim=True,
                                            ).to(torch.int64)
                                        ),
                                        dim=1
                                    )

                # 统计当前图像的结果
                cls_pr = pr_data[class_id][:, 1]
                results['class_metrics'][class_id]['TP'] += torch.sum(cls_pr == 1).item()
                results['class_metrics'][class_id]['FP'] += torch.sum(cls_pr == 0).item()
                results['class_metrics'][class_id]['FN'] += class_id_info["boxes_gt"].shape[0] - torch.sum(cls_pr == 1).item()

        # 第二步：计算每个类别的指标
        aps = []
        for class_id in class_list:
            class_data = results['class_metrics'][class_id]
            tp = class_data['TP']
            fp = class_data['FP']
            fn = class_data['FN']

            # 计算 precision 和 recall 指标
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            # 特殊情况判断
            if (tp == 0 and fn == 0) or (tp == 0 and fp == 0):
                # 无真实框 和 无预测框
                class_data['Recall'] = recall
                class_data['Precision'] = precision
                class_data['AP'] = 0
                aps.append(0)
                continue

            elif tp == 0 and fp == 0 and fn == 0:
                # 两种框都没有
                class_data['Recall'] = recall
                class_data['Precision'] = precision
                class_data['AP'] = 0
                aps.append(None)
                continue

            # 初始化 AP 和 数据
            cls_pr_data, ap = pr_data[class_id], 0

            if torch.numel(cls_pr_data) > 0:
                # 按置信度降序排序 cls_pr_data.shape: [n_cls_pred, 2]
                pr, pr_indices = torch.sort(cls_pr_data, dim=0, descending=True)

                # 计算临时 TP 和 FP
                cum_tp, cum_fp, pr_match = 0, 0, []
                """
                    pr_match: List[List[cum_precision, cum_recall]]
                """
                for i in range(1, pr.shape[0] + 1, -1):
                    # 计算临时 precision 和 recall 指标
                    cum_pr = pr[:i][:, 1]
                    cum_tp = torch.sum(cum_pr == 1).item()
                    cum_fp = torch.sum(cum_pr == 0).item()
                    cum_fn = i - cum_tp
                    cum_precision = cum_tp / (cum_tp + cum_fp) if (cum_tp + cum_fp) > 0 else 0.0
                    cum_recall = cum_tp / (cum_tp + cum_fn) if (cum_tp + cum_fn) > 0 else 0.0
                    pr_match.append([cum_precision, cum_recall])
                # 提取出 pr 最大值
                pr_match = filter_pr(torch.tensor(pr_match, dtype=torch.float32), label_2_info[class_id]["boxes_gt"].shape[0])
                ap = compute_ap_from_pr(pr_match, label_2_info[class_id]["boxes_gt"].shape[0])

            # 更新结果
            class_data['Recall'] = recall
            class_data['Precision'] = precision
            class_data['AP'] = ap
            aps.append(ap)
        
        # 计算mAP（所有类别AP的平均）
        filtered = [x for x in aps if x is not None]
        results['mAP'] = sum(filtered) / len(filtered) if filtered else None

        return results['mAP']