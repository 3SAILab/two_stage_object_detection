import torch
import numpy as np
from typing import List

def bbox_iou(bbox_a, bbox_b):
    """
    计算IOU（Torch版本）
    ---
    输入:
        bbox_a: Tensor, shape [n_bbox_a, 4], 格式为 (x1, y1, x2, y2)
        bbox_b: Tensor, shape [n_bbox_b, 4], 格式为 (x1, y1, x2, y2)
    输出:
        iou: Tensor, shape [n_bbox_a, n_bbox_b]
    """
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        print(bbox_a, bbox_b)
        raise IndexError
    # 扩展维度以支持广播：bbox_a -> [n_a, 1, 2], bbox_b -> [n_b, 2]
    tl = torch.maximum(bbox_a[:, None, :2], bbox_b[:, :2])  # top-left
    br = torch.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])  # bottom-right
    # 计算交集宽高，负数说明无交集，应为0
    wh = br - tl
    wh.clamp_(min=0)  # 等价于 (tl < br).all(axis=2) 的效果，但更高效且避免布尔索引
    area_i = torch.prod(wh, dim=2)  # [n_a, n_b]
    area_a = torch.prod(bbox_a[:, 2:] - bbox_a[:, :2], dim=1)  # [n_a]
    area_b = torch.prod(bbox_b[:, 2:] - bbox_b[:, :2], dim=1)  # [n_b]
    iou = area_i / (area_a[:, None] + area_b - area_i + 1e-8)  # 防止除零
    return iou

def loc2bbox(src_bbox, loc):
    """
        应用loc到bbox并返回迭代后的bbox
    """
    if src_bbox.size()[0] == 0:
        return torch.zeros((0, 4), dtype=loc.dtype)

    src_width   = torch.unsqueeze(src_bbox[:, 2] - src_bbox[:, 0], -1).type_as(loc) # 预选框
    src_height  = torch.unsqueeze(src_bbox[:, 3] - src_bbox[:, 1], -1).type_as(loc)
    src_ctr_x   = torch.unsqueeze(src_bbox[:, 0], -1).type_as(loc) + 0.5 * src_width
    src_ctr_y   = torch.unsqueeze(src_bbox[:, 1], -1).type_as(loc) + 0.5 * src_height
    # 将 (x_min, y_min, x_max, y_max) 格式转换成 (ax, ay, aw, ah) 格式

    dx          = loc[:, 0::4]
    dy          = loc[:, 1::4]
    dw          = loc[:, 2::4]
    dh          = loc[:, 3::4]
    # 提取预测回归偏移量 dx, dy, dw, dh

    ctr_x = dx * src_width + src_ctr_x
    ctr_y = dy * src_height + src_ctr_y
    w = torch.exp(dw) * src_width
    h = torch.exp(dh) * src_height
    # 迭代并计算pred_anchor的 (ax, ay, aw, ah) 坐标

    dst_bbox = torch.zeros_like(loc)
    dst_bbox[:, 0::4] = ctr_x - 0.5 * w
    dst_bbox[:, 1::4] = ctr_y - 0.5 * h
    dst_bbox[:, 2::4] = ctr_x + 0.5 * w
    dst_bbox[:, 3::4] = ctr_y + 0.5 * h
    # 将 pred_anchor 的 (ax, ay, aw, ah) 格式转换成 (x_min, y_min, x_max, y_max) 格式

    return dst_bbox

def bbox2loc(src_bbox, dst_bbox):
    """
        获取输入框到目标框的偏移量 gt
    """
    width = src_bbox[:, 2] - src_bbox[:, 0]
    height = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_x = src_bbox[:, 0] + 0.5 * width
    ctr_y = src_bbox[:, 1] + 0.5 * height

    base_width = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_height = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_x = dst_bbox[:, 0] + 0.5 * base_width
    base_ctr_y = dst_bbox[:, 1] + 0.5 * base_height

    eps = torch.finfo(height.dtype).eps
    width = torch.maximum(width, torch.tensor(eps))
    height = torch.maximum(height, torch.tensor(eps))

    dx = (base_ctr_x - ctr_x) / width
    dy = (base_ctr_y - ctr_y) / height
    dw = torch.log(base_width / width)
    dh = torch.log(base_height / height)

    loc = torch.vstack((dx, dy, dw, dh)).T

    # (x_min, y_min, x_max, y_max) 格式转换成 (ax, ay, aw, ah) 格式, 即偏移量的gt
    return loc

def xywh2xyxy(anchor: List[List]) -> List[List]:
    """
        (x, y, w, h) -> (x_min, y_min, x_max, y_max)
    """
    anchor[2] += anchor[0]
    anchor[3] += anchor[1]
    return anchor