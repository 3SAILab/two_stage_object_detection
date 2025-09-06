import torch
import numpy as np
from typing import List
def bbox_iou(bbox_a, bbox_b):
    """
        计算IOU
        ---
        IoU: [n_bbox_a, n_bbox_b]
    """
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        print(bbox_a, bbox_b)
        raise IndexError
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)

def loc2bbox(src_bbox, loc):
    """
        应用loc到bbox并返回迭代后的bbox
    """
    if src_bbox.size()[0] == 0:
        return torch.zeros((0, 4), dtype=loc.dtype)

    src_width   = torch.unsqueeze(src_bbox[:, 2] - src_bbox[:, 0], -1) # 预选框
    src_height  = torch.unsqueeze(src_bbox[:, 3] - src_bbox[:, 1], -1)
    src_ctr_x   = torch.unsqueeze(src_bbox[:, 0], -1) + 0.5 * src_width
    src_ctr_y   = torch.unsqueeze(src_bbox[:, 1], -1) + 0.5 * src_height
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

    eps = np.finfo(height.dtype).eps
    width = np.maximum(width, eps)
    height = np.maximum(height, eps)

    dx = (base_ctr_x - ctr_x) / width
    dy = (base_ctr_y - ctr_y) / height
    dw = np.log(base_width / width)
    dh = np.log(base_height / height)

    loc = np.vstack((dx, dy, dw, dh)).transpose()

    # (x_min, y_min, x_max, y_max) 格式转换成 (ax, ay, aw, ah) 格式, 即偏移量的gt
    return loc

def xywh2xyxy(anchor: List[int]) -> List[int]:
    """
        (x, y, w, h) -> (x_min, y_min, x_max, y_max)
    """
    anchor[2] = anchor[0] + anchor[2]
    anchor[3] = anchor[1] + anchor[3]
    return anchor