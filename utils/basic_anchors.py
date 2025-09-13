import torch
import os
import json

config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs/config.json")
with open(config_path, "r") as f:
    config = json.load(f)

device = config['device']

def generate_basic_anchor(base_size=8, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
    # shape: [3 * 3, 4]
    anchor_base = torch.zeros((len(ratios) * len(anchor_scales), 4), dtype=torch.float32).to(device)
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * torch.sqrt(torch.tensor(ratios[i]).type_as(anchor_base))
            w = base_size * anchor_scales[j] * torch.sqrt(torch.tensor(1. / ratios[i]).type_as(anchor_base))
            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = - w / 2.
            anchor_base[index, 1] = - h / 2.
            anchor_base[index, 2] = w / 2.
            anchor_base[index, 3] = h / 2.
    return anchor_base
    # anchor_base 格式为 (x_min, y_min, x_max, y_max)


def enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    # 创建 shift_x 和 shift_y，步长为 feat_stride
    shift_x = torch.arange(0, width * feat_stride, feat_stride).type_as(anchor_base)
    shift_y = torch.arange(0, height * feat_stride, feat_stride).type_as(anchor_base)
    shift_x, shift_y = torch.meshgrid(shift_x, shift_y, indexing='xy')

    # 将 shift_x 转换成行向量, shift_y 转换成列向量, 然后各自广播到相同 shape
    # shift_x = [1, 37] -> [37, 37] 其中每一行元素都相同
    # shift_y = [37, 1] -> [37, 37] 其中每一列元素都相同
    shift = torch.stack(
        (  # ravel() 在 Torch 中对应 .view(-1) 或 .flatten()
            shift_x.ravel(),  # shift_x = [37, 37] -> [1369]
            shift_y.ravel(),  # shift_y = [37, 37] -> [1369]
            shift_x.ravel(),  # shift_x = [37, 37] -> [1369]
            shift_y.ravel(),  # shift_y = [37, 37] -> [1369]
        ),
        dim=1  # axis=1 表示按列堆叠 → Torch 中是 dim=1
    )  # shift = [K, 4], K = height * width = 1369, 列格式为 (x_min, y_min, x_max, y_max)
    # shift 格式为 (x1, y1, x2, y2)
    # shift 就是从左往右，从上往下数的坐标

    A = anchor_base.shape[0]  # A = 9
    K = shift.shape[0]        # K = 1369

    # 平移: anchor = [1, 9, 4] + [1369, 1, 4] = [1369, 9, 4]
    anchor = anchor_base.view(-1, A, 4) + shift.view(K, 1, 4)

    # anchor = [1369, 9, 4] -> [12321, 4]
    anchor = anchor.view(K * A, 4).float()

    return anchor
    # anchor 平移完的所有先验框
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    nine_anchors = generate_basic_anchor()
    print(nine_anchors)

    height, width, feat_stride  = 38,38,38
    anchors_all                 = enumerate_shifted_anchor(nine_anchors, feat_stride, height, width)
    print(torch.shape(anchors_all))
    
    fig     = plt.figure()
    ax      = fig.add_subplot(111)
    plt.ylim(-300,1500)
    plt.xlim(-300,1500)
    shift_x = torch.arange(0, width * feat_stride, feat_stride)
    shift_y = torch.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y = torch.meshgrid(shift_x, shift_y)
    plt.scatter(shift_x,shift_y)
    box_widths  = anchors_all[:,2]-anchors_all[:,0]
    box_heights = anchors_all[:,3]-anchors_all[:,1]
    
    for i in [108, 109, 110, 111, 112, 113, 114, 115, 116]:
        rect = plt.Rectangle([anchors_all[i, 0],anchors_all[i, 1]],box_widths[i],box_heights[i],color="r",fill=False)
        ax.add_patch(rect)
    plt.show()
