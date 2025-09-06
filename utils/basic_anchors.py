import numpy as np

def generate_basic_anchor(base_size=8, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
    # shape: [3 * 3, 4]
    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4), dtype=np.float32)
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])
            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = - w / 2.
            anchor_base[index, 1] = - h / 2.
            anchor_base[index, 2] = w / 2.
            anchor_base[index, 3] = h / 2.
    return anchor_base
    # anchor_base 格式为 (x_min, y_min, x_max, y_max)

def enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    shift_x             = np.arange(0, width * feat_stride, feat_stride)
    shift_y             = np.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y    = np.meshgrid(shift_x, shift_y) 
    # 将shift_x转换成行向量, shift_y转换成列向量, 然后各自广播到相同shape
    # shift_x = [1, 37] -> [37, 37] 其中每一行元素都相同
    # shift_y = [37, 1] -> [37, 37] 其中每一列元素都相同
    shift               = np.stack( # stack()将多个数组沿着新轴堆叠起来
        ( # ravel()将数组flatten成一维数组
            shift_x.ravel(), # shift_x = [37, 37] -> [1369]
            shift_y.ravel(), # shift_y = [37, 37] -> [1369]
            shift_x.ravel(), # shift_x = [37, 37] -> [1369]
            shift_y.ravel(), # shift_y = [37, 37] -> [1369]
        ), 
        axis=1 # axis=1表示按列堆叠
    ) # shift = [K, 4], K = height * width = 1369, 列格式为 (x_min, y_min, x_max, y_max)
    # shift 格式为 (x1, y1, x2, y2)
    # shift 就是从左往右，从上往下数的坐标
    A       = anchor_base.shape[0] # A = 9
    K       = shift.shape[0] # K = 1369
    anchor  = anchor_base.reshape((1, A, 4)) + shift.reshape((K, 1, 4)) 
    # 平移: anchor = [1, 9, 4] + [1369, 1, 4] = [1369, 9, 4]
    anchor  = anchor.reshape((K * A, 4)).astype(np.float32)
    # anchor = [1369, 9, 4] -> [12321, 4]
    return anchor
    # anchor 平移完的所有先验框
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    nine_anchors = generate_basic_anchor()
    print(nine_anchors)

    height, width, feat_stride  = 38,38,38
    anchors_all                 = enumerate_shifted_anchor(nine_anchors, feat_stride, height, width)
    print(np.shape(anchors_all))
    
    fig     = plt.figure()
    ax      = fig.add_subplot(111)
    plt.ylim(-300,1500)
    plt.xlim(-300,1500)
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    plt.scatter(shift_x,shift_y)
    box_widths  = anchors_all[:,2]-anchors_all[:,0]
    box_heights = anchors_all[:,3]-anchors_all[:,1]
    
    for i in [108, 109, 110, 111, 112, 113, 114, 115, 116]:
        rect = plt.Rectangle([anchors_all[i, 0],anchors_all[i, 1]],box_widths[i],box_heights[i],color="r",fill=False)
        ax.add_patch(rect)
    plt.show()
