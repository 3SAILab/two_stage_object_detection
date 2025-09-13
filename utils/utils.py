import random
import torch

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def update_ema(current_value, ema_alpha, last_ema=None):
    if last_ema is None:
        return current_value
    return ema_alpha * current_value + (1 - ema_alpha) * last_ema

def filter_pr(x, n_gt):
    if x.numel() == 0:
        return torch.zeros(n_gt + 1, 2)
    recalls = torch.arange(n_gt, -1, -1).float() / n_gt
    precisions = [x[x[:, 1] >= r - 1e-6, 0].max().item() if (x[:, 1] >= r - 1e-6).any() else 0.0 for r in recalls]
    return torch.stack([torch.tensor(precisions), recalls], dim=1)

def compute_ap(pr_tensor):
    if len(pr_tensor) == 0:
        return 0.0
    precisions = pr_tensor[:, 0].clone()
    recalls = pr_tensor[:, 1]
    # 单调化：从右往左传播最大值（recall 降低时 precision 不降低）
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    # 积分：每个区间 [r[i+1], r[i]] 使用 precision[i+1]
    ap = 0.0
    for i in range(len(precisions) - 1):
        width = recalls[i] - recalls[i + 1]
        height = precisions[i + 1]
        ap += width * height
    return ap

if __name__ == "__main__":
    x = torch.tensor([
        [0.5, 5 / 7],
        [0.44, 4 / 7],
        [0.375, 3 / 7],
        [0.43, 3 / 7],
        [0.5, 3 / 7],
        [0.4, 2 / 7],
        [0.5, 2 / 7],
        [0.66, 2 / 7],
        [1, 2 / 7],
        [1, 1 / 7],
    ])
    pr = filter_pr(x, 7)
    ap = compute_ap_coco(pr)
    print("PR Curve (原始):\n", pr)
    precisions = pr[:, 0].clone()
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    print("单调化后 precision:", precisions.tolist())
    print(f"COCO AP = {ap:.4f}")