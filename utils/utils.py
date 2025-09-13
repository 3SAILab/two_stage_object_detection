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

def compute_ap_from_pr(pr_tensor, n_gt):
    precisions = pr_tensor[:, 0]
    recalls = pr_tensor[:, 1]

    right_recalls = [i / n_gt for i in range(1, n_gt)] + [1.0]
    left_recalls = [0.0] + [i / n_gt for i in range(1, n_gt)]

    ap = 0.0
    for left_r, right_r in zip(left_recalls, right_recalls):
        mask = torch.isclose(recalls, torch.tensor(right_r), atol=1e-5)
        p = precisions[mask][0] if mask.any() else torch.tensor(0.0)
        ap += (right_r - left_r) * p

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
    ap = compute_ap_from_pr(pr, 7)
    print("PR Curve:\n", pr)
    print(f"AP = {ap:.4f}")