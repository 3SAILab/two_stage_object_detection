import torchvision.transforms.v2 as T
import torch

transform = T.Compose([
    T.RandomPhotometricDistort(),
    T.RandomHorizontalFlip(p=0.5),
    T.ScaleJitter(target_size=(600, 600), scale_range=(0.8, 1.2)),
    T.SanitizeBoundingBoxes(min_size=1.0),
    T.ToTensor(),         # 转为 Tensor[C, H, W]
    T.ConvertImageDtype(torch.float32),  # 归一化到 [0, 1]
])