from torch.utils.data import DataLoader, Dataset
from utils.loc_bbox_iou import xywh2xyxy
from dataset.data_organise import mydata
from dataset.transform import transform
from torchvision import tv_tensors
from PIL import Image
import platform
import torch
import json
import os

# 配置文件路径
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs/config.json")

# 读取文件
with open(config_path, 'r') as f:
    config = json.load(f)

batch_size = config['batch_size']
device = config["device"]
num_workers = config['num_workers']
prefetch_factor = config['prefetch_factor']
persistent_workers = config['persistent_workers']

class FRCNNDataSet(Dataset):
    def __init__(self, data, transform=None):
        super().__init__()
        self.transform = transform
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        img = Image.open(data["image_path"]).convert("RGB")
        labels = torch.tensor(data["labels"], dtype=torch.int32)
        if self.transform:
            img = tv_tensors.Image(img, dtype=torch.float32)
            bboxes = tv_tensors.BoundingBoxes(
                data["bboxes"],
                format="XYXY",
                canvas_size=img.shape[-2:]
            )
            result = self.transform({"image": img, "boxes": bboxes, "labels": labels})
            return result["image"], result["boxes"], result["labels"]
        else:
            bboxes = data["bboxes"]
            return img, bboxes, labels

def collate_fn(batch):
    images = []
    bboxes = []
    labels = []
    for img, bbox, label in batch:
        images.append(img.requires_grad_(True))
        bboxes.append(bbox)
        labels.append(label)
    return images, bboxes, labels

train_dataset = FRCNNDataSet(mydata["train"], transform)
eval_dataset = FRCNNDataSet(mydata["eval"], transform)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size, 
    shuffle=True,
    pin_memory=True,
    drop_last=False,
    persistent_workers=persistent_workers,
    num_workers=num_workers,
    prefetch_factor=prefetch_factor,
    collate_fn=collate_fn,
    multiprocessing_context="spawn" if platform.system() == "Windows" else None
)

eval_dataloader = DataLoader(
    dataset=eval_dataset,
    batch_size=batch_size, 
    shuffle=True,
    pin_memory=True,
    drop_last=False,
    persistent_workers=True,
    num_workers=num_workers,
    prefetch_factor=prefetch_factor,
    collate_fn=collate_fn,
    multiprocessing_context="spawn" if platform.system() == "Windows" else None
)