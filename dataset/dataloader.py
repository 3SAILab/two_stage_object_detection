from torch.utils.data import DataLoader, Dataset
from utils.loc_bbox_iou import xywh2xyxy
from dataset.data_organise import mydata
from dataset.transform import transform
from PIL import Image
import numpy as np
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
        if self.transform:
            img = self.transform(img)
        label = data["label"]
        return img, label

def collate_fn(batch):
    images = []
    bboxes = []
    labels = []
    for img, label in batch:
        images.append(img.to(device))

        if label:
            # 提取 bboxes 并转换为 [x_min, y_min, x_max, y_max]
            bbox = [xywh2xyxy(obj[:-1]) for obj in label]
            # [num_boxes, 4]
            bbox = torch.tensor(bbox, dtype=torch.float32)  
            # 提取 classes
            classes = [obj[-1] for obj in label]
            # [num_boxes]
            classes = torch.tensor(classes, dtype=torch.int64)  
        else:
            bbox = torch.empty(0, 4, dtype=torch.float32)
            classes = torch.empty(0, dtype=torch.int64)
        
        bboxes.append(bbox.to(device))
        labels.append(classes.to(device))
    
    images = torch.stack(images)
    return images, bboxes, labels

train_dataset = FRCNNDataSet(mydata["train"], transform)
eval_dataset = FRCNNDataSet(mydata["eval"], transform)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size, 
    shuffle=True,
    pin_memory=True,
    drop_last=False,
    persistent_workers=True,
    num_workers=num_workers,
    prefetch_factor=prefetch_factor,
    collate_fn=collate_fn
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
    collate_fn=collate_fn
)