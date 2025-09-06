import datetime
import os
import json
import torch
from torch.optim import lr_scheduler
from torch import nn
from tqdm import tqdm
from nets.frcnn import FasterRCNN
from nets.frcnn_training import FasterRCNNTrainer, weights_init, HarDNetFeatureExtraction, HarNetClassifier, HarNetRoIHead
from dataset.dataloader import FRCNNDataSet, train_dataloader, eval_dataloader
from dataset.data_organise import classes

if __name__ == "__main__":
    input_shape = [600, 600]
    anchors_size = [8, 16, 32]
    save_period = 5
    save_dir = 'logs'
    num_classes = len(classes)

    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs/config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    lr = config['lr']
    device = config['device']

    time_str = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(save_dir, "loss_" + str(time_str))

    # 主训练流程
    def train():
        model = FasterRCNNTrainer(
            mode = "train",
            optimizer = optimizer,
            num_classes = len(classes),
            feat_stride = 16,
            anchor_scales = [8, 16, 32],
            ratios = [0.5, 1, 2],
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=1e-4
        )
        scheduler = lr_scheduler.StepLR(
            optimizer, 
            step_size=10, 
            gamma=0.5
        )
        
        for epoch in range(num_epochs):

            loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", colour='green')

            for batch_index, (imgs, bboxes, labels) in loop:
                """
                    数据结构：
                        img: torch.tensor: [batch_size, channel, width, height]
                        bboxes: List: [torch.tensor: [, 4]]
                        labels: List: [torch.tensor: [one_img_num_gt_anchors]]
                """
                optimizer.zero_grad()
                losses = model(imgs, bboxes, labels)
                losses[-1].backward()
                optimizer.step()