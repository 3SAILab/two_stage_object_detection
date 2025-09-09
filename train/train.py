import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import json
import torch
import logging
import datetime
from tqdm import tqdm
from torch.optim import lr_scheduler
from nets.frcnn_training import FasterRCNNTrainer
from dataset.data_organise import classes
from dataset.dataloader import train_dataloader, eval_dataloader
from utils.draw import plot_training_metrics
from utils.utils import update_ema, show_model_flops_and_params, set_seed

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
def train(visualization=True):
    train_loss = []
    eval_loss = []
    mAP_list = []
    model = FasterRCNNTrainer(
        mode = "train",
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

    # show_model_flops_and_params(model)

    for epoch in range(num_epochs):
        model.train()
        loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", colour='green')
        for _, (imgs, bboxes, labels) in loop:
            """
                数据结构：
                    img: torch.tensor: [batch_size, channel, width, height]
                    bboxes: torch.tensor: [batch_size, n_gt, 4]], (x_min, y_min, x_max, y_max)
                    labels: torch.tensor: [batch_size, n_gt, 1]]
            """
            optimizer.zero_grad()
            model_output = model(imgs, bboxes, labels)
            losses = model_output[0]  # 损失是返回值的第一个元素
            total_loss = losses[-1]
            
            # 检查损失是否需要梯度
            if not total_loss.requires_grad:
                print(f"Warning: total_loss doesn't require grad. Loss value: {total_loss}")
                continue
                
            total_loss.backward()
            train_loss.append(total_loss.detach().cpu())
            optimizer.step()
        
        if epoch % 10 == 0:
            mAP_05_095 = 0
            mAP_05 = 0
            mAP_095 = 0
            
            with torch.inference_mode():
                # 使用numpy的arange来处理浮点数范围
                import numpy as np
                iou_thresholds = np.arange(0.5, 1.0, 0.05)
                for iou_threshold in iou_thresholds:
                    val_loss, mAP = model.eval_fn(eval_dataloader, iou_threshold=iou_threshold)
                    if abs(iou_threshold - 0.5) < 1e-6:  # 更好的浮点数比较
                        mAP_05 = mAP
                    elif abs(iou_threshold - 0.95) < 1e-6:  # 更好的浮点数比较
                        mAP_095 = mAP
                    mAP_05_095 += mAP

            mAP_05_095 /= len(iou_thresholds)
            eval_loss.append(val_loss.detach().cpu() if hasattr(val_loss, 'detach') else val_loss)

            logging.info(f"eval: mAP_50%: {mAP_05}, mAP_50%_95%: {mAP_05_095}, mAP_95%: {mAP_095}")

        scheduler.step()

    # 可视化数据
    if visualization:
        ema_alpha = 0.01
        ema_train_loss = []
        for i in range(len(train_loss)):
            if i == 0:
                ema_train_loss.append(train_loss[0])
            else:
                ema_train_loss.append(update_ema(train_loss[i], ema_alpha, ema_train_loss[i-1]))

        ema_eval_loss = []
        for i in range(len(eval_loss)):
            if i == 0:
                ema_eval_loss.append(eval_loss[0])
            else:
                ema_eval_loss.append(update_ema(eval_loss[i], ema_alpha, ema_eval_loss[i-1]))

        step_list = list(range(len(train_loss)))

        # 绘制训练结果图表
        draw_data = {
            "epoch_num" : num_epochs,
            "step_num" : step_list,
            "train_loss" : train_loss,
            "ema_train_loss" : ema_train_loss,
            "eval_loss" : eval_loss,
            "ema_eval_loss" : ema_eval_loss,
            "eval_mAP" : mAP_list
        }

        plot_training_metrics(**draw_data)

if __name__ == "__main__":
    train()