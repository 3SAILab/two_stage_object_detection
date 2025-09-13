import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import json
import torch
import logging
import datetime
import numpy as np
from tqdm import tqdm
from torch.optim import lr_scheduler
from nets.frcnn_training import FasterRCNNTrainer
from dataset.dataloader import train_dataloader, eval_dataloader
from utils.draw import plot_training_metrics
from utils.utils import update_ema

logging.basicConfig(level=logging.INFO)

input_shape = [600, 600]
anchors_size = [8, 16, 32]
save_period = 5
save_dir = 'logs'

config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs/config.json")
with open(config_path, "r") as f:
    config = json.load(f)

num_epochs = config['num_epochs']
batch_size = config['batch_size']
lr = config['lr']
device = config['device']

model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights")
time_str = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
log_dir = os.path.join(save_dir, "loss_" + str(time_str))
pre_train = True

# 主训练流程
def train(visualization=True):
    train_loss, eval_loss, mAP50_list, mAP50_95_list, mAP95_list = [], [], [], [], []

    model = FasterRCNNTrainer(
        mode = "train",
        num_classes = 80,
        feat_stride = 16,
        anchor_scales = [8, 16, 32],
        ratios = [0.5, 1, 2],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr,
        weight_decay=1e-4
    )

    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max = 5
    )

    if pre_train:
        model.load_state_dict(
            torch.load(
                os.path.join(
                    model_path, 
                    f"{model.__class__.__name__}_best.pth"
                ), 
                map_location=device, 
                weights_only=True
            )['model_state_dict'], 
            strict=True
        )
        logging.info(f"✅ Successfully loaded pretrained model")

    for epoch in range(num_epochs):
        model.train()
        loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", colour='green')
        for batch_index, (imgs, bboxes, labels) in loop:
            """
                数据结构：
                    img: torch.tensor: [batch_size, channel, width, height]
                    bboxes: torch.tensor: [batch_size, n_gt, 4]], (x_min, y_min, x_max, y_max)
                    labels: torch.tensor: [batch_size, n_gt, 1]]
            """
            model_output = model(imgs, bboxes, labels)
            losses = model_output[0]
            total_loss = losses[-1] / 32
            total_loss.backward()
            train_loss.append(total_loss.detach().cpu())

            if (batch_index + 1) % 32 == 0:    # 每 32 步更新一次
                optimizer.step()
                optimizer.zero_grad()
        
        if epoch % 10 == 0:
            mAP_05_095, mAP_05, mAP_095, eval_loss_batch, min_eval_loss = 0, 0, 0, 0, 100
            with torch.inference_mode():
                map_iou_thresholds = np.arange(0.5, 1.0, 0.05)
                for map_iou_threshold in map_iou_thresholds:
                    val_loss, mAP = model.eval_fn(
                        eval_dataloader, 
                        nms_iou_threshold=0.7, 
                        map_iou_threshold=map_iou_threshold
                    )
                    eval_loss_batch += val_loss
                    if abs(map_iou_threshold - 0.5) < 1e-6:
                        mAP_05 = mAP
                    elif abs(map_iou_threshold - 0.95) < 1e-6:
                        mAP_095 = mAP
                    mAP_05_095 += mAP

            eval_loss_batch = eval_loss_batch / len(map_iou_thresholds)
            
            # 只在每10个epoch时添加到列表中
            mAP50_list.append(mAP_05)
            mAP95_list.append(mAP_095)
            mAP_05_095 /= len(map_iou_thresholds)
            mAP50_95_list.append(mAP_05_095)
            eval_loss.append(val_loss.detach().cpu() if hasattr(val_loss, 'detach') else val_loss)

            if eval_loss_batch < min_eval_loss:
                min_eval_loss = eval_loss_batch
                torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    }, 
                    os.path.join(model_path, f"{str(model.__class__.__name__)}_best.pth")
                )
                logging.info(f"✅ Best model saved to {model_path}")

            logging.info(f"eval: mAP_50%: {mAP_05:.4f}, mAP_50%_95%: {mAP_05_095:.4f}, mAP_95%: {mAP_095:.4f}")

        scheduler.step()

    # 训练结束，保存最终模型
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, 
        os.path.join(model_path, f"{str(model.__class__.__name__)}_last.pth")
    )
    logging.info(f"✅ Last model saved to {model_path}")

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
            "mAP50_list" : mAP50_list,
            "mAP50_95_list" : mAP50_95_list,
            "mAP95_list" : mAP95_list
        }

        plot_training_metrics(**draw_data)

if __name__ == "__main__":
    train()