import sys
import os
sys.path.append(os.path.dirname(__file__))
import json
import torch
import logging
import random
from PIL import Image
from nets.frcnn_training import FasterRCNNTrainer
from dataset.data_organise import mydata, class_index_2_class_name
from torchvision.ops import nms
from dataset.transform import eval_transform
from torchvision import tv_tensors
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

logging.basicConfig(level=logging.INFO)

num_inference = 5

model_path = os.path.join(os.path.dirname(__file__), "weights")

output_dir = os.path.join(os.path.dirname(__file__), "inference_results")
os.makedirs(output_dir, exist_ok=True)

config_path = os.path.join(os.path.dirname(__file__), "configs/config.json")
with open(config_path, "r") as f:
    config = json.load(f)

num_epochs = config['num_epochs']
batch_size = config['batch_size']
lr = config['lr']
device = config['device']

data = mydata["eval"]
choice_data = random.sample(list(data.keys()), num_inference)

with torch.inference_mode():

    model = FasterRCNNTrainer(
            mode = "train",
            num_classes = 80,
            feat_stride = 16,
            anchor_scales = [8, 16, 32],
            ratios = [0.5, 1, 2],
        ).to(device)

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

    for i in choice_data:
        img_path = data[i]["image_path"]
        img = Image.open(img_path).convert("RGB")
        boxes_gt = torch.tensor(data[i]["bboxes"], dtype=torch.float32).to(device)
        labels_gt = torch.tensor(data[i]["labels"], dtype=torch.int64).to(device)

        img = tv_tensors.Image(img, dtype=torch.float32)
        bboxes = tv_tensors.BoundingBoxes(
            data[i]["bboxes"],
            format="XYXY",
            canvas_size=img.shape[-2:]
        )
        result = eval_transform({"image": img, "boxes": boxes_gt, "labels": labels_gt})
        img, boxes_gt, labels_gt = result["image"], result["boxes"], result["labels"]

        model_output = model(torch.unsqueeze(img, dim=0), torch.unsqueeze(boxes_gt, dim=0), torch.unsqueeze(labels_gt, dim=0))

        boxes_pred = torch.squeeze(model_output[1])
        labels_pred = torch.squeeze(model_output[2])
        labels_score_pred = torch.squeeze(model_output[3])

        keep = nms(boxes_pred, labels_score_pred, iou_threshold=0.1)
        boxes_pred = boxes_pred[keep]
        labels_pred = labels_pred[keep]
        labels_score_pred = labels_score_pred[keep]

        """
            已知：
                img_path: data[i]["image_path"] 直接从路径绘制图像
                boxes_gt: tensor.shape[n_gt, 4] 格式为XYXY
                labels_gt: tensor.shape[n_gt]

                boxes_pred: tensor.shape[n_pred, 4] 格式为XYXY
                labels_pred: tensor.shape[n_pred]
                labels_score_pred: tensor.shape[n_pred]
        """
        
        # 加载原始图像
        img_width, img_height = img.shape[1], img.shape[2]
        
        # 创建matplotlib figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        img = Image.open(img_path).convert("RGB").resize((600, 600))
        ax.imshow(img)
        
        # 转换tensor到numpy并移到CPU
        boxes_gt_np = boxes_gt.cpu().numpy() if boxes_gt.numel() > 0 else np.array([]).reshape(0, 4)
        labels_gt_np = labels_gt.cpu().numpy() if labels_gt.numel() > 0 else np.array([])
        boxes_pred_np = boxes_pred.cpu().numpy() if boxes_pred.numel() > 0 else np.array([]).reshape(0, 4)
        labels_pred_np = labels_pred.cpu().numpy() if labels_pred.numel() > 0 else np.array([])
        labels_score_pred_np = labels_score_pred.cpu().numpy() if labels_score_pred.numel() > 0 else np.array([])
        
        # 绘制Ground Truth框（绿色）
        for j in range(len(boxes_gt_np)):
            x1, y1, x2, y2 = boxes_gt_np[j]
            width = x2 - x1
            height = y2 - y1
            
            # 绘制GT框（绿色）
            rect_gt = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor='green', facecolor='none'
            )
            ax.add_patch(rect_gt)
            
            # 添加GT标签
            label_idx = int(labels_gt_np[j])
            label_name = class_index_2_class_name.get(label_idx, f"class_{label_idx}")
            ax.text(x1, y1 - 5, f'GT: {label_name}', 
                    fontsize=10, color='green', weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # 绘制预测框（红色）
        for j in range(len(boxes_pred_np)):
            x1, y1, x2, y2 = boxes_pred_np[j]
            width = x2 - x1
            height = y2 - y1
            
            # 绘制预测框（红色）
            rect_pred = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect_pred)
            
            # 添加预测标签和置信度
            label_idx = int(labels_pred_np[j])
            confidence = labels_score_pred_np[j]
            label_name = class_index_2_class_name.get(label_idx, f"class_{label_idx}")
            ax.text(x1, y2 + 15, f'Pred: {label_name}\nConf: {confidence:.3f}', 
                    fontsize=10, color='red', weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # 设置标题和去除坐标轴
        ax.set_title(f'Image {i}: GT({len(boxes_gt_np)}) vs Pred({len(boxes_pred_np)})', 
                    fontsize=14, weight='bold')
        ax.axis('off')
        
        # 添加图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='green', lw=2, label='Ground Truth'),
            Line2D([0], [0], color='red', lw=2, label='Prediction')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # 保存图像
        output_filename = f"inference_result_{i:03d}.png"
        output_path = os.path.join(output_dir, output_filename)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logging.info(f"✅ Saved visualization for image {i}: {output_filename}")
        logging.info(f"   GT boxes: {len(boxes_gt_np)}, Pred boxes: {len(boxes_pred_np)}")

logging.info(f"🎉 完成！所有 {num_inference} 张图像的推理可视化已保存到: {output_dir}")