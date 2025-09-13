import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg') 
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_training_metrics(
    epoch_num, 
    step_num,
    train_loss, 
    ema_train_loss, 
    eval_loss, 
    ema_eval_loss, 
    mAP50_list,
    mAP50_95_list,
    mAP95_list
):
    """
    绘制训练和评估指标
    
    参数:
    epoch_num: epoch 数 (int)
    step_num: step 数列表 (list)
    train_loss: 训练损失列表
    ema_train_loss: EMA训练损失列表
    eval_loss: 评估损失列表
    ema_eval_loss: EMA评估损失列表
    mAP50_list: mAP50列表
    mAP50_95_list: mAP50_95列表
    mAP95_list: mAP95列表
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    fig.suptitle('Training and Evaluation Metrics', fontsize=16, fontweight='bold')
    plt.subplots_adjust(top=0.93, hspace=0.3)
    # 处理step_num为list的情况
    total_steps = len(step_num)
    steps_per_epoch = total_steps // epoch_num
    step_list = step_num
    
    # 创建评估数据对应的epoch列表
    eval_epochs = []
    if eval_loss:
        # 假设评估是每10个epoch进行一次，从epoch 0开始
        for i in range(len(eval_loss)):
            eval_epochs.append((i * 10) + 1)  # 转换为1-based indexing
    
    train_color = '#1f77b4'
    ema_train_color = '#ff7f0e'
    eval_color = '#2ca02c'
    ema_eval_color = '#d62728'
    acc_color = '#9467bd'
    
    # 绘制训练损失
    # 为train_loss和ema_train_loss分别创建对应的x轴数据
    if len(train_loss) == len(step_list):
        # train_loss使用完整的step_list
        axes[0].plot(step_list, train_loss, label='Train Loss', alpha=0.5, color=train_color, linewidth=1)
    else:
        # 如果长度不匹配，创建对应长度的x轴
        train_x = step_list[:len(train_loss)] if len(train_loss) <= len(step_list) else list(range(len(train_loss)))
        axes[0].plot(train_x, train_loss, label='Train Loss', alpha=0.5, color=train_color, linewidth=1)
    
    if len(ema_train_loss) == len(step_list):
        # ema_train_loss使用完整的step_list
        axes[0].plot(step_list, ema_train_loss, label='EMA Train Loss', linewidth=0.8, color=ema_train_color)
    else:
        # 如果长度不匹配，创建对应长度的x轴
        ema_x = step_list[:len(ema_train_loss)] if len(ema_train_loss) <= len(step_list) else list(range(len(ema_train_loss)))
        axes[0].plot(ema_x, ema_train_loss, label='EMA Train Loss', linewidth=0.8, color=ema_train_color)
    axes[0].set_title('Training Loss', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, color='black', linewidth=0.5, alpha=0.3)
    axes[0].set_facecolor('white')
    
    # 在训练损失图上添加epoch分隔线和标签
    epoch_ticks, epoch_labels = [], []
    for i in range(epoch_num + 1):
        epoch_step_idx = i * steps_per_epoch
        if epoch_step_idx < total_steps:
            step_val = step_list[epoch_step_idx]
            epoch_ticks.append(step_val)
            epoch_labels.append(i)
            # 添加分隔线（除了第一个点）
            if i > 0:
                axes[0].axvline(x=step_val, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    
    axes[0].set_xticks(epoch_ticks)
    axes[0].set_xticklabels(epoch_labels, rotation=45)
    
    # 绘制评估损失
    if eval_loss and len(eval_loss) > 0:
        axes[1].plot(
            eval_epochs, 
            eval_loss, 
            label='Eval Loss', 
            marker='o', 
            color=eval_color, 
            alpha=0.7, 
            markersize=4, 
            linewidth=1
        )
        axes[1].plot(
            eval_epochs, 
            ema_eval_loss, 
            label='EMA Eval Loss', 
            marker='s',
            color=ema_eval_color,
            linewidth=0.8, 
            markersize=4
        )
    axes[1].set_title('Evaluation Loss', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Loss', fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, color='black', linewidth=0.5, alpha=0.3)
    axes[1].set_facecolor('white')
    # 设置x轴标签倾斜45度
    axes[1].tick_params(axis='x', rotation=45)
    
    # 绘制评估准确率 - 显示三种mAP指标
    # 定义三种mAP的颜色和标记
    map50_color = '#9467bd'
    map50_95_color = '#8c564b'  
    map95_color = '#e377c2'
    
    if mAP50_list and len(mAP50_list) > 0:
        axes[2].plot(
            eval_epochs, 
            mAP50_list, 
            label='mAP@0.5', 
            color=map50_color, 
            marker='o', 
            markersize=5, 
            linewidth=1.5,
            alpha=0.8
        )
        
    if mAP50_95_list and len(mAP50_95_list) > 0:
        axes[2].plot(
            eval_epochs, 
            mAP50_95_list, 
            label='mAP@0.5:0.95', 
            color=map50_95_color, 
            marker='s', 
            markersize=5, 
            linewidth=1.5,
            alpha=0.8
        )
        
    if mAP95_list and len(mAP95_list) > 0:
        axes[2].plot(
            eval_epochs, 
            mAP95_list, 
            label='mAP@0.95', 
            color=map95_color, 
            marker='^', 
            markersize=5, 
            linewidth=1.5,
            alpha=0.8
        )

    axes[2].set_title('Evaluation mAP', fontsize=13, fontweight='bold')
    axes[2].set_xlabel('Epoch', fontsize=11)
    axes[2].set_ylabel('mAP', fontsize=11)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, color='black', linewidth=0.5, alpha=0.3)
    axes[2].set_facecolor('white')
    axes[2].set_ylim(0, 1)
    # 设置x轴标签倾斜45度
    axes[2].tick_params(axis='x', rotation=45)
    
    fig.patch.set_facecolor('white')
    
    # 保存图片而不是显示（因为使用了Agg后端）
    plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()  # 关闭图形以释放内存
    print("训练指标图表已保存为 training_metrics.png")