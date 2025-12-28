import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from resnet_unet import ResNetUNet
from dataset import NYUDepthV2Dataset
from loss import DiceCELoss
import os
import time
import json
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 修改 train_model 函数以记录超参数和准确率
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, num_epochs=25, device=torch.device('cuda'), patience=10, accumulation_steps=8, start_epoch=0):
    since = time.time()

    best_mIoU = 0.0
    epochs_no_improve = 0

    # 初始化日志字典
    training_log = {
        "hyperparameters": {
            "learning_rate": optimizer.param_groups[0]['lr'],
            "batch_size": train_loader.batch_size,
            "accumulation_steps": accumulation_steps,
            "effective_batch_size": train_loader.batch_size * accumulation_steps,
            "num_epochs": num_epochs,
            "optimizer": optimizer.__class__.__name__,
            "weight_decay": optimizer.param_groups[0]['weight_decay']
        },
        "batch_logs": [],  # 记录每10个batch的loss
        "epoch_logs": []   # 记录每个epoch的loss和准确率
    }

    num_classes = model.n_classes if hasattr(model, 'n_classes') else 41
    for epoch in range(start_epoch, num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr}")

        epoch_record = {"epoch": epoch, "learning_rate": current_lr}

        # 用于IoU统计
        iou_inter = {"train": np.zeros(num_classes, dtype=np.float64), "val": np.zeros(num_classes, dtype=np.float64)}
        iou_union = {"train": np.zeros(num_classes, dtype=np.float64), "val": np.zeros(num_classes, dtype=np.float64)}

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0
            
            # 在每个epoch开始时清零梯度
            optimizer.zero_grad()

            # Iterate over data.
            for batch_idx, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # Normalize loss to account for accumulation
                        loss = loss / accumulation_steps
                        loss.backward()
                        
                        if (batch_idx + 1) % accumulation_steps == 0:
                            optimizer.step()
                            optimizer.zero_grad()

                # statistics (restore loss scale for logging)
                current_loss = loss.item() * accumulation_steps if phase == 'train' else loss.item()
                running_loss += current_loss * inputs.size(0)
                
                # 只统计label!=0的像素
                valid_mask = (labels != 0)
                running_corrects += torch.sum((preds == labels) & valid_mask).item()
                total_samples += torch.sum(valid_mask).item()

                # IoU统计（忽略0类）
                for cls in range(1, num_classes):
                    pred_mask = (preds == cls)
                    label_mask = (labels == cls)
                    inter = torch.sum(pred_mask & label_mask).item()
                    union = torch.sum(pred_mask | label_mask).item()
                    iou_inter[phase][cls] += inter
                    iou_union[phase][cls] += union

                # Print progress
                print(f'\rPhase: {phase} | Epoch: {epoch} | Batch: {batch_idx + 1}/{len(dataloader)} | Loss: {current_loss:.4f}', end='', flush=True)

                # 每10个batch保存一次临时模型 (覆盖式保存，避免文件过多)
                if phase == 'train' and (batch_idx + 1) % 10 == 0:
                    torch.save(model.state_dict(), 'latest_unet_rgbd.pth')

            print() # New line

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects / total_samples if total_samples > 0 else 0

            # 计算IoU
            iou_per_class = np.zeros(num_classes)
            for cls in range(1, num_classes):
                if iou_union[phase][cls] > 0:
                    iou_per_class[cls] = iou_inter[phase][cls] / iou_union[phase][cls]
                else:
                    iou_per_class[cls] = np.nan
            mIoU = np.nanmean(iou_per_class[1:])  # 忽略0类

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} mIoU: {mIoU:.4f}')

            # 记录每个epoch的Loss、准确率和mIoU，保留4位小数
            epoch_record[f"{phase}_loss"] = float(f"{epoch_loss:.4f}")
            epoch_record[f"{phase}_accuracy"] = float(f"{epoch_acc:.4f}")
            epoch_record[f"{phase}_mIoU"] = float(f"{mIoU:.4f}")
            # 可选：记录每类IoU
            epoch_record[f"{phase}_IoU_per_class"] = [float(f"{x:.4f}") if not np.isnan(x) else None for x in iou_per_class.tolist()]

            # deep copy the model and Early Stopping
            if phase == 'val':
                # 保存最新的 checkpoint (包含优化器状态)
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'best_mIoU': best_mIoU
                }
                torch.save(checkpoint, 'latest_unet_rgbd.pth')

                if mIoU > best_mIoU:
                    best_mIoU = mIoU
                    torch.save(model.state_dict(), 'best_unet_rgbd.pth') # 保持 best 只存权重，方便推理脚本加载
                    print(f"New best mIoU: {best_mIoU:.4f}. Model saved!")
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    print(f"No improvement in mIoU for {epochs_no_improve} epochs.")

        if scheduler:
            scheduler.step()

        training_log["epoch_logs"].append(epoch_record)
        print()

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epochs_no_improve} epochs without improvement.")
            break

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val mIoU: {best_mIoU:.4f}')

    # 保存日志到JSON文件
    with open('training_log.json', 'w') as f:
        json.dump(training_log, f, indent=4)
    print("Training log saved to training_log.json")

    # 生成混淆矩阵
    print("Generating confusion matrix...")
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # 将预测和标签展平为一维数组并添加到列表中
            all_preds.append(preds.cpu().numpy().flatten())
            all_labels.append(labels.cpu().numpy().flatten())

    # 将所有 batch 的结果拼接成一个巨大的长一维数组
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # 可选：如果想忽略背景类（0），可以取消下面两行的注释
    # mask = all_labels != 0
    # all_preds = all_preds[mask]
    # all_labels = all_labels[mask]

    cm = confusion_matrix(all_labels, all_preds)
    
    # 调整绘图大小以容纳41个类别，防止内容挤在一起
    fig, ax = plt.subplots(figsize=(24, 24))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(len(cm)))
    
    # 绘制混淆矩阵，旋转x轴标签，使用更大的字体
    disp.plot(cmap='Blues', ax=ax, xticks_rotation='vertical', values_format='d')
    
    plt.title("Confusion Matrix", fontsize=20)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    print("Confusion matrix saved as confusion_matrix.png")

def main():
    # 配置
    data_dir = 'nyu_depthv2_seg_dataset'
    # Batch Size: 显存允许的情况下越大越好。
    # 对于 640x480 输入，8GB 显存建议设为 4 或 8；16GB+ 可设为 16。
    batch_size = 4
    learning_rate = 1e-3
    # Epochs: 分割任务通常需要较多轮次收敛。
    # 建议 30-50 轮。如果 Loss 还在下降，可以继续训练。
    num_epochs = 300
    patience = 30 # Early stopping patience
    accumulation_steps = 2 # Gradient accumulation steps (Effective batch size = 4 * 2 = 8)
    
    # 检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据集
    # 开启 use_40_classes=True 以减少显存占用
    train_dataset = NYUDepthV2Dataset(data_dir, split='train', use_40_classes=True)
    val_dataset = NYUDepthV2Dataset(data_dir, split='val', use_40_classes=True)
    
    # 获取类别数
    num_classes = train_dataset.num_classes

    # DEBUG: 仅使用 50 个样本
    # print("DEBUG: Using subset of 50 samples for training and 20 for validation.")
    # train_dataset = torch.utils.data.Subset(train_dataset, range(50))
    # val_dataset = torch.utils.data.Subset(val_dataset, range(20))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    print(f"Num classes: {num_classes}")

    # 模型
    # 输入通道=4 (RGB+Depth), 输出通道=类别数
    # model = UNetRGBD(n_channels=4, n_classes=num_classes)
    print("Using ResNet34-UNet with pretrained weights...")
    model = ResNetUNet(n_channels=4, n_classes=num_classes)
    model = model.to(device)

    # 计算类别权重以处理类别不平衡
    # 全量数据集的类别权重
    weights_list = [0.1074, 0.1360, 0.2747, 0.3607, 0.6980, 0.5546, 0.7115, 0.6986, 0.6427, 0.7275, 
                0.5143, 0.6976, 1.0992, 1.0581, 0.8956, 0.8068, 0.9041, 1.2413, 1.4249, 1.3494, 
                1.3991, 1.3374, 1.3044, 0.8456, 1.4286, 1.0235, 1.0936, 1.3718, 0.9806, 1.4100, 
                1.2643, 1.4031, 0.8701, 1.3532, 1.2028, 1.2989, 0.9901, 1.2714, 1.4046, 1.4357, 1.4080]
    class_weights = torch.tensor(weights_list, dtype=torch.float32).to(device)
    
    # 损失函数
    criterion = DiceCELoss(weight=class_weights, ignore_index=-100, label_smoothing=0.1, lambda_dice=1.0, lambda_ce=1.0)
    
    # 使用 Adam 优化器并添加 L2 正则化 (weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150, 200, 250], gamma=0.1)

    # 尝试加载断点 (支持旧格式和新格式)
    checkpoint_path = 'latest_unet_rgbd.pth'
    start_epoch = 0
    
    if os.path.exists(checkpoint_path):
        print(f"Resuming training from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # 检查是否是新格式 (包含 epoch, optimizer 等)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                print(f"Resumed from epoch {start_epoch}")
            else:
                # 旧格式 (仅 model state dict)
                model.load_state_dict(checkpoint)
                print("Resumed from legacy checkpoint (model weights only). Starting from epoch 0.")
                
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Starting from scratch.")

    # 开始训练
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=scheduler, num_epochs=num_epochs, device=device, patience=patience, accumulation_steps=accumulation_steps, start_epoch=start_epoch)

if __name__ == '__main__':
    main()
