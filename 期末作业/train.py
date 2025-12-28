import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from unet_model import UNetRGBD
from dataset import NYUDepthV2Dataset
import os
import time
import json
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 修改 train_model 函数以记录超参数和准确率
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, device=torch.device('mps')):
    since = time.time()

    best_loss = float('inf')

    # 初始化日志字典
    training_log = {
        "hyperparameters": {
            "learning_rate": optimizer.param_groups[0]['lr'],
            "batch_size": train_loader.batch_size,
            "num_epochs": num_epochs
        },
        "batch_logs": [],  # 记录每10个batch的loss
        "epoch_logs": []   # 记录每个epoch的loss和准确率
    }

    num_classes = model.n_classes if hasattr(model, 'n_classes') else 41
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        epoch_record = {"epoch": epoch}

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

            # Iterate over data.
            for batch_idx, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
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
                print(f'\rPhase: {phase} | Epoch: {epoch} | Batch: {batch_idx + 1}/{len(dataloader)} | Loss: {loss.item():.4f}', end='', flush=True)

                # 记录每10个batch的Loss (仅训练阶段)
                if phase == 'train' and (batch_idx + 1) % 10 == 0:
                    training_log["batch_logs"].append({
                        "epoch": epoch,
                        "batch": batch_idx + 1,
                        "loss": loss.item()
                    })
                    # 每10个batch保存一次临时模型 (覆盖式保存，避免文件过多)
                    torch.save(model.state_dict(), 'latest_unet_rgbd.pth')

            print() # New line

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects / total_samples

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

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), 'best_unet_rgbd.pth')
                print("Model saved!")

        training_log["epoch_logs"].append(epoch_record)
        print()

        # MPS优化：每个epoch后清理缓存，减少显存碎片
        if device.type == 'mps':
            torch.mps.empty_cache()
            print("[MPS] 已清理缓存，减少显存碎片")

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Loss: {best_loss:.4f}')

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
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(len(cm)))
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    print("Confusion matrix saved as confusion_matrix.png")

def main():
    # 配置
    data_dir = 'nyu_depthv2_seg_dataset'
    # Batch Size: 显存允许的情况下越大越好。
    # 对于 640x480 输入，8GB 显存建议设为 4 或 8；16GB+ 可设为 16。
    batch_size = 2
    learning_rate = 1e-4
    # Epochs: 分割任务通常需要较多轮次收敛。
    # 建议 30-50 轮。如果 Loss 还在下降，可以继续训练。
    num_epochs = 100
    
    # 检测设备
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据集
    # 开启 use_40_classes=True 以减少显存占用
    train_dataset = NYUDepthV2Dataset(data_dir, split='train', use_40_classes=True)
    num_classes = train_dataset.num_classes
    # 仅用前20个样本做快速验证
    train_subset_size = min(20, len(train_dataset))
    train_dataset = Subset(train_dataset, list(range(train_subset_size)))
    val_dataset = NYUDepthV2Dataset(data_dir, split='val', use_40_classes=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    print(f"Num classes: {num_classes}")

    # 模型
    # 输入通道=4 (RGB+Depth), 输出通道=类别数
    model = UNetRGBD(n_channels=4, n_classes=num_classes)
    model = model.to(device)

    # 损失函数和优化器
    # 假设 0 是背景/未知，如果不需要忽略，去掉 ignore_index
    # 如果显存不够，可以减小 batch_size
    criterion = nn.CrossEntropyLoss(ignore_index=0) 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 开始训练
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs, device=device)

if __name__ == '__main__':
    main()
