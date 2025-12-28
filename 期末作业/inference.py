import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from unet_model import UNetRGBD
from dataset import NYUDepthV2Dataset
import os
import random

def colorize_mask(mask, num_classes):
    # 生成随机颜色表
    np.random.seed(42)
    cmap = np.random.randint(0, 255, (num_classes, 3), dtype=np.uint8)
    cmap[0] = [0, 0, 0] # 背景黑色
    
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for c in range(num_classes):
        color_mask[mask == c] = cmap[c]
        
    return color_mask

def main():
    # 配置
    data_dir = 'nyu_depthv2_seg_dataset'
    model_path = 'best_unet_rgbd.pth'
    output_dir = 'inference_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # 设备
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")    
    # 加载数据集获取类别数
    val_dataset = NYUDepthV2Dataset(data_dir, split='val')
    num_classes = val_dataset.num_classes
    
    # 加载模型
    model = UNetRGBD(n_channels=4, n_classes=num_classes)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("模型加载成功")
    else:
        print("未找到模型文件，请先训练！")
        return

    model.to(device)
    model.eval()
    
    # 随机抽取几张进行测试
    indices = random.sample(range(len(val_dataset)), 5)
    
    for i, idx in enumerate(indices):
        inputs, label = val_dataset[idx]
        
        # 增加 batch 维度
        inputs_batch = inputs.unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(inputs_batch)
            preds = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()
            
        # 准备可视化
        rgb = inputs[:3].permute(1, 2, 0).numpy()
        depth = inputs[3].numpy()
        label = label.numpy()
        
        # 彩色化 Mask
        pred_color = colorize_mask(preds, num_classes)
        label_color = colorize_mask(label, num_classes)
        
        # 绘图
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.title("RGB Input")
        plt.imshow(rgb)
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        plt.title("Depth Input")
        plt.imshow(depth, cmap='gray')
        plt.axis('off')
        
        plt.subplot(2, 2, 3)
        plt.title("Ground Truth")
        plt.imshow(label_color)
        plt.axis('off')
        
        plt.subplot(2, 2, 4)
        plt.title("Prediction")
        plt.imshow(pred_color)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"result_{i}.png"))
        plt.close()
        print(f"Saved result_{i}.png")

if __name__ == '__main__':
    main()
