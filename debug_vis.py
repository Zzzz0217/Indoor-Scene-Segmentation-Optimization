import torch
import matplotlib.pyplot as plt
import numpy as np
from dataset import NYUDepthV2Dataset
from unet_model import UNetRGBD
from torch.utils.data import DataLoader

def visualize_prediction():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载数据集 (Validation set)
    data_dir = 'nyu_depthv2_seg_dataset'
    val_dataset = NYUDepthV2Dataset(data_dir, split='val', use_40_classes=True)
    # 使用前20个样本的子集 (与 train.py 一致)
    val_dataset = torch.utils.data.Subset(val_dataset, range(20))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    
    # 2. 加载模型
    num_classes = 41
    model = UNetRGBD(n_channels=4, n_classes=num_classes)
    
    # 加载权重
    try:
        model.load_state_dict(torch.load('latest_unet_rgbd.pth', map_location=device))
        print("Loaded latest_unet_rgbd.pth")
    except:
        try:
            model.load_state_dict(torch.load('best_unet_rgbd.pth', map_location=device))
            print("Loaded best_unet_rgbd.pth")
        except:
            print("No model found, using random weights (Expect garbage output)")
            
    model.to(device)
    model.eval()
    
    # 3. 获取一个样本
    inputs, labels = next(iter(val_loader))
    inputs = inputs.to(device)
    
    # 4. 预测
    with torch.no_grad():
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        
    # 5. 可视化
    rgb = inputs[0, :3, :, :].cpu().permute(1, 2, 0).numpy()
    depth = inputs[0, 3, :, :].cpu().numpy()
    label = labels[0].cpu().numpy()
    pred = preds[0].cpu().numpy()
    
    print(f"Label unique values: {np.unique(label)}")
    print(f"Prediction unique values: {np.unique(pred)}")
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    
    axs[0, 0].imshow(rgb)
    axs[0, 0].set_title("RGB Input")
    axs[0, 0].axis('off')
    
    axs[0, 1].imshow(depth, cmap='gray')
    axs[0, 1].set_title("Depth Input")
    axs[0, 1].axis('off')
    
    axs[1, 0].imshow(label, cmap='jet', vmin=0, vmax=40)
    axs[1, 0].set_title("Ground Truth Label")
    axs[1, 0].axis('off')
    
    axs[1, 1].imshow(pred, cmap='jet', vmin=0, vmax=40)
    axs[1, 1].set_title("Model Prediction")
    axs[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig("debug_prediction.png")
    print("Saved visualization to debug_prediction.png")

if __name__ == '__main__':
    visualize_prediction()
