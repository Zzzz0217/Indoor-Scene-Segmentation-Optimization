import torch
import numpy as np
from dataset import NYUDepthV2Dataset
from torch.utils.data import DataLoader

def calculate_weights():
    # 1. 加载数据集 (Full Train set)
    data_dir = 'nyu_depthv2_seg_dataset'
    train_dataset = NYUDepthV2Dataset(data_dir, split='train', use_40_classes=True)
    # train_dataset = torch.utils.data.Subset(train_dataset, range(50)) # Remove subset
    
    num_classes = 41
    class_counts = np.zeros(num_classes)
    
    print(f"Calculating class counts on {len(train_dataset)} images...")
    # Use DataLoader for faster loading (optional, but good for large datasets)
    loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    from tqdm import tqdm
    for inputs, label in tqdm(loader):
        # label is tensor (1, H, W)
        label_np = label.numpy()
        counts = np.bincount(label_np.flatten(), minlength=num_classes)
        class_counts += counts
        
    # Calculate weights: total_pixels / (num_classes * class_pixels)
    # Or simpler: 1 / (frequency + epsilon)
    
    total_pixels = np.sum(class_counts)
    frequency = class_counts / total_pixels
    
    # Avoid division by zero
    weights = np.zeros(num_classes)
    for i in range(num_classes):
        if frequency[i] > 0:
            weights[i] = 1.0 / np.log(1.02 + frequency[i])
        else:
            weights[i] = 0.0 # Ignore classes not present
            
    # Normalize weights so mean is 1
    valid_weights = weights[weights > 0]
    if len(valid_weights) > 0:
        weights = weights / np.mean(valid_weights)
        
    # Set background weight (class 0) to 0 or low value if we want to ignore it
    # Usually we ignore index 0 in loss, but if we don't, we set it low.
    # Here we assume ignore_index=0 is NOT used in CrossEntropyLoss yet, 
    # but usually it is better to ignore it or weight it down.
    # Let's check train.py. It uses CrossEntropyLoss().
    
    print("Class Weights:", list(weights))
    return weights

if __name__ == '__main__':
    calculate_weights()
