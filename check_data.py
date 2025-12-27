import os
import numpy as np
from PIL import Image
# import matplotlib.pyplot as plt
from dataset import NYUDepthV2Dataset
from collections import Counter

def check_dataset():
    print("Checking dataset statistics...")
    dataset = NYUDepthV2Dataset(root_dir='nyu_depthv2_seg_dataset', split='train', use_40_classes=False)
    
    # 随机采样几张图片进行检查
    indices = np.random.choice(len(dataset), 5, replace=False)
    
    all_labels = []
    
    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        # label is tensor, convert to numpy
        label_np = label.numpy()
        
        unique_ids = np.unique(label_np)
        print(f"\nSample {i} (Index {idx}):")
        print(f"  Label shape: {label_np.shape}")
        print(f"  Unique Label IDs: {unique_ids}")
        print(f"  Max Label ID: {np.max(label_np)}")
        
        # 统计 > 40 的像素比例
        mask_gt_40 = label_np > 40
        ratio = np.sum(mask_gt_40) / label_np.size
        print(f"  Ratio of pixels with ID > 40: {ratio:.2%}")
        
        all_labels.extend(label_np.flatten())

    # 全局统计（仅基于采样）
    all_labels = np.array(all_labels)
    counts = Counter(all_labels)
    print("\nTop 50 most frequent classes in samples:")
    for cls_id, count in counts.most_common(50):
        cls_name = dataset.class_map.get(str(cls_id), "unknown")
        print(f"  ID {cls_id}: {cls_name} - {count} pixels")

if __name__ == '__main__':
    check_dataset()
