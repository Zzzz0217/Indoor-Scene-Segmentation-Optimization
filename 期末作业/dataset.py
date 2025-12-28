import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import json

class NYUDepthV2Dataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, target_transform=None, use_40_classes=True):
        """
        Args:
            root_dir (string): 数据集根目录 (例如 'nyu_depthv2_seg_dataset')
            split (string): 'train' 或 'val'
            transform (callable, optional): 应用于输入 (RGB+Depth) 的变换
            target_transform (callable, optional): 应用于标签的变换
            use_40_classes (bool): 是否将类别映射到 40 类 (推荐 True 以减少显存)
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.use_40_classes = use_40_classes

        # 加载索引
        if split == 'train':
            self.indices = np.load(os.path.join(root_dir, 'train_indices.npy'))
        elif split == 'val':
            self.indices = np.load(os.path.join(root_dir, 'val_indices.npy'))
        else:
            raise ValueError("split must be 'train' or 'val'")

        # 加载类别映射
        with open(os.path.join(root_dir, 'class_map.json'), 'r', encoding='utf-8') as f:
            self.class_map = json.load(f)
        
        if self.use_40_classes:
            self.num_classes = 41 # 0-40, 0 is background/other
        else:
            # 计算类别数量
            ids = [int(k) for k in self.class_map.keys()]
            self.num_classes = max(ids) + 1 if ids else 0

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_idx = self.indices[idx]
        file_name = f"{file_idx:04d}.png"

        # 路径
        rgb_path = os.path.join(self.root_dir, 'rgb', file_name)
        depth_path = os.path.join(self.root_dir, 'depth', file_name)
        label_path = os.path.join(self.root_dir, 'seg_label', file_name)

        # 加载图像
        rgb = Image.open(rgb_path).convert('RGB')
        depth = Image.open(depth_path).convert('L') # 8-bit 灰度
        label = Image.open(label_path) # 保持原始模式 (通常是 L 或 P)
        
        # --- 显存优化关键点 ---
        # 如果显存依然不够，可以在这里缩小图像尺寸
        # 例如缩小到 320x240
        # rgb = rgb.resize((320, 240), Image.BILINEAR)
        # depth = depth.resize((320, 240), Image.BILINEAR)
        # label = label.resize((320, 240), Image.NEAREST)

        # 转换为 Tensor
        # RGB: (3, H, W), 0-1
        rgb_tensor = transforms.ToTensor()(rgb)
        
        # Depth: (1, H, W), 0-1
        depth_tensor = transforms.ToTensor()(depth)

        # 合并 RGB 和 Depth -> (4, H, W)
        input_tensor = torch.cat([rgb_tensor, depth_tensor], dim=0)

        # Label: (H, W)
        # 注意：CrossEntropyLoss 需要 LongTensor，且不包含通道维度
        label_np = np.array(label, dtype=np.int64)
        
        if self.use_40_classes:
            # 简单的映射策略：大于40的类别全部设为0 (背景/未知)
            # 这是一个简化的处理，标准的 NYU-40 映射比较复杂
            # 但对于作业来说，关注主要物体即可，这样能大幅减少最后一层的参数量
            label_np[label_np > 40] = 0
            
        label_tensor = torch.from_numpy(label_np)

        # 应用变换 (如果有)
        if self.transform:
            input_tensor = self.transform(input_tensor)
        if self.target_transform:
            label_tensor = self.target_transform(label_tensor)

        return input_tensor, label_tensor

if __name__ == '__main__':
    # 测试 Dataset
    dataset = NYUDepthV2Dataset(root_dir='nyu_depthv2_seg_dataset', split='train')
    print(f"Dataset size: {len(dataset)}")
    print(f"Num classes: {dataset.num_classes}")
    
    img, label = dataset[0]
    print(f"Input shape: {img.shape}")
    print(f"Label shape: {label.shape}")
    print(f"Label unique values: {torch.unique(label)}")
