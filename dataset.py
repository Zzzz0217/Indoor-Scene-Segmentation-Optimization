import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import json
import random

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
            # 基于统计的高频类别 ID (Top 40)
            # 这些 ID 将被映射到 1-40，其他 ID 映射到 0
            self.valid_classes = [
                21, 11, 3, 19, 5, 59, 28, 83, 88, 157, 
                64, 85, 143, 89, 80, 42, 26, 241, 13, 130, 
                242, 8, 7, 16, 119, 141, 174, 169, 177, 136, 
                79, 4, 66, 15, 2, 122, 24, 9, 14, 6
            ]
            # 构建映射表: old_id -> new_id
            # 默认值为 0
            self.id_mapping = np.zeros(895, dtype=np.int64) 
            for i, old_id in enumerate(self.valid_classes):
                if old_id < 895:
                    self.id_mapping[old_id] = i + 1
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
        
        # --- 数据增强 (仅训练集) ---
        if self.split == 'train':
            # 随机水平翻转
            if random.random() > 0.5:
                rgb = TF.hflip(rgb)
                depth = TF.hflip(depth)
                label = TF.hflip(label)
            
            # 随机旋转 (-10 ~ 10 度)
            if random.random() > 0.5:
                angle = random.uniform(-10, 10)
                rgb = TF.rotate(rgb, angle)
                depth = TF.rotate(depth, angle)
                label = TF.rotate(label, angle, interpolation=transforms.InterpolationMode.NEAREST)

            # 颜色抖动 (仅 RGB)
            if random.random() > 0.5:
                color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
                rgb = color_jitter(rgb)

            # --- 强力增强: 随机缩放 + 裁剪 ---
            # 模拟不同距离和视角
            if random.random() > 0.5:
                scale = random.uniform(0.8, 1.2) # 缩放比例
                w, h = rgb.size
                new_w, new_h = int(w * scale), int(h * scale)
                
                # Resize
                rgb = TF.resize(rgb, (new_h, new_w), interpolation=transforms.InterpolationMode.BILINEAR)
                depth = TF.resize(depth, (new_h, new_w), interpolation=transforms.InterpolationMode.NEAREST)
                label = TF.resize(label, (new_h, new_w), interpolation=transforms.InterpolationMode.NEAREST)
                
                # Crop or Pad back to 480x640
                # 如果变大了，随机裁剪
                if scale > 1.0:
                    i, j, h, w = transforms.RandomCrop.get_params(rgb, output_size=(480, 640))
                    rgb = TF.crop(rgb, i, j, h, w)
                    depth = TF.crop(depth, i, j, h, w)
                    label = TF.crop(label, i, j, h, w)
                # 如果变小了，填充
                else:
                    pad_w = 640 - new_w
                    pad_h = 480 - new_h
                    padding = (pad_w//2, pad_h//2, pad_w - pad_w//2, pad_h - pad_h//2)
                    rgb = TF.pad(rgb, padding, fill=0)
                    depth = TF.pad(depth, padding, fill=0)
                    label = TF.pad(label, padding, fill=0) # 0 is background

        # 转换为 Tensor
        # RGB: (3, H, W), 0-1
        rgb_tensor = transforms.ToTensor()(rgb)
        rgb_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(rgb_tensor)
        
        # Depth: (1, H, W), 0-1
        depth_tensor = transforms.ToTensor()(depth)
        
        # --- 深度图增强 (模拟 RealSense 噪声) ---
        if self.split == 'train':
            # 1. 高斯噪声
            if random.random() > 0.5:
                noise = torch.randn_like(depth_tensor) * 0.05 # 5% 的噪声
                depth_tensor = depth_tensor + noise
                depth_tensor = torch.clamp(depth_tensor, 0, 1)
            
            # 2. 随机丢失 (Dropout) - 模拟深度缺失
            if random.random() > 0.3:
                mask = torch.rand_like(depth_tensor) > 0.05 # 5% 的像素丢失
                depth_tensor = depth_tensor * mask

        depth_tensor = transforms.Normalize(mean=[0.5], std=[0.5])(depth_tensor)

        # 合并 RGB 和 Depth -> (4, H, W)
        input_tensor = torch.cat([rgb_tensor, depth_tensor], dim=0)

        # Label: (H, W)
        # 注意：CrossEntropyLoss 需要 LongTensor，且不包含通道维度
        label_np = np.array(label, dtype=np.int64)
        
        if self.use_40_classes:
            # 使用映射表进行转换
            # 任何不在 valid_classes 中的 ID 都会变成 0
            # 任何 > 894 的 ID 也会变成 0 (通过 clip 防止越界)
            label_np = np.clip(label_np, 0, 894)
            label_np = self.id_mapping[label_np]
            
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
