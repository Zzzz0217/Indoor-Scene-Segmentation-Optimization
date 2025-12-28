import h5py
import json
import numpy as np
from PIL import Image
import os
import sys

# 1. 定义路径（替换成你的数据集路径）
mat_path = "/Users/zy/数据集/nyu_depth_v2_labeled.mat"  # 你的mat文件路径
save_root = "nyu_depthv2_seg_dataset"  # 整理后的数据保存路径

# 检查文件是否存在
if not os.path.exists(mat_path):
    print(f"警告: 文件不存在: {mat_path}")
    print("请确保路径正确，或者将 mat 文件放到指定位置。")

os.makedirs(os.path.join(save_root, "rgb"), exist_ok=True)
os.makedirs(os.path.join(save_root, "seg_label"), exist_ok=True)
os.makedirs(os.path.join(save_root, "depth"), exist_ok=True)

# 2. 用 h5py 解析mat文件（NYU的标注存在mat里，包含1449张标注图像）
try:
    with h5py.File(mat_path, 'r') as f:
        print(f"正在读取: {mat_path}")
        # h5py 读取出来的 shape 通常是 (N, H, W, C) 或 (N, C, H, W)
        # 对于 MATLAB v7.3 文件，h5py 会转置维度。
        # 原始 MATLAB: images (H, W, 3, N) -> h5py: (N, 3, W, H)
        rgb_images = f['images']  
        seg_labels = f['labels']  # (N, W, H)
        depths = f['depths']      # (N, W, H)
        
        num_samples = rgb_images.shape[0]
        print(f"样本数量: {num_samples}")

        # --- 修正类别名提取方式 ---
        print("正在提取类别名称...")
        names_ref = f['names']
        class_names = []
        
        # 处理 names 维度可能是 (1, N) 或 (N, 1)
        if names_ref.shape[0] == 1:
            refs = names_ref[0] # (N,)
        else:
            refs = names_ref[:, 0] # (N,)

        for ref in refs:
            # 通过引用获取实际的字符串数据
            obj = f[ref]
            arr = obj[()]
            # 解析字符串 (MATLAB 字符串通常存储为 uint16, utf-16le)
            try:
                name = arr.tobytes().decode('utf-16le').strip('\x00')
            except Exception:
                # 如果解码失败，尝试直接转换
                name = str(arr)
            class_names.append(name)

        # --- 构建 class_map ---
        # 这里的 seg_labels 是 1-based 的索引，对应 class_names 中的位置
        # 我们直接生成映射，跳过读取可能出错的 namesToIds
        class_map = {i+1: name for i, name in enumerate(class_names)}
        
        # 保存为json
        with open(os.path.join(save_root, 'class_map.json'), 'w', encoding='utf-8') as jf:
            json.dump(class_map, jf, ensure_ascii=False, indent=2)

        print("开始导出图像...")
        for idx in range(num_samples):
            if idx % 100 == 0:
                print(f"处理进度: {idx}/{num_samples}")

            # --- 处理 RGB ---
            # 读取 (3, W, H)
            rgb = np.array(rgb_images[idx])  
            # 转置为 (W, H, 3)
            if rgb.shape[0] == 3:
                rgb = np.transpose(rgb, (1, 2, 0))
            # 旋转 -90度 (顺时针90度) 以矫正方向
            rgb = np.rot90(rgb, k=-1)
            
            rgb_img = Image.fromarray(rgb.astype(np.uint8))
            rgb_img.save(os.path.join(save_root, "rgb", f"{idx:04d}.png"))

            # --- 处理 Label ---
            # 读取 (W, H)
            label = np.array(seg_labels[idx])  
            label = np.rot90(label, k=-1)
            label_img = Image.fromarray(label.astype(np.uint8))
            label_img.save(os.path.join(save_root, "seg_label", f"{idx:04d}.png"))

            # --- 处理 Depth ---
            # 读取 (W, H)
            depth = np.array(depths[idx])  
            depth = np.rot90(depth, k=-1)
            
            # 归一化到0-255，便于可视化
            max_val = depth.max()
            min_val = depth.min()
            if max_val > min_val:
                depth_norm = ((depth - min_val) * 255 / (max_val - min_val)).astype(np.uint8)
            else:
                depth_norm = np.zeros_like(depth, dtype=np.uint8)
                
            depth_img = Image.fromarray(depth_norm)
            depth_img.save(os.path.join(save_root, "depth", f"{idx:04d}.png"))

    # 4. 划分训练/验证集（NYU无官方划分，按8:2拆分1449张）
    print("正在划分数据集...")
    np.random.seed(42)  # 固定随机种子
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    split_idx = int(num_samples * 0.8)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    # 保存划分文件
    np.save(os.path.join(save_root, "train_indices.npy"), train_indices)
    np.save(os.path.join(save_root, "val_indices.npy"), val_indices)
    print("全部完成！")

except Exception as e:
    print(f"\n发生错误: {e}")
    import traceback
    traceback.print_exc()