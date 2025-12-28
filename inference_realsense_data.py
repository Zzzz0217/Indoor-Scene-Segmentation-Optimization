import os
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from resnet_unet import ResNetUNet
import json

def colorize_mask(mask, num_classes):
    # 生成随机颜色表 (固定种子以保持颜色一致)
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
    data_root = 'realsense_data'
    rgb_dir = os.path.join(data_root, 'rgb')
    depth_raw_dir = os.path.join(data_root, 'depth_raw') # 优先使用 16-bit raw
    depth_norm_dir = os.path.join(data_root, 'depth')    # 备用 8-bit
    
    output_dir = os.path.join(data_root, 'inference_results')
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = 'realsense_data/best_unet_rgbd.pth'
    num_classes = 41
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    print(f"Loading model from {model_path}...")
    model = ResNetUNet(n_classes=num_classes, n_channels=4)
    if os.path.exists(model_path):
        # 尝试加载完整 checkpoint 或 仅权重
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"Error: Model file {model_path} not found.")
        return
    
    model.to(device)
    model.eval()

    # 获取所有 RGB 图片
    if not os.path.exists(rgb_dir):
        print(f"Error: RGB directory {rgb_dir} not found.")
        return
        
    image_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png') or f.endswith('.jpg')])
    print(f"Found {len(image_files)} images.")

    for img_name in image_files:
        print(f"Processing {img_name}...")
        
        # 1. 读取 RGB
        rgb_path = os.path.join(rgb_dir, img_name)
        rgb_img = Image.open(rgb_path).convert('RGB')
        
        # 2. 读取 Depth
        # 尝试读取 raw depth (16-bit)
        base_name = os.path.splitext(img_name)[0]
        depth_raw_path = os.path.join(depth_raw_dir, f"{base_name}_raw.png")
        depth_norm_path = os.path.join(depth_norm_dir, img_name)
        
        if os.path.exists(depth_raw_path):
            # 读取 16-bit 深度图
            depth_img_raw = cv2.imread(depth_raw_path, cv2.IMREAD_UNCHANGED)
            
            # 预处理: 去噪
            depth_img_raw = cv2.medianBlur(depth_img_raw, 5)
            
            # --- 改进: 鲁棒的动态归一化 (Robust Dynamic Normalization) ---
            # 1. 排除 0 值 (无效深度)
            valid_mask = depth_img_raw > 0
            
            if valid_mask.any():
                # 2. 使用百分位数代替绝对 min/max，抵抗噪声
                # RealSense 常有极远处的飞点噪声，导致 max 虚高，压缩了有效深度范围
                # 取 1% 和 99% 分位点，忽略极值噪声
                min_val = np.percentile(depth_img_raw[valid_mask], 1)
                max_val = np.percentile(depth_img_raw[valid_mask], 99)
                
                # 截断异常值
                depth_clamped = np.clip(depth_img_raw, min_val, max_val)
                
                # 归一化
                if max_val > min_val:
                    depth_norm = ((depth_clamped - min_val) * 255.0 / (max_val - min_val)).astype(np.uint8)
                else:
                    depth_norm = np.zeros_like(depth_img_raw, dtype=np.uint8)
                    
                # (可选) 将无效区域设为 0 或保持归一化后的值
                # 这里保持归一化后的值，因为 0 在训练中通常也是有效值
            else:
                depth_norm = np.zeros_like(depth_img_raw, dtype=np.uint8)
            
            depth_pil = Image.fromarray(depth_norm)
            
        elif os.path.exists(depth_norm_path):
            print(f"  Warning: Raw depth not found, using normalized depth {depth_norm_path}")
            depth_pil = Image.open(depth_norm_path).convert('L')
        else:
            print(f"  Error: No depth file found for {img_name}, skipping.")
            continue

        # 3. 预处理 (Transforms)
        rgb_tensor = transforms.ToTensor()(rgb_img)
        rgb_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(rgb_tensor)
        
        depth_tensor = transforms.ToTensor()(depth_pil)
        depth_tensor = transforms.Normalize(mean=[0.5], std=[0.5])(depth_tensor)
        
        # Concat -> (1, 4, H, W)
        input_tensor = torch.cat([rgb_tensor, depth_tensor], dim=0).unsqueeze(0).to(device)

        # 4. 推理 (Multi-Scale TTA)
        # 使用多尺度 + 翻转增强来提高鲁棒性
        scales = [0.8, 1.0, 1.2] 
        final_output = 0
        
        with torch.no_grad():
            b, c, h, w = input_tensor.shape
            
            for scale in scales:
                if scale == 1.0:
                    scaled_input = input_tensor
                else:
                    new_h, new_w = int(h * scale), int(w * scale)
                    scaled_input = torch.nn.functional.interpolate(input_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
                
                # Forward
                output = model(scaled_input)
                
                # Flip TTA
                input_flip = torch.flip(scaled_input, [3])
                output_flip = model(input_flip)
                output_flip = torch.flip(output_flip, [3])
                output = (output + output_flip) / 2.0
                
                # Resize back
                if scale != 1.0:
                    output = torch.nn.functional.interpolate(output, size=(h, w), mode='bilinear', align_corners=False)
                
                final_output += output
            
            # Average
            final_output /= len(scales)
            pred = torch.argmax(final_output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            # --- 后处理: 去除孤立噪点 (Post-processing) ---
            # 现象: 同一个物体内部出现孤立的其他类别碎块 (椒盐噪声)
            # 解决方法: 对预测结果(Mask)进行中值滤波
            # 中值滤波可以有效平滑分类结果，去除孤立像素，同时保留边缘
            pred = cv2.medianBlur(pred, 7) 
            
            # 进一步平滑: 形态学开闭运算
            # 开运算: 去除背景中的噪点 (断开细小连接)
            # 闭运算: 填充物体内的孔洞 (连接断裂部分)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            pred = cv2.morphologyEx(pred, cv2.MORPH_OPEN, kernel)
            pred = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, kernel)

        # 5. 可视化保存
        pred_color = colorize_mask(pred, num_classes)
        
        # 转换 RGB 图片用于显示
        rgb_cv = cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)
        pred_cv = cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR)
        
        # 叠加
        alpha = 0.5
        overlay = cv2.addWeighted(rgb_cv, 1 - alpha, pred_cv, alpha, 0)
        
        # 拼接: RGB | Prediction | Overlay
        combined = np.hstack((rgb_cv, pred_cv, overlay))
        
        save_path = os.path.join(output_dir, f"pred_{img_name}")
        cv2.imwrite(save_path, combined)
        print(f"  Saved result to {save_path}")

    print("Done!")

if __name__ == "__main__":
    main()
