import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from unet_model import UNetRGBD
import json
import os

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
    model_path = 'best_unet_rgbd.pth'
    class_map_path = 'nyu_depthv2_seg_dataset/class_map.json'
    
    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")

    # 加载类别信息
    if os.path.exists(class_map_path):
        with open(class_map_path, 'r') as f:
            class_map = json.load(f)
        ids = [int(k) for k in class_map.keys()]
        num_classes = max(ids) + 1 if ids else 0
    else:
        print("未找到 class_map.json，默认使用 40 类")
        num_classes = 40 # 默认值

    # 加载模型
    model = UNetRGBD(n_channels=4, n_classes=num_classes)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("模型加载成功")
    else:
        print(f"警告: 未找到模型文件 {model_path}，将使用随机初始化模型进行演示")
    
    model.to(device)
    model.eval()

    # RealSense 初始化
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    profile = pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)

    print("开始实时推理，按 'q' 退出")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                continue

            # 获取图像数据
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data()) # BGR

            # --- 预处理 ---
            # 1. RGB: BGR -> RGB, PIL Image
            rgb_pil = Image.fromarray(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
            
            # 2. Depth: 归一化 -> PIL Image (8-bit)
            min_val = depth_image.min()
            max_val = depth_image.max()
            if max_val > min_val:
                depth_norm = ((depth_image - min_val) * 255 / (max_val - min_val)).astype(np.uint8)
            else:
                depth_norm = np.zeros_like(depth_image, dtype=np.uint8)
            depth_pil = Image.fromarray(depth_norm)

            # 3. ToTensor
            rgb_tensor = transforms.ToTensor()(rgb_pil)
            depth_tensor = transforms.ToTensor()(depth_pil)
            
            # 4. Concat -> (1, 4, H, W)
            input_tensor = torch.cat([rgb_tensor, depth_tensor], dim=0).unsqueeze(0).to(device)

            # --- 推理 ---
            with torch.no_grad():
                output = model(input_tensor)
                pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

            # --- 可视化 ---
            # 彩色化预测结果
            pred_color = colorize_mask(pred, num_classes)
            # BGR for OpenCV
            pred_color_bgr = cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR)

            # 混合显示 (RGB + Prediction)
            alpha = 0.5
            overlay = cv2.addWeighted(color_image, 1 - alpha, pred_color_bgr, alpha, 0)

            # 拼接显示: [RGB, Overlay]
            display_img = np.hstack((color_image, overlay))
            
            cv2.imshow('RealSense Segmentation', display_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
