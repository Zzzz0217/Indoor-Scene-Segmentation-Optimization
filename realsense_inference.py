import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from resnet_unet import ResNetUNet
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 设置类别数量
    # 注意：训练时使用了 Top-40 映射，所以类别数固定为 41 (0-40)
    # 原始 class_map.json 包含 894 类，直接读取会导致模型结构不匹配 (895 vs 41)
    num_classes = 41 

    # 加载模型
    model = ResNetUNet(n_classes=num_classes, n_channels=4)
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

    print("开始实时推理，按 'q' 退出 (如果无窗口显示，请按 Ctrl+C 结束)")

    # 视频保存设置
    out = None
    save_video = True
    output_filename = 'inference_output.avi'
    
    # 深度归一化模式
    # 'dynamic': 使用当前帧的 min/max (训练时的做法，但易受场景影响)
    # 'fixed': 使用固定范围 (0-10m)，保持绝对尺度一致性 (推荐用于泛化)
    depth_mode = 'fixed' 
    fixed_max_depth = 10000 # 10 meters in mm (RealSense default scale is 1mm)

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
            # 0. 深度去噪 (简单的孔洞填充或中值滤波)
            # RealSense 深度图常有噪声，中值滤波可以平滑
            depth_image = cv2.medianBlur(depth_image, 5)

            # 1. RGB: BGR -> RGB, PIL Image
            rgb_pil = Image.fromarray(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
            
            # 2. Depth: 归一化 -> PIL Image (8-bit)
            if depth_mode == 'dynamic':
                min_val = depth_image.min()
                max_val = depth_image.max()
                if max_val > min_val:
                    depth_norm = ((depth_image - min_val) * 255 / (max_val - min_val)).astype(np.uint8)
                else:
                    depth_norm = np.zeros_like(depth_image, dtype=np.uint8)
            else: # fixed
                # 截断超过 max_depth 的值
                depth_clamped = np.clip(depth_image, 0, fixed_max_depth)
                depth_norm = (depth_clamped * 255 / fixed_max_depth).astype(np.uint8)

            depth_pil = Image.fromarray(depth_norm)

            # 3. ToTensor
            rgb_tensor = transforms.ToTensor()(rgb_pil)
            rgb_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(rgb_tensor)
            depth_tensor = transforms.ToTensor()(depth_pil)
            
            # 4. Concat -> (1, 4, H, W)
            input_tensor = torch.cat([rgb_tensor, depth_tensor], dim=0).unsqueeze(0).to(device)

            # --- 推理 (含 TTA) ---
            with torch.no_grad():
                # 原始预测
                output = model(input_tensor)
                
                # TTA: 水平翻转
                input_flip = torch.flip(input_tensor, [3])
                output_flip = model(input_flip)
                output_flip = torch.flip(output_flip, [3])
                
                # 平均
                output = (output + output_flip) / 2.0
                
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
            
            # 初始化视频写入器 (如果需要)
            if save_video and out is None:
                h, w = display_img.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(output_filename, fourcc, 20.0, (w, h))
                print(f"正在保存视频到 {output_filename}")

            if out is not None:
                out.write(display_img)

            try:
                cv2.imshow('RealSense Segmentation', display_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except cv2.error:
                # 如果 GUI 不可用，仅打印一次警告并继续保存视频
                pass

    except KeyboardInterrupt:
        print("用户中断")
    finally:
        pipeline.stop()
        if out is not None:
            out.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass

if __name__ == "__main__":
    main()
