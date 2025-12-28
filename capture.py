import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time

def main():
    # 配置保存路径
    save_root = "realsense_data"
    rgb_dir = os.path.join(save_root, "rgb")
    depth_dir = os.path.join(save_root, "depth")
    depth_raw_dir = os.path.join(save_root, "depth_raw")
    
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(depth_raw_dir, exist_ok=True)

    # 配置 RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 启用 RGB 和 深度 流 (640x480)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # 启动 pipeline
    print("正在启动 RealSense 相机...")
    profile = pipeline.start(config)

    # 创建对齐对象 (深度对齐到颜色)
    align_to = rs.stream.color
    align = rs.align(align_to)

    print("\n=== RealSense Capture Tool (Single Shot) ===")
    print(f"数据将保存到: {save_root}")
    
    # 查找当前最大的索引，避免覆盖
    existing_files = os.listdir(rgb_dir)
    indices = [int(f.split('.')[0]) for f in existing_files if f.endswith('.png') and f.split('.')[0].isdigit()]
    idx = max(indices) + 1 if indices else 0
    print(f"保存索引: {idx:04d}")

    try:
        # 预热相机 (让自动曝光和白平衡稳定)
        print("正在预热相机 (约 2 秒)...")
        for i in range(60): # 30fps * 2s = 60 frames
            frames = pipeline.wait_for_frames()
            # 简单的进度条
            if i % 10 == 0:
                print(".", end='', flush=True)
        print("\n预热完成，正在拍摄...")

        # 拍摄一帧
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            print("错误: 无法获取有效帧")
            return

        # 转换为 numpy 数组
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 执行保存
        print(f"正在保存帧 {idx:04d}...", end='', flush=True)
        
        # 保存 RGB
        cv2.imwrite(os.path.join(rgb_dir, f"{idx:04d}.png"), color_image)
        
        # 保存 16-bit 原始深度 (最重要)
        cv2.imwrite(os.path.join(depth_raw_dir, f"{idx:04d}_raw.png"), depth_image)
        
        # 保存 8-bit 归一化深度 (用于可视化)
        min_val = depth_image.min()
        max_val = depth_image.max()
        if max_val > min_val:
            depth_norm = ((depth_image - min_val) * 255 / (max_val - min_val)).astype(np.uint8)
        else:
            depth_norm = np.zeros_like(depth_image, dtype=np.uint8)
        cv2.imwrite(os.path.join(depth_dir, f"{idx:04d}.png"), depth_norm)
        
        print(" 完成")
        print("拍摄结束，程序退出。")

    finally:
        pipeline.stop()

if __name__ == "__main__":
    main()
