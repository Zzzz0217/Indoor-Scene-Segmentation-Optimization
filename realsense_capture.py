import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time

def main():
    # 配置保存路径
    save_dir = "realsense_data"
    os.makedirs(os.path.join(save_dir, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "depth"), exist_ok=True)

    # 配置 RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 启用 RGB 和 深度 流
    # 640x480 是比较通用的分辨率，与 NYU 数据集接近
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # 启动 pipeline
    profile = pipeline.start(config)

    # 获取深度传感器的深度标度 (Depth Scale)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)

    # 创建对齐对象 (深度对齐到颜色)
    align_to = rs.stream.color
    align = rs.align(align_to)

    print("按 's' 保存当前帧，按 'q' 退出")
    
    idx = 0
    try:
        while True:
            # 等待一帧数据
            frames = pipeline.wait_for_frames()
            
            # 对齐深度帧到颜色帧
            aligned_frames = align.process(frames)
            
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                continue

            # 转换为 numpy 数组
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # 深度图可视化 (归一化到 0-255)
            # 注意：保存时我们可能想要保存原始深度数据，或者与训练集一致的格式
            # 这里为了可视化方便，显示伪彩色
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # 显示图像
            images = np.hstack((color_image, depth_colormap))
            cv2.imshow('RealSense', images)

            key = cv2.waitKey(1)
            
            # 按 's' 保存
            if key & 0xFF == ord('s'):
                # 保存 RGB
                cv2.imwrite(os.path.join(save_dir, "rgb", f"{idx:04d}.png"), color_image)
                
                # 保存深度
                # 训练集使用的是归一化后的 8-bit png，这里为了保持一致，也做归一化
                # 但实际应用中，保留原始深度信息 (16-bit) 可能更好
                # 这里我们保存两份：一份 16-bit 原始数据 (png)，一份 8-bit 归一化数据 (用于匹配当前训练流程)
                
                # 16-bit raw
                cv2.imwrite(os.path.join(save_dir, "depth", f"{idx:04d}_raw.png"), depth_image)
                
                # 8-bit normalized (简单线性归一化，类似 export.py)
                # 注意：export.py 是基于单张图的最大最小值归一化，这在推理时可能不稳定
                # 建议：如果训练时用了 min-max 归一化，推理时也得用。
                # 这里简单复刻 export.py 的逻辑
                min_val = depth_image.min()
                max_val = depth_image.max()
                if max_val > min_val:
                    depth_norm = ((depth_image - min_val) * 255 / (max_val - min_val)).astype(np.uint8)
                else:
                    depth_norm = np.zeros_like(depth_image, dtype=np.uint8)
                
                cv2.imwrite(os.path.join(save_dir, "depth", f"{idx:04d}.png"), depth_norm)
                
                print(f"Saved frame {idx}")
                idx += 1

            # 按 'q' 退出
            if key & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
