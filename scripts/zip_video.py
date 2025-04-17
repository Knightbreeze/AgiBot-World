import cv2
import os

def downsample_video_with_opencv(input_path, output_path, target_fps=10):
    # 打开输入视频
    cap = cv2.VideoCapture(input_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 计算抽帧间隔
    interval = round(original_fps / target_fps)
    if interval < 1:
        interval = 1
    
    # 创建输出视频写入器
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 每隔interval帧保存一帧
        if frame_count % interval == 0:
            out.write(frame)
            saved_count += 1
            
        frame_count += 1
    
    # 释放资源
    cap.release()
    out.release()
# 使用示例
input_path = "/fs-computility/efm/shared/datasets/agibot-world/lerobotV2_AgiBotWorld-Alpha_test/video/1.mp4"
output_path = "/fs-computility/efm/shared/datasets/agibot-world/lerobotV2_AgiBotWorld-Alpha_test/video/5_opencv.mp4"
downsample_video_with_opencv(input_path, output_path)
