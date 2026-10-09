import cv2
import os
import numpy as np

# # # 打开视频文件
# video_path = r"D:\code\yuanshiwenjian\vfexfl1_3\vfexfl1_3\model\render2.avi"
# cap = cv2.VideoCapture(video_path)
#
# # 检查视频是否成功打开
# if not cap.isOpened():
#     print("Error: Could not open video.")
#     exit()
#
# # 获取视频帧率
# fps = cap.get(cv2.CAP_PROP_FPS)
#
# # 打印帧率
# print(f"视频帧率: {fps} frames per second")
#
# # 获取视频的总帧数
# total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#
# print(f"Total frames in the video: {total_frames}")
#
# # 获取视频每帧图像的大小
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#
# print(f"Frame size: {width} x {height}")

###########################################
'''
将视频保存成每一帧图片
'''


def save_frames(video_path, output_folder):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    # 循环读取每一帧图片
    frame_number = 0
    while True:
        ret, frame = cap.read()
        # frame = cv2.resize(frame, (1200, 680))
        # 检查是否到达视频末尾
        if not ret:
            break
        # 保存图片
        frame_number += 1
        # output_path = os.path.join(output_folder, f'frame_{frame_number}.png')
        # cv2.imwrite(output_path, frame)
        frame.save('{}/frame_{:04}.png'.format(output_folder, frame_number))

    # 释放资源
    cap.release()


if __name__ == "__main__":
    video_path = r"D:\code\yuanshiwenjian\vfexfl1_3\vfexfl1_3\model\render22.avi"
    output_folder = 'raw_synthetic_river_data_drone01'

    save_frames(video_path, output_folder)

###########################################
# '''
# 截取视频某一段
# '''
# # 设置剪切的时间段（单位：秒）
# start_time = 0  # 开始时间
# end_time = 1   # 结束时间
#
# # 计算开始和结束的帧数
# start_frame = int(start_time * fps)
# end_frame = int(end_time * fps)
#
# # 创建 VideoWriter 对象，用于保存剪切后的视频
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(r"D:\code\yuanshiwenjian\vfexfl1_3\vfexfl1_3\model\render22.avi", fourcc, fps, (1920, 1080))
#
# # 设置视频帧指针到开始的帧
# cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
#
# # 读取并写入视频帧
# for frame_num in range(start_frame, end_frame):
#     ret, frame = cap.read()
#     frame = cv2.resize(frame, (1920, 1080))
#     if not ret:
#         break  # 读取失败或到达视频末尾时退出循环
#     out.write(frame)
# # 释放视频捕获和写入对象
# cap.release()
# out.release()
###########################################
'''
将视频保存成每一帧图片
'''
video_path = r"D:\code\yuanshiwenjian\vfexfl1_3\vfexfl1_3\model\render2.avi"
cap = cv2.VideoCapture(video_path)
# 检查视频是否成功打开
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
# 获取视频帧率
fps = cap.get(cv2.CAP_PROP_FPS)

# 设置剪切的时间段（单位：秒）
start_time = 0  # 开始时间
end_time = 1   # 结束时间

# 计算开始和结束的帧数
start_frame = int(start_time * fps)
end_frame = int(end_time * fps)
# 设置视频帧指针到开始的帧
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# 读取并写入视频帧
for frame_num in range(start_frame, end_frame):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1920, 1080))
    img = np.array(frame).astype(np.uint8)[..., :3]
    img = cv2.resize(img, (560, 280))
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    images.append(img)
    if not ret:
        break  # 读取失败或到达视频末尾时退出循环
    out.write(frame)
# 释放视频捕获和写入对象
cap.release()
out.release()