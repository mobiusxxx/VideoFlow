import sys

sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from core.utils.misc import process_cfg
from core.utils import flow_viz
from core.Networks import build_network
from core.utils import frame_utils
from core.utils.utils import InputPadder, forward_interpolate
import itertools
import imageio


def prepare_video(cfg, cap, video_fps, rect_x, rect_y, rect_w, rect_h):
    print(f"preparing image...")
    print(f"Input image sequence dir = {cap}")
    # 设置剪切的时间段（单位：秒）
    start_time = 0.8  # 开始时间
    end_time = 1.2  # 结束时间

    # 计算开始和结束的帧数
    start_frame = int(start_time * video_fps)
    end_frame = int(end_time * video_fps)
    # 设置视频帧指针到开始的帧
    k = cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    infer_size = cfg.infer_size
    images = []

    # 读取并写入视频帧
    for frame_num in range(start_frame, end_frame):
        ret, frame = cap.read()
        # img = np.array(frame).astype(np.uint8)
        # if len(img.shape) == 2:
        #     img = np.tile(img[..., None], (1, 1, 3))  # 使用 np.tile 函数将灰度图像的通道复制三次，将其转换为彩色图像。
        # else:
        #     img = img[..., :3]
        # img = cv2.resize(frame, (1280, 720))  # (560, 280)  (1920,1080)  (1280,720)  (600,336)
        img = cv2.resize(frame, infer_size)
        # img = torch.from_numpy(img).permute(2, 0, 1).float()

        # 获取ROI区域
        cropped_img = img[rect_y:rect_y + rect_h, rect_x:rect_x+rect_w]
        # 将裁剪后的图像转换为PyTorch张量
        img = torch.from_numpy(cropped_img).permute(2, 0, 1).float()

        images.append(img)
        if not ret:
            break  # 读取失败或到达视频末尾时退出循环
    # 释放视频捕获和写入对象
    cap.release()

    return torch.stack(images)


def vis_pre(flow_pre, vis_dir):
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    N = flow_pre.shape[0]

    for idx in range(N // 2):
        flow_img = flow_viz.flow_to_image(flow_pre[idx].permute(1, 2, 0).numpy())
        image = Image.fromarray(flow_img)
        image.save('{}/forward_flow_{:04}_to_{:04}.png'.format(vis_dir, idx + 2, idx + 3))

    for idx in range(N // 2, N):
        flow_img = flow_viz.flow_to_image(flow_pre[idx].permute(1, 2, 0).numpy())
        image = Image.fromarray(flow_img)
        image.save('{}/backward_flow_{:04}_to_{:04}.png'.format(vis_dir, idx - N // 2 + 2, idx - N // 2 + 1))


@torch.no_grad()
def MOF_inference(model, cfg):
    roi = np.array([cfg.rois])
    roi = np.ceil(roi).astype(np.int32)
    rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(roi[0])
    model.eval()
    cap = cv2.VideoCapture(cfg.video_dir)
    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    # 获取视频帧率
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    input_images = prepare_video(cfg, cap, video_fps, rect_x, rect_y, rect_w, rect_h)
    input_images = input_images[None].cuda()
    padder = InputPadder(input_images.shape)
    input_images = padder.pad(input_images)
    flow_pre, _ = model(input_images, {})
    flow_pre = padder.unpad(flow_pre[0]).cpu()

    return flow_pre, video_fps


@torch.no_grad()
def BOF_inference(model, cfg):
    model.eval()

    cap = cv2.VideoCapture(cfg.video_dir)
    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    # 获取视频帧率
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    input_images = prepare_video(cfg, cap, video_fps)

    input_images = input_images[None].cuda()
    padder = InputPadder(input_images.shape)
    input_images = padder.pad(input_images)
    flow_pre, _ = model(input_images, {})
    flow_pre = padder.unpad(flow_pre[0]).cpu()

    return flow_pre


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='MOF')
    parser.add_argument('--video_dir', default='default')
    parser.add_argument('--vis_dir', default='default')

    args = parser.parse_args()

    if args.mode == 'MOF':
        from configs.multiframes_sintel_submission import get_cfg
    elif args.mode == 'BOF':
        from configs.sintel_submission import get_cfg

    cfg = get_cfg()
    cfg.update(vars(args))  # vars(args) 将命名空间（namespace）或参数对象 args 转换为字典,将命令行参数传递给配置对象，以便在程序中使用这些参数。

    model = torch.nn.DataParallel(build_network(cfg))
    model.load_state_dict(torch.load(cfg.model))

    model.cuda()
    model.eval()

    print(cfg.model)
    print("Parameter Count: %d" % count_parameters(model))

    with torch.no_grad():
        if args.mode == 'MOF':
            from configs.multiframes_sintel_submission import get_cfg

            flow_pre, video_fps = MOF_inference(model.module, cfg)
        elif args.mode == 'BOF':
            from configs.sintel_submission import get_cfg

            flow_pre = BOF_inference(model.module, cfg)
    N = flow_pre.shape[0]
    forward_flow_pre = flow_pre[0:(N // 2), :, :, :].permute(0, 2, 3, 1).numpy()
    backward_flow_pre = flow_pre[(N // 2):N, :, :, :].permute(0, 2, 3, 1).numpy()

    Calibrationdy = 0.001066436233332257  # 0.0010581530435579412
    Calibrationdx = 0.0009398727355172452   # 0.0009491174159469241
    for_dis_x = forward_flow_pre[:, :, :, 0] * Calibrationdy * video_fps
    for_dis_y = forward_flow_pre[:, :, :, 1] * Calibrationdx * video_fps
    # for_dist = np.sqrt(for_dis_x ** 2 + for_dis_y ** 2)
    # forward_ave_dist = np.mean(for_dist, axis=0)
    # formax = np.max(forward_ave_dist)
    # print("formax: ", formax)
    # formin = np.min(forward_ave_dist)
    # print("formin: ", formin)
    # forward_all_ave_dist = np.mean(forward_ave_dist)
    # print("forward_all_ave_dist: ", forward_all_ave_dist)
    # forward_ave_mid_dist1 = np.mean(for_dist[:, 60:660, 50:100], axis=0)
    # forward_ave_mid_dist = np.mean(forward_ave_mid_dist1)
    # print("forward_ave_mid_dist", forward_ave_mid_dist)

    # line_velocity = forward_ave_dist[200, :]
    # # 画图
    # plt.plot(line_velocity, label='Pixel Displacement along Line')
    # plt.xlabel('Pixel Position')
    # plt.ylabel('Pixel Displacement')
    # plt.title('Pixel Displacement along a Line')
    # plt.legend()
    # # plt.savefig('{}/pixel_displacement_line_plot_{}.png'.format(cfg.vis_dir,cfg.infer_size))
    # plt.show()
    # # plt.close()
    vis_pre(flow_pre, cfg.vis_dir)
    print("finish")







