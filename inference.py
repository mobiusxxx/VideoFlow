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


def prepare_image(seq_dir):
    print(f"preparing image...")
    print(f"Input image sequence dir = {seq_dir}")

    images = []

    image_list = sorted(os.listdir(seq_dir))

    for fn in image_list:
        img = Image.open(os.path.join(seq_dir, fn))
        img = np.array(img).astype(np.uint8)
        if len(img.shape) == 2:
            img = np.tile(img[..., None], (1, 1, 3))  # 使用 np.tile 函数将灰度图像的通道复制三次，将其转换为彩色图像。
        else:
            img = img[..., :3]
        img = cv2.resize(img, (560, 280))
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        images.append(img)
    
    return torch.stack(images)


def vis_pre(flow_pre, vis_dir):

    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    N = flow_pre.shape[0]

    for idx in range(N//2):
        flow_img = flow_viz.flow_to_image(flow_pre[idx].permute(1, 2, 0).numpy())
        image = Image.fromarray(flow_img)
        image.save('{}/flow_{:04}_to_{:04}.png'.format(vis_dir, idx+2, idx+3))
    
    for idx in range(N//2, N):
        flow_img = flow_viz.flow_to_image(flow_pre[idx].permute(1, 2, 0).numpy())
        image = Image.fromarray(flow_img)
        image.save('{}/flow_{:04}_to_{:04}.png'.format(vis_dir, idx-N//2+2, idx-N//2+1))

@torch.no_grad()
def MOF_inference(model, cfg):

    model.eval()

    input_images = prepare_image(cfg.seq_dir)
    input_images = input_images[None].cuda()
    padder = InputPadder(input_images.shape)
    input_images = padder.pad(input_images)
    flow_pre, _ = model(input_images, {})
    flow_pre = padder.unpad(flow_pre[0]).cpu()

    return flow_pre

@torch.no_grad()
def BOF_inference(model, cfg):

    model.eval()

    input_images = prepare_image(cfg.seq_dir)
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
    parser.add_argument('--seq_dir', default='default')
    parser.add_argument('--vis_dir', default='default')
    
    args = parser.parse_args()

    if args.mode == 'MOF':
        from configs.multiframes_sintel_submission import get_cfg
    elif args.mode == 'BOF':
        from configs.sintel_submission import get_cfg

    cfg = get_cfg()
    cfg.update(vars(args))   # vars(args) 将命名空间（namespace）或参数对象 args 转换为字典,将命令行参数传递给配置对象，以便在程序中使用这些参数。

    model = torch.nn.DataParallel(build_network(cfg))
    model.load_state_dict(torch.load(cfg.model))

    model.cuda()
    model.eval()

    print(cfg.model)
    print("Parameter Count: %d" % count_parameters(model))
    
    with torch.no_grad():
        if args.mode == 'MOF':
            from configs.multiframes_sintel_submission import get_cfg
            flow_pre = MOF_inference(model.module, cfg)
        elif args.mode == 'BOF':
            from configs.sintel_submission import get_cfg
            flow_pre = BOF_inference(model.module, cfg)
    N = flow_pre.shape[0]
    forward_flow_pre = flow_pre[0:(N//2), :, :, :].permute(0, 2, 3, 1).numpy()
    backward_flow_pre = flow_pre[(N//2):N, :, :, :].permute(0, 2, 3, 1).numpy()
    forward_ave_store_field = np.mean(forward_flow_pre, axis=0)
    backward_ave_store_field = np.mean(backward_flow_pre, axis=0)
    # torch.save(forward_ave_store_field, 'forward_videoflow_Hurunui_River_800_360.pt')
    # torch.save(forward_ave_store_field, 'backward_videoflow_Hurunui_River_800_360.pt')

    vis_pre(flow_pre, cfg.vis_dir)



