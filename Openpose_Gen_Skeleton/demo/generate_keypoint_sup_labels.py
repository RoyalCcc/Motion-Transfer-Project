import os
import re
import sys
import cv2
import math
import time
import scipy
import argparse
import matplotlib
import numpy as np
import pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import gaussian_filter, maximum_filter

from lib.network.rtpose_vgg import get_model
from lib.network import im_transform
from evaluate.coco_eval import get_outputs, handle_paf_and_heat
from lib.utils.common import Human, BodyPart, CocoPart, CocoColors, CocoPairsRender, draw_pose
from lib.utils.paf_to_pose import paf_to_pose_cpp
from lib.config import cfg, update_config
import os
import shutil

print(os.getcwd())
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', help='experiment configure file name',
                    default='./experiments/vgg19_368x368_sgd.yaml', type=str)
parser.add_argument('--weight', type=str,
                    default='./lib/network/weight/pose_model.pth')
parser.add_argument('opts',
                    help="Modify config options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER)
args = parser.parse_args()

# update config file
update_config(cfg, args)

weight_name = args.weight

model = get_model('vgg19')
model.load_state_dict(torch.load(weight_name))
if torch.cuda.is_available():  # enable cuda
    print("Using CUDA: " + str(torch.cuda.is_available()))
    model = torch.nn.DataParallel(model).cuda()

model.float()
model.eval()

root = '../Motion-Datasets-mixamo/Mixamo-2D-test'

sup_root = root + '_img'
bbx_root = root + '_bbx'
pose_root = root + '_pose'
fail_root = root + '_fail'

vid_names = sorted(os.listdir(sup_root))
vid_names = [name for name in vid_names if name != '.DS_Store']

for vid_name in vid_names:
    pose_dir = os.path.join(pose_root, vid_name)
    if not os.path.isdir(pose_dir):
        os.makedirs(pose_dir)

    bbx_dir = os.path.join(bbx_root, vid_name)
    if not os.path.isdir(bbx_dir):
        os.makedirs(bbx_dir)

    failed_dir = os.path.join(fail_root, vid_name)
    if not os.path.isdir(failed_dir):
        os.makedirs(failed_dir)

    img_dir = os.path.join(sup_root, vid_name)
    img_names = sorted(os.listdir(img_dir))
    img_names = [name for name in img_names if name != '.DS_Store']

    result_w = 0.0
    result_x_min = 1.0
    result_x_max = 0.0

    result_h = 0.0
    result_y_min = 1.0
    result_y_max = 0.0

    removed_file_count = 0

    for idx, img_name in enumerate(img_names):
        img_path = os.path.join(img_dir, img_name)

        oriImg = cv2.imread(img_path)  # B,G,R order
        shape_dst = np.min(oriImg.shape[0:2])
        ori_h = oriImg.shape[0]
        ori_w = oriImg.shape[1]

        # Get results of original image

        with torch.no_grad():
            paf, heatmap, im_scale = get_outputs(oriImg, model, 'rtpose')

        # print(im_scale)
        human, bbx_info = paf_to_pose_cpp(heatmap, paf, cfg)

        if human is None:
            failed_path = os.path.join(failed_dir, img_name)
            #os.remove(img_path)
            shutil.move(img_path, failed_path)
            print(vid_name + ': ' + str(idx) + ': ' + img_name + ' has been removed!!')
            removed_file_count += 1
            continue

        w = bbx_info[0]
        h = bbx_info[3]

        if w > result_w:
            result_w = w
            result_x_min = bbx_info[1]
            result_x_max = bbx_info[2]

        if h > result_h:
            result_h = h
            result_y_min = bbx_info[4]
            result_y_max = bbx_info[5]

        bbx_path = os.path.join(bbx_dir, img_name.split('.')[0]+'.npy')
        np.save(bbx_path, np.array(bbx_info))

        # out = draw_humans(oriImg, humans)
        out = draw_pose(oriImg, human)

        pose_path = os.path.join(pose_dir, img_name)
        # cv2.imshow('11', out)
        print(vid_name + ': ' + str(idx) + ': ' + img_name + ' has been finished!!')
        cv2.imwrite(pose_path, out)

    print(vid_name + ': ' + str(removed_file_count) + ' have been removed!!')
    print(vid_name + ': ' + ' Max width is :' + str(result_w) + ' and ' + str(result_w * ori_w))
    print(vid_name + ': ' + ' Max height is :' + str(result_h) + ' and ' + str(result_h * ori_h))

    result_bbx_info = np.array([result_w, result_x_min, result_x_max,
                                  result_h, result_y_min, result_y_max])

    # bbx_path = os.path.join(args.img_root, 'bbx.npy')
    # np.save(bbx_path, result_bbx_info)

# x = np.load(width_path)
# print(x)


