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
# from evaluate.coco_eval import get_outputs, handle_paf_and_heat
# from lib.utils.common import Human, BodyPart, CocoPart, CocoColors, CocoPairsRender, draw_humans
# from lib.utils.paf_to_pose import paf_to_pose_cpp
from lib.config import cfg, update_config

from lib.datasets.preprocessing import (inception_preprocess,
                                              rtpose_preprocess,
                                              ssd_preprocess, vgg_preprocess)


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', help='experiment configure file name',
                    default='../experiments/vgg19_368x368_sgd.yaml', type=str)
parser.add_argument('--weight', type=str,
                    default='../demo/pose_model.pth')
parser.add_argument('opts',
                    help="Modify config options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER)
args = parser.parse_args()

# update config file
update_config(cfg, args)

        
weight_name = '../demo/pose_model.pth'

model = get_model('vgg19')     
model.load_state_dict(torch.load(weight_name))
# model = torch.nn.DataParallel(model).cuda()
model.float()
model.eval()

test_image = '../readme/ski.jpg'
oriImg = cv2.imread(test_image) # B,G,R order
shape_dst = np.min(oriImg.shape[0:2])

# Get results of original image

def get_outputs1(img, model, preprocess):
    """Computes the averaged heatmap and paf for the given image
    :param multiplier:
    :param origImg: numpy array, the image being processed
    :param model: pytorch model
    :returns: numpy arrays, the averaged paf and heatmap
    """
    inp_size = cfg.DATASET.IMAGE_SIZE

    # padding
    im_croped, im_scale, real_shape = im_transform.crop_with_factor(
        img, inp_size, factor=cfg.MODEL.DOWNSAMPLE, is_ceil=True)

    if preprocess == 'rtpose':
        im_data = rtpose_preprocess(im_croped)

    batch_images= np.expand_dims(im_data, 0)

    # several scales as a batch
    # batch_var = torch.from_numpy(batch_images).cuda().float()
    batch_var = torch.from_numpy(batch_images).float()
    predicted_outputs, _ = model(batch_var)
    output1, output2 = predicted_outputs[-2], predicted_outputs[-1]
    heatmap = output2.cpu().data.numpy().transpose(0, 2, 3, 1)[0]
    paf = output1.cpu().data.numpy().transpose(0, 2, 3, 1)[0]



    return paf, heatmap, im_scale

with torch.no_grad():
    paf, heatmap, im_scale = get_outputs1(oriImg, model,  'rtpose')
          
print(im_scale)
# humans = paf_to_pose_cpp(heatmap, paf, cfg)
        
# out = draw_humans(oriImg, humans)
# cv2.imwrite('result.png',out)

