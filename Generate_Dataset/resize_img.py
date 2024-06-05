import torch
import os
import cv2
import numpy as np
import math
import time

phases = ['train', 'test']

subs = ['subject1', 'subject2', 'subject3', 'subject4', 'subject5']
# subs = ['subject5']

img_root = os.path.join('Motion-Datasets', 'EDN-subs')

prefix = 'fixmid_'
cur_suffix = '_1000'
new_suffix = '_256'

fix_w = 256
fix_h = 256

count = 0

for phase in phases:
    oriImg_dir = os.path.join(img_root, prefix + phase + cur_suffix)

    new_img_256_dir = os.path.join(img_root, prefix + phase + cur_suffix + new_suffix)
    if not os.path.isdir(new_img_256_dir):
        os.makedirs(new_img_256_dir)

    poseImg_dir = os.path.join(img_root, prefix + phase + '_poseImg' + cur_suffix)

    new_poseImg_256_dir = os.path.join(img_root, prefix + phase + '_poseImg' + cur_suffix + new_suffix)
    if not os.path.isdir(new_poseImg_256_dir):
        os.makedirs(new_poseImg_256_dir)

    for sub in subs:
        sub_oriImg_dir = os.path.join(oriImg_dir, sub)
        oriImg_names = sorted(os.listdir(sub_oriImg_dir))
        oriImg_names = [name for name in oriImg_names if name != '.DS_Store']

        new_sub_img_256_dir = os.path.join(new_img_256_dir, sub)
        if not os.path.isdir(new_sub_img_256_dir):
            os.makedirs(new_sub_img_256_dir)

        sub_poseImg_dir = os.path.join(poseImg_dir, sub)
        poseImg_names = sorted(os.listdir(sub_poseImg_dir))
        poseImg_names = [name for name in poseImg_names if name != '.DS_Store']

        new_sub_poseImg_256_dir = os.path.join(new_poseImg_256_dir, sub)
        if not os.path.isdir(new_sub_poseImg_256_dir):
            os.makedirs(new_sub_poseImg_256_dir)

        for idx, poseImg_name in enumerate(poseImg_names):
            start_time = time.time()

            oriImg_path = os.path.join(sub_oriImg_dir, poseImg_name)
            poseImg_path = os.path.join(sub_poseImg_dir, poseImg_name)

            oriImg = cv2.imread(oriImg_path)  # B,G,R order
            poseImg = cv2.imread(poseImg_path)  # B,G,R order
            ori_h = poseImg.shape[0]
            ori_w = poseImg.shape[1]

            resized_OriImg = cv2.resize(oriImg, (fix_w, fix_h))
            resized_PoseImg = cv2.resize(poseImg, (fix_w, fix_h))

            count += 1
            resized_OriImg_path = os.path.join(new_sub_img_256_dir, poseImg_name)
            print('%d: %d of %s: Resized Original Imag: %s has been finished!!' % (count, idx, sub, resized_OriImg_path))
            cv2.imwrite(resized_OriImg_path, resized_OriImg)

            resized_PoseImg_path = os.path.join(new_sub_poseImg_256_dir, poseImg_name)
            print('%d: %d of %s: Resized Pose Imag: %s has been finished!!' % (count, idx, sub, resized_PoseImg_path))
            cv2.imwrite(resized_PoseImg_path, resized_PoseImg)

            end_time = time.time()
            print('%d: %d of %s: Time consumption: %f s' % (count, idx, sub, float(end_time - start_time)))
            

