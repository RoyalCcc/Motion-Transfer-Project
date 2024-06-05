import torch
import os
import cv2
import numpy as np
import math

folder_name = 'Sup-2D-Back-total'

img_root = '../Motion-Datasets/Sup-Back'
img_dst_root = '../Motion-Datasets/Sup-Back'

# img_root = os.path.join('../Motion-Datasets', folder_name)
# img_dst_root = os.path.join('../Motion-Datasets', folder_name)

oriImg_dir = os.path.join(img_root, folder_name)
oriImg_names = sorted(os.listdir(oriImg_dir))
oriImg_names = [name for name in oriImg_names if name != '.DS_Store']

poseImg_dir = os.path.join(img_root, folder_name + '_poseImg')
poseImg_names = sorted(os.listdir(poseImg_dir))
poseImg_names = [name for name in poseImg_names if name != '.DS_Store']

bbx_dir = os.path.join(img_root, folder_name + '_bbx')
bbx_names = sorted(os.listdir(bbx_dir))
bbx_names = [name for name in bbx_names if name != '.DS_Store']

preprocess = 'fixsize' # 'fixratio' or 'fixsize'ls


cropedOriImg_dir = os.path.join(img_dst_root,
                                folder_name + '_' + preprocess +'_cropedOriImg')
if not os.path.isdir(cropedOriImg_dir):
    os.makedirs(cropedOriImg_dir)

cropedPoseImg_dir = os.path.join(img_dst_root,
                                 folder_name + '_' + preprocess + '_cropedPoseImg')
if not os.path.isdir(cropedPoseImg_dir):
    os.makedirs(cropedPoseImg_dir)

if preprocess == 'fixsize':
    # if phase == 'train':
    #     fix_w = math.ceil(subjects_w[subject_idx - 1] / 8.0) * 8
    #     fix_h = math.ceil(subjects_h[subject_idx - 1] / 8.0) * 8
    # else:
    #     fix_w = math.ceil(subjects_w_test[subject_idx - 1] / 8.0) * 8
    #     fix_h = math.ceil(subjects_h_test[subject_idx - 1] / 8.0) * 8

    fix_w = 512
    fix_h = 512

for idx, poseImg_name in enumerate(poseImg_names):
    poseImg_path = os.path.join(poseImg_dir, poseImg_name)
    oriImg_path = os.path.join(oriImg_dir, poseImg_name)
    bbx_path = os.path.join(bbx_dir, poseImg_name.split('.')[0] + '.npy')

    poseImg = cv2.imread(poseImg_path)  # B,G,R order
    oriImg = cv2.imread(oriImg_path)  # B,G,R order
    ori_h = poseImg.shape[0]
    ori_w = poseImg.shape[1]

    bbx_info = np.load(bbx_path)

    w_ratio = bbx_info[0]
    w_len = w_ratio * ori_w

    h_ratio = bbx_info[3]
    h_len = h_ratio * ori_h

    if preprocess == 'fixsize':
        w_offset = ((fix_w - w_len) / 2)
        h_offset = ((fix_h - h_len) / 2)
    else:
        len_max = min(max(w_len, h_len) + 50, 512)
        w_offset = ((len_max - w_len) / 2)
        h_offset = ((len_max - h_len) / 2)

    x_min_ratio = bbx_info[1]
    x_min_coor = x_min_ratio * ori_w
    x_min_coor = np.maximum(0, x_min_coor - w_offset)

    x_max_ratio = bbx_info[2]
    x_max_coor = x_max_ratio * ori_w
    x_max_coor = np.minimum(ori_w, x_max_coor + w_offset)

    if x_min_coor == 0:
        x_max_coor = x_min_coor + fix_w
    if x_max_coor == ori_w:
        x_min_coor = x_max_coor - fix_w

    y_min_ratio = bbx_info[4]
    y_min_coor = y_min_ratio * ori_h
    y_min_coor = np.maximum(0, y_min_coor - h_offset)

    y_max_ratio = bbx_info[5]
    y_max_coor = y_max_ratio * ori_h
    y_max_coor = np.minimum(ori_h, y_max_coor + h_offset)

    if y_min_coor == 0:
        y_max_coor = y_min_coor + fix_h
    if y_max_coor == ori_h:
        y_min_coor = y_max_coor - fix_h

    x_min_coor = int(x_min_coor)
    x_max_coor = int(x_max_coor)
    y_min_coor = int(y_min_coor)
    y_max_coor = int(y_max_coor)

    cropedOriImg = oriImg[y_min_coor:y_max_coor, x_min_coor:x_max_coor]
    cropedPoseImg = poseImg[y_min_coor:y_max_coor, x_min_coor:x_max_coor]

    # if preprocess != 'fixsize':
    #     cropedOriImg = cv2.resize(cropedOriImg, (fix_w, fix_h))
    #     cropedPoseImg = cv2.resize(cropedPoseImg, (fix_w, fix_h))

    cropedOriImg_path = os.path.join(cropedOriImg_dir, poseImg_name)
    print(str(idx) + '. Croped Original Imag: ' + poseImg_name + ' has been finished!!')
    cv2.imwrite(cropedOriImg_path, cropedOriImg)

    cropedPoseImg_path = os.path.join(cropedPoseImg_dir, poseImg_name)
    print(str(idx) + '. Croped Pose Imag: ' + poseImg_name + ' has been finished!!')
    cv2.imwrite(cropedPoseImg_path, cropedPoseImg)



