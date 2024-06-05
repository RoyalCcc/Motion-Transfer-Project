import torch
import os
import cv2
import numpy as np
import math

preprocesses = ['fixratio', 'fixsize']
phases = ['train', 'test']
roots = ['../Motion-Datasets-new/Support-Data/Sup-2D-',
         '../Motion-Datasets-new/Subjects-Data/Sub-']

for preprocess in preprocesses:
    for phase in phases:
        for pre_root in roots:
            # preprocess = 'fixratio' # 'fixratio' or 'fixsize'

            if preprocess == 'fixsize':
                # if phase == 'train':
                #     fix_w = math.ceil(subjects_w[subject_idx - 1] / 8.0) * 8
                #     fix_h = math.ceil(subjects_h[subject_idx - 1] / 8.0) * 8
                # else:
                #     fix_w = math.ceil(subjects_w_test[subject_idx - 1] / 8.0) * 8
                #     fix_h = math.ceil(subjects_h_test[subject_idx - 1] / 8.0) * 8

                fix_w = 512
                fix_h = 512

            root = pre_root + phase
            # root = '../Motion-Datasets-local/Support-Data/Sup-2D-train'
            #root = '../Motion-Datasets/Support-Data/Sup-2D-test'
            # root = '../Motion-Datasets/support-test'

            sup_root = root + '_img'
            bbx_root = root + '_bbx'
            pose_root = root + '_poseImg'
            crop_pose_root = root + '_' + preprocess + '_cropedPoseImg'
            crop_ori_root = root + '_' + preprocess + '_cropedOriImg'

            vid_names = sorted(os.listdir(sup_root))
            vid_names = [name for name in vid_names if name != '.DS_Store']

            for vid_name in vid_names:
                crop_pose_dir = os.path.join(crop_pose_root, vid_name)
                if not os.path.isdir(crop_pose_dir):
                    os.makedirs(crop_pose_dir)

                crop_ori_dir = os.path.join(crop_ori_root, vid_name)
                if not os.path.isdir(crop_ori_dir):
                    os.makedirs(crop_ori_dir)

                oriImg_dir = os.path.join(sup_root, vid_name)
                oriImg_names = sorted(os.listdir(oriImg_dir))
                oriImg_names = [name for name in oriImg_names if name != '.DS_Store']

                poseImg_dir = os.path.join(pose_root, vid_name)
                poseImg_names = sorted(os.listdir(poseImg_dir))
                poseImg_names = [name for name in poseImg_names if name != '.DS_Store']

                bbx_dir = os.path.join(bbx_root, vid_name)
                bbx_names = sorted(os.listdir(bbx_dir))
                bbx_names = [name for name in bbx_names if name != '.DS_Store']

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
                        fix_w = len_max
                        fix_h = len_max

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

                    cropedOriImg_path = os.path.join(crop_ori_dir, poseImg_name)
                    print(str(idx) + '. Croped Original Imag: ' + cropedOriImg_path + ' has been finished!!')
                    cv2.imwrite(cropedOriImg_path, cropedOriImg)

                    cropedPoseImg_path = os.path.join(crop_pose_dir, poseImg_name)
                    print(str(idx) + '. Croped Pose Imag: ' + cropedPoseImg_path + ' has been finished!!')
                    cv2.imwrite(cropedPoseImg_path, cropedPoseImg)





