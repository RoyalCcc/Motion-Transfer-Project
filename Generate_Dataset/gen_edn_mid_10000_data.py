import os
import cv2
import math
import shutil

subs = ['subject1', 'subject2', 'subject3', 'subject4', 'subject5']
# subs = ['subject5']

root = os.path.join('Motion-Datasets', 'EDN-subs')

phases = ['train', 'test']
for phase in phases:
    src_img_dir = os.path.join(root, 'fixmid_' + phase)
    src_pose_img_dir = os.path.join(root, 'fixmid_' + phase + '_poseImg')

    tar_img_dir = os.path.join(root, 'fixmid_' + phase + '_10000')
    tar_pose_img_dir = os.path.join(root, 'fixmid_' + phase + '_poseImg_10000')

    # train_src_dir = os.path.join(root, 'fixmid_train')
    # test_src_dir = os.path.join(root, 'fixmid_test')
    # train_pose_src_dir = os.path.join(root, 'fixmid_train_poseImg')
    # test_pose_src_dir = os.path.join(root, 'fixmid_test_poseImg')

    # train_1000_dir = os.path.join(root, 'fixmid_train_1000')
    # test_1000_dir = os.path.join(root, 'fixmid_test_1000')
    # train_pose_1000_dir = os.path.join(root, 'fixmid_train_poseImg_1000')
    # test_pose_1000_dir = os.path.join(root, 'fixmid_test_poseImg_1000')

    for sub in subs:
        ############### generate train img ##################
        sub_src_img_dir = os.path.join(src_img_dir, sub)
        sub_src_pose_img_dir = os.path.join(src_pose_img_dir, sub)

        sub_tar_img_dir = os.path.join(tar_img_dir, sub)
        sub_tar_pose_img_dir = os.path.join(tar_pose_img_dir, sub)
        
        if not os.path.exists(sub_tar_img_dir):
            os.makedirs(sub_tar_img_dir)

        if not os.path.exists(sub_tar_pose_img_dir):
            os.makedirs(sub_tar_pose_img_dir)

        # all train images
        train_names = sorted(os.listdir(sub_src_pose_img_dir))
        train_names = [name for name in train_names if name != '.DS_Store']
        num_train_name = len(train_names)

        if phase == 'train':
            num_data = min(len(train_names), 10000)
        else:
            num_data = 1000

        interval = math.floor(num_train_name / num_data)
        redun = num_train_name - num_data * interval

        src_img_idx = 0
        count = 0
        for idx in range(num_data):
            train_name = train_names[src_img_idx]

            src_img_path = os.path.join(sub_src_img_dir, train_name)
            src_pose_img_path = os.path.join(sub_src_pose_img_dir, train_name)

            tar_img_path = os.path.join(sub_tar_img_dir, train_name)
            tar_pose_img_path = os.path.join(sub_tar_pose_img_dir, train_name)

            shutil.copy(src_img_path, tar_img_path)
            # print('Move file %s to %s' % (src_img_path, tar_img_path))
            shutil.copy(src_pose_img_path, tar_pose_img_path)
            # print('Move file %s to %s' % (src_pose_img_path, tar_pose_img_path))
            # x = 1
            if idx < redun:
                src_img_idx += interval + 1
            else:
                src_img_idx += interval
            count += 1
        
        print('In %s phase, %s extract %d images' % (phase, sub, count))
