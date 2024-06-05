import os
import shutil

names = ['sub1', 'sub2']

phases = ['train', 'test']

src_suffixs = ['_img', '_bbx', '_poseImg', '_fixsize_cropedOriImg',
               '_fixsize_cropedPoseImg', '_fixratio_cropedOriImg',
               '_fixratio_cropedPoseImg', '_fail']

root_dir = '../Motion-Datasets-new/Subjects-Data'
tar_dir = '../Motion-Datasets-sub/Subjects-Data'

if not os.path.exists(tar_dir):
    os.makedirs(tar_dir)

for idx, sub in enumerate(names):

    for phase in phases:
        src_dir_name_pre = 'Sub-' + phase
        tar_dir_name_pre = 'Sub-' + phase

        for src_suffix in src_suffixs:
            src_dir_name = src_dir_name_pre + src_suffix
            tar_dir_name = tar_dir_name_pre + src_suffix

            src_type_dir = os.path.join(root_dir, src_dir_name)
            tar_type_dir = os.path.join(tar_dir, tar_dir_name)

            if not os.path.exists(tar_type_dir):
                os.makedirs(tar_type_dir)

            src_video_dir = os.path.join(src_type_dir, sub)
            tar_video_dir = os.path.join(tar_type_dir, sub)

            if not os.path.exists(tar_video_dir):
                os.makedirs(tar_video_dir)


            file_names = os.listdir(src_video_dir)
            file_names = [name for name in file_names if name != '.DS_Store']

            if phase == 'train':
                time = 5
            else:
                time = 3
            for idx2, file_name in enumerate(file_names):
                if idx2 % time != 0:
                    continue

                src_file_path = os.path.join(src_video_dir, file_name)
                tar_file_path = os.path.join(tar_video_dir, file_name)

                shutil.move(src_file_path, tar_file_path)
