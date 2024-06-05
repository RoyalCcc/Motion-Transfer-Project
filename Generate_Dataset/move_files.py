import os
import shutil

def copy_dir(src_dir, tar_dir):
    if not os.path.exists(src_dir):
        print('%s is not existed !!!!' % src_dir)

    file_names = sorted(os.listdir(src_dir))
    file_names = [name for name in file_names if name != '.DS_Store']

    for file_name in file_names:
        src_file_path = os.path.join(src_dir, file_name)
        tar_file_path = os.path.join(tar_dir, file_name)

        shutil.copyfile(src_file_path, tar_file_path)

if __name__ == "__main__":
    src_root = '../Motion-Datasets'
    tar_root = '../Motion-Datasets1/Subjects-Data'

    if not os.path.exists(tar_root):
        os.makedirs(tar_root)

    phases = ['train', 'test']

    src_subs = ['subject1', 'subject2', 'subject3', 'subject4', 'subject5']
    tar_subs = ['sub1', 'sub2', 'sub3', 'sub4', 'sub5']

    src_suffixs = ['_img', '_bbx', '_poseImg', '_fixsize_cropedOriImg',
                   '_fixsize_cropedPoseImg', '_fixratio_cropedOriImg',
                   '_fixratio_cropedPoseImg', 'failed_img']

    for idx, tar_sub in enumerate(tar_subs):
        src_sub = src_subs[idx]
        print('Current subject is %s' % src_sub)
        src_sub_dir = os.path.join(src_root, src_sub)

        for phase in phases:
            src_sub_phase_dir = os.path.join(src_sub_dir, phase)
            tar_dir_name_pre = 'Sub-' + phase

            for src_suffix in src_suffixs:
                if src_suffix.find('failed') != -1:
                    src_file_dir = os.path.join(src_sub_phase_dir, src_suffix)
                    tar_dir_name = tar_dir_name_pre + '_fail'
                else:
                    src_file_dir = os.path.join(src_sub_phase_dir, phase + src_suffix)
                    tar_dir_name = tar_dir_name_pre + src_suffix

                tar_type_dir = os.path.join(tar_root, tar_dir_name)
                if not os.path.exists(tar_type_dir):
                    os.makedirs(tar_type_dir)

                tar_sub_dir = os.path.join(tar_type_dir, tar_sub)
                if not os.path.exists(tar_sub_dir):
                    os.makedirs(tar_sub_dir)

                copy_dir(src_file_dir, tar_sub_dir)



