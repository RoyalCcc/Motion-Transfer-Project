import os
import shutil

#videos_dir = '../Motion-Datasets-new/Mixamo-4/Sup-Characters-2D-Back'
videos_dir = '../Motion-Datasets-new/Mixamo-4/Sup-Characters-2D'
#videos_dir = '../Motion-Datasets/Sup-2D-Ori-Test'
char_names = sorted(os.listdir(videos_dir))
char_names = [name for name in char_names if name != '.DS_Store']

tar_root_train = '../Motion-Datasets-new/Mixamo-2D-train'
# tar_root_train = '../Motion-Datasets/Sup-2D-test'
# tar_root_total = videos_dir + '-total'
# tar_root_test = '../Motion-Datasets/Mixamo-2D-test'

if videos_dir.find('-Back') != -1:
    flag = '_back'
else:
    flag = ''

num_frame = 0
if not os.path.exists(tar_root_train):
    os.makedirs(tar_root_train)

for char_name in char_names:
    char_path = os.path.join(videos_dir, char_name)

    motion_names = sorted(os.listdir(char_path))
    motion_names = [name for name in motion_names if name != '.DS_Store']

    num_motions = len(motion_names)

    char_num_frame = 0
    # motion_names_train = motion_names[:10]
    # motion_names_test = motion_names[10:]

    for motion_name in motion_names:
        motion_path = os.path.join(char_path, motion_name)
        dis_names = sorted(os.listdir(motion_path))
        dis_names = [name for name in dis_names if name != '.DS_Store']
        dis_names = dis_names[:1]
        num_dis = len(dis_names)

        for dis_name in dis_names:
            dis_path = os.path.join(motion_path, dis_name)
            file_names = sorted(os.listdir(dis_path))
            file_names = [name for name in file_names if name != '.DS_Store']
            num_files = len(file_names)

            time = 1

            tar_dir = char_name.replace(' ', '') + '_' \
                      + motion_name.replace(' ', '') + '_' \
                      + dis_name.replace(' ', '')

            tar_dir_path = os.path.join(tar_root_train, tar_dir)

            tar_dir_path += flag

            if not os.path.exists(tar_dir_path):
                os.makedirs(tar_dir_path)

            for i in range(num_files):
                if i % time != 0:
                    img_name = file_names[i]
                    img_path = os.path.join(dis_path, img_name)
                    # os.remove(img_path)
                else:
                    img_name = file_names[i]
                    img_path = os.path.join(dis_path, img_name)
                    tar_img_path = os.path.join(tar_dir_path, img_name)
                    shutil.copyfile(img_path, tar_img_path)
                    print('Saved %d Frame to %s !!' % (num_frame, tar_img_path))

                    # tar_img_path = os.path.join(tar_root_total, str(num_frame)+'.png')
                    # shutil.copyfile(img_path, tar_img_path)

                    num_frame += 1
                    char_num_frame += 1

    print('%s has %d Frames!!' % (char_name, char_num_frame))

print('Total %d Frames!!' % num_frame)

    # for motion_name in motion_names_test:
    #     motion_path = os.path.join(char_path, motion_name)
    #     dis_names = sorted(os.listdir(motion_path))
    #     dis_names = [name for name in dis_names if name != '.DS_Store']
    #     num_dis = len(dis_names)
    #
    #     for dis_name in dis_names:
    #         dis_path = os.path.join(motion_path, dis_name)
    #         file_names = sorted(os.listdir(dis_path))
    #         file_names = [name for name in file_names if name != '.DS_Store']
    #         num_files = len(file_names)
    #
    #         time = 3
    #
    #         tar_dir = char_name.replace(' ', '') + '_' \
    #                   + motion_name.replace(' ', '') + '_' \
    #                   + dis_name.replace(' ', '')
    #
    #         tar_dir_path = os.path.join(tar_root_test, tar_dir)
    #
    #         if not os.path.exists(tar_dir_path):
    #             os.mkdir(tar_dir_path)
    #
    #         for i in range(num_files):
    #             if i % 3 != 0:
    #                 img_name = file_names[i]
    #                 img_path = os.path.join(dis_path, img_name)
    #                 # os.remove(img_path)
    #             else:
    #                 img_name = file_names[i]
    #                 img_path = os.path.join(dis_path, img_name)
    #                 tar_img_path = os.path.join(tar_dir_path, img_name)
    #                 shutil.copyfile(img_path, tar_img_path)