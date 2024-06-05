import os

root = '../Motion-Datasets-mixamo/Mixamo-2D-train_img'

target_names = ['Andromeda', 'Liam', 'Remy', 'Stefani']

for target_name in target_names:
    cur_video_names = sorted(os.listdir(root))
    cur_video_names = [name for name in cur_video_names if name != '.DS_Store']
    print('The results of %s' % target_name)
    for cur_video_name in cur_video_names:
        if cur_video_name.find(target_name) != -1:
            cur_video_path = os.path.join(root, cur_video_name)
            cur_file_names = sorted(os.listdir(cur_video_path))
            cur_file_names = [name for name in cur_file_names if name != '.DS_Store']

            cur_file_num = len(cur_file_names)
            print('%s has %d frames !!!' % (cur_video_name, cur_file_num))

    print('\n')


