import os
import cv2

# subs = ['subject1', 'subject2', 'subject3', 'subject4']
subs = ['subject5']

root = os.path.join('Motion-Datasets', 'EDN-subs')
train_dir = os.path.join(root, 'fixmid_train')
test_dir = os.path.join(root, 'fixmid_test')

if not os.path.exists(train_dir):
    os.makedirs(train_dir)

if not os.path.exists(test_dir):
    os.makedirs(test_dir)

for sub in subs:
    cur_src_train_dir = os.path.join(root, sub, 'train', 'train_img')
    cur_src_test_dir = os.path.join(root, sub, 'val', 'test_img')

    ############### generate train img ##################
    sub_train_dir = os.path.join(train_dir, sub)
    
    if not os.path.exists(sub_train_dir):
        os.makedirs(sub_train_dir)

    train_names = sorted(os.listdir(cur_src_train_dir))
    train_names = [name for name in train_names if name != '.DS_Store']
    num_train_name = len(train_names)

    for idx, train_name in enumerate(train_names):  
        train_path = os.path.join(cur_src_train_dir, train_name)
        train_img = cv2.imread(train_path)  # B,G,R order
        ori_h = train_img.shape[0]
        ori_w = train_img.shape[1]

        w_offset = ori_h // 2  # 512 / 2
        x_center = ori_w // 2  # 1024 / 2
        x_min_coor = x_center - w_offset
        x_max_coor = x_center + w_offset

        fixmid_img = train_img[:, x_min_coor:x_max_coor]

        fixmid_img_path = os.path.join(sub_train_dir, train_name)
        #print(str(idx) + '. Croped Original Imag: ' + train_name + ' has been finished!!')
        cv2.imwrite(fixmid_img_path, fixmid_img)
    
    print('%s finished %d images for training' % (sub, num_train_name))

    ############### generate test img ##################
    sub_test_dir = os.path.join(test_dir, sub)
    
    if not os.path.exists(sub_test_dir):
        os.makedirs(sub_test_dir)

    test_names = sorted(os.listdir(cur_src_test_dir))
    test_names = [name for name in test_names if name != '.DS_Store']
    num_test_name = len(test_names)

    for idx, test_name in enumerate(test_names):  
        test_path = os.path.join(cur_src_test_dir, test_name)
        test_img = cv2.imread(test_path)  # B,G,R order
        ori_h = test_img.shape[0]
        ori_w = test_img.shape[1]

        w_offset = ori_h // 2  # 512 / 2
        x_center = ori_w // 2  # 1024 / 2
        x_min_coor = x_center - w_offset
        x_max_coor = x_center + w_offset

        fixmid_img = test_img[:, x_min_coor:x_max_coor]

        fixmid_img_path = os.path.join(sub_test_dir, test_name)
        #print(str(idx) + '. Croped Original Imag: ' + test_name + ' has been finished!!')
        cv2.imwrite(fixmid_img_path, fixmid_img)
    print('%s finished %d images for test' % (sub, num_test_name))