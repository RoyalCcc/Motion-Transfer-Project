import os
import cv2
import numpy as np

a = os.getcwd()
os.chdir(os.getcwd())

person = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']

root = 'mpi_inf_3dhp'

result_dir = os.path.join(root, 'image_results')
x = os.path.abspath(result_dir)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

for idx in range(8):
    person_dir = os.path.join(root, person[idx])
    sequence_dir = os.path.join(person_dir, 'Seq1') 
    video_dir = os.path.join(sequence_dir, 'imageSequence')
    video_names = sorted(os.listdir(video_dir)) 
    video_names = [name for name in video_names if name.find('8.avi') != -1]

    for video_name in video_names:
        cur_video_train_dir = person[idx] + '_seq1_' + video_name.split('.')[0] + '_train'
        result_image_train_dir = os.path.join(result_dir,  cur_video_train_dir)

        cur_video_test_dir = person[idx] + '_seq1_' + video_name.split('.')[0] + '_test'
        result_image_test_dir = os.path.join(result_dir,  cur_video_test_dir)

        if not os.path.exists(result_image_train_dir):
            os.makedirs(result_image_train_dir)
        
        if not os.path.exists(result_image_test_dir):
            os.makedirs(result_image_test_dir)

        video_path = os.path.join(video_dir, video_name)
        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        frame_num = cap.get(7)

        print('%s: fps %d, num %d' % (video_path, fps, frame_num))
        
        if fps == 50:
            interval = 4
        else:
            interval = 2
        
        set_split = (frame_num // interval) // 2

        image_count = 0
        w = 512
        h = 512
        dim = (w, h)
        ret = True
        while(ret):
            # get a frame
            ret, frame = cap.read()
            if not ret:
                break

            resized_frame = cv2.resize(frame, dim)

            if image_count % interval == 0:
                name_idx = image_count // interval
                image_name = str(name_idx) + '.jpg'
                if name_idx <= set_split:
                    result_image_dir = result_image_train_dir
                else:
                    result_image_dir = result_image_test_dir

                image_path = os.path.join(result_image_dir, image_name)
                cv2.imwrite(image_path, frame)
                print('curretn %s' % image_path)

            image_count += 1

        cap.release()
        cv2.destroyAllWindows() 

