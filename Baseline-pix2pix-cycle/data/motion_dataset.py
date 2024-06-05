import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image


class MotionDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        # opt.dataroot = '../Motion-Datasets-new/Support-Data'
        # opt.input_data_dir = 'Sup-2D-train_fixratio_cropedPoseImg'
        # opt.gt_data_dir = 'Sup-2D-train_fixratio_cropedOriImg'
        self.dataroot = opt.dataroot
        self.input_data_dir = opt.input_data_dir
        self.gt_data_dir = opt.gt_data_dir

        self.dir_pose = os.path.join(opt.dataroot, opt.input_data_dir)
        self.dir_gt = os.path.join(opt.dataroot, opt.gt_data_dir)

        self.num_clip_len = 1

        # When syn video, opt['target_name'] must have only one source.
        if opt.phase == 'train':
            self.target_name = opt.target_name
        else:
            self.target_name = opt.source_name

        self.target_frames_ac = []
        self.target_infos = dict()
        self.target_infos['video_names'] = []
        self.target_infos['video_frames'] = []
        self.target_infos['video_frames_ac'] = []
        self.target_infos['target_total_frames'] = 0

        self.total_frames = 0
        self.data_paths = []
        self.pose_data_paths = []
        self.cur_video_names = sorted(os.listdir(self.dir_gt))
        self.cur_video_names = [name for name in self.cur_video_names if name != '.DS_Store']
        for cur_video_name in self.cur_video_names:
            if cur_video_name.find(self.target_name) != -1:
                self.target_infos['video_names'].append(cur_video_name)

                cur_path = os.path.join(self.dir_gt, cur_video_name)
                cur_pose_path = os.path.join(self.dir_pose, cur_video_name)

                img_names = sorted(os.listdir(cur_pose_path))
                img_names = [name for name in img_names if name != '.DS_Store']
                for name in img_names:
                    img_path = os.path.join(cur_path, name)
                    self.data_paths.append(img_path)

                    img_pose_path = os.path.join(cur_pose_path, name)
                    self.pose_data_paths.append(img_pose_path)

                cur_num_frames = len(img_names)
                cur_num_syn_frame = cur_num_frames - self.num_clip_len + 1

                self.target_infos['video_frames'].append(cur_num_frames)
                self.target_infos['video_frames_ac'].append(self.total_frames + cur_num_syn_frame)
                self.target_infos['target_total_frames'] += cur_num_frames
                self.total_frames += cur_num_syn_frame

        self.target_frames_ac.append(self.total_frames)

        # self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        # self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def find_video(self, index):
        video_frames_ac = self.target_infos['video_frames_ac']
        for video_idx, frame_ac in enumerate(video_frames_ac):
            if index < frame_ac:
                return video_idx

        return -1

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        video_index = self.find_video(index)

        base = 0
        for idx in range(video_index):
            base += self.target_infos['video_frames'][idx]

        index -= base

        cur_index = index
        img_path = self.data_paths[cur_index + base]
        img = Image.open(img_path).convert('RGB')

        pose_img_path = self.pose_data_paths[cur_index + base]
        pose_img = Image.open(pose_img_path).convert('RGB')

        A = pose_img
        A_path = pose_img_path

        B = img
        B_path = img_path

        # # read a image given a random integer index
        # AB_path = self.AB_paths[index]
        # AB = Image.open(AB_path).convert('RGB')
        # # split AB image into A and B
        # w, h = AB.size
        # w2 = int(w / 2)
        # A = AB.crop((0, 0, w2, h))
        # B = AB.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        # return len(self.AB_paths)
        return self.total_frames - (self.num_clip_len - 1)
