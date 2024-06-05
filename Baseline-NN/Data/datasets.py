from __future__ import print_function

import os
import os.path
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import random
import time
import sys

def get_transform(transform_params, size):
    scale_type = transform_params['scale_type']
    load_width = transform_params['load_width']
    load_height = transform_params['load_height']
    normalize = transform_params['normalize']
    toTensor = transform_params['toTensor']

    w, h = size
    new_w = w
    new_h = h

    if scale_type == 'resize':
        new_w = load_width
        new_h = load_height
    if scale_type == 'scale_width':
        new_w = load_width
        new_h = (new_w / w) * h
    elif scale_type == 'scale_height':
        new_h = load_height
        new_w = (new_h / h) * w

    new_w = int(round(new_w / 8)) * 8
    new_h = int(round(new_h / 8)) * 8

    transform_list = []
    transform_list.append(transforms.Resize((new_h, new_w)))

    if toTensor:
        transform_list += [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

class UnitedSourceDataset(data.Dataset):
    def __init__(self, opt):
        self.phase = opt['phase']

        self.root_dir = opt['root_dir']
        self.data_dir = os.path.join(self.root_dir, opt['data_dir'])

        self.num_clip_len = opt['num_clip_len']

        # When syn video, opt['target_names'] must have only one source.
        self.target_name = opt['target_names']
        self.target_frames_ac = []
        self.target_infos = dict()
        self.target_infos['video_names'] = []
        self.target_infos['video_frames'] = []
        self.target_infos['video_frames_ac'] = []
        self.target_infos['target_total_frames'] = 0

        self.total_frames = 0
        self.data_paths = []
        self.pose_data_paths = []
        self.cur_video_names = sorted(os.listdir(self.data_dir))
        self.cur_video_names = [name for name in self.cur_video_names if name != '.DS_Store']
        for cur_video_name in self.cur_video_names:
            if cur_video_name.find('_back') != -1:
                continue

            if cur_video_name.find(self.target_name) != -1:
                self.target_infos['video_names'].append(cur_video_name)

                cur_path = os.path.join(self.data_dir, cur_video_name)

                img_names = sorted(os.listdir(cur_path))
                img_names = [name for name in img_names if name != '.DS_Store']
                for name in img_names:
                    img_path = os.path.join(cur_path, name)
                    self.data_paths.append(img_path)

                cur_num_frames = len(img_names)
                cur_num_syn_frame = cur_num_frames - self.num_clip_len + 1

                self.target_infos['video_frames'].append(cur_num_frames)
                self.target_infos['video_frames_ac'].append(self.total_frames + cur_num_syn_frame)
                self.target_infos['target_total_frames'] += cur_num_frames
                self.total_frames += cur_num_syn_frame

        self.target_frames_ac.append(self.total_frames)

        self.transform_params = opt['transform_params']

    def find_video(self, index):
        video_frames_ac = self.target_infos['video_frames_ac']
        for video_idx, frame_ac in enumerate(video_frames_ac):
            if index < frame_ac:
                return video_idx

        return -1

    def __getitem__(self, index):
        video_index = self.find_video(index)

        base = 0
        for idx in range(video_index):
            base += self.target_infos['video_frames'][idx]

        index -= base

        dyc_imgs = []
        dyc_names = []

        for i in range(self.num_clip_len):
            cur_index = index + i
            img_path = self.data_paths[cur_index + base]
            img = Image.open(img_path).convert('RGB')

            size = img.size
            transform = get_transform(self.transform_params, size)

            if transform is not None:
                img = transform(img)

            dyc_imgs.append(img)
            dyc_names.append(img_path)

        dyc_imgs = torch.stack(dyc_imgs)

        return dyc_imgs, dyc_names

    def __len__(self):
        return self.total_frames - (self.num_clip_len - 1)

    def initialize(self, opt):
        pass


    def __init__(self, opt):
        self.phase = opt['phase']
        # self.root_dir = opt['root_dir']
        # self.data_dir = os.path.join(self.root_dir, opt['data_dir'])
        self.data_dir = opt['video_dir']
        self.data_names = sorted(os.listdir(self.data_dir))
        self.data_names = [name for name in self.data_names if name != '.DS_Store']

        self.num_frames = len(self.data_names)
        self.num_clip_len = opt['num_clip_len']

        self.transform_params = opt['transform_params']

    def __getitem__(self, index):
        dyc_imgs = []
        # dyc_index = []
        dyc_names = []

        for i in range(self.num_clip_len):
            cur_index = index + i
            img_name = self.data_names[cur_index]
            img_path = os.path.join(self.data_dir, img_name)
            img = Image.open(img_path).convert('RGB')

            size = img.size
            transform = get_transform(self.transform_params, size)

            if transform is not None:
                img = transform(img)

            dyc_imgs.append(img)
            # dyc_index.append(cur_index)
            dyc_names.append(img_path)

        dyc_imgs = torch.stack(dyc_imgs)

        return dyc_imgs, dyc_names

    def __len__(self):
        return self.num_frames - (self.num_clip_len - 1)

    def initialize(self, opt):
        pass