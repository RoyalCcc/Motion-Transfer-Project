from __future__ import print_function
import os

import numpy as np
import cv2
import torch
import torch.nn.functional as F

import Models.warmUpLR as warmup
from Models.experiments_builder import Experiments_Builder

from Models.SSIM import *
from Data.dataloaders import DataloaderFactory
from Data.datasets import get_transform
from PIL import Image
import torch.nn as nn
import time

class Experiments_Trainer(Experiments_Builder):
    def __init__(self, config):
        Experiments_Builder.__init__(self, config)

    def get_heatmap(self, imgs):
        b, c, h, w = imgs.size()
        imgs = imgs.view(b, -1, 3, h, w)

        heatmaps = []
        for i in range(imgs.size()[1]):
            img = imgs[:, i, :, :, :]
            predicted_outputs, _ = self.networks['openpose'](img)
            paf, heatmap = predicted_outputs[0], predicted_outputs[1]
            heatmap = heatmap[:, :18, :, :]

            b, c, h, w = heatmap.size()
            heatmap = heatmap.view(b, c, -1)

            heatmap_thrs2 = heatmap.view(b, c, h, w)

            #heatmap_thrs2 = torch.cat((paf, heatmap_thrs2), 1)

            heatmaps.append(heatmap_thrs2)

        heatmaps = torch.stack(heatmaps, 1)
        b, n, c, h, w = heatmaps.size()
        assert n == self.config['in_dyc_encoder']
        heatmaps = heatmaps.view(b, -1, h, w)
        #heatmaps = output2.cpu().data.numpy().transpose(0, 2, 3, 1)[0]
        #paf = output1.cpu().data.numpy().transpose(0, 2, 3, 1)[0]

        return heatmaps

    def get_l2_loss(self, real_features, syn_features):
        l2_loss = nn.MSELoss(reduction='sum')
        #diff = syn_features - real_features.detach()
        #loss = torch.mean(torch.abs(diff[:]))
        loss = l2_loss(real_features, syn_features)
        return loss.cpu().numpy()

    def synVideo(self, train_dataloader, test_dataloader, test_name):
        for key, network in self.networks.items():
            network.eval()

        dir_name = test_name + '_syn_NN'
        source_result_dir = os.path.join(self.exp_dir,
                                         dir_name)

        if not os.path.exists(source_result_dir):
            os.mkdir(source_result_dir)

        frame_num = 0
        for src_idx, source_batch in enumerate(test_dataloader):
            frame_num += 1

            min_l2_loss = 100000.0
            best_result_img = None
            best_result_name = None
            start_time = time.time()

            source_img, source_name = source_batch
            if self.cuda:
                source_img = source_img.cuda()

            b, _, c, h, w = source_img.size()
            source_img = source_img.view(b, -1, h, w)

            with torch.no_grad():
                cur_source_heatmap = self.get_heatmap(source_img)

            # train_opt['target_names'] = train_opt['target_names'][0]
            print('########################################')
            # print(train_opt['target_names'])
            # train_dataloader = DataloaderFactory(train_opt).load_dataloader()

            for tar_idx, target_batch in enumerate(train_dataloader):
                if tar_idx % self.config['train_interval'] != 0:
                    continue

                target_img, target_name = target_batch

                if self.cuda:
                    target_img = target_img.cuda()

                b, _, c, h, w = target_img.size()
                target_img = target_img.view(b, -1, h, w)

                tar_file_name = target_name[0][0].split('/')[-1]
                img_suffix = tar_file_name.split('.')[-1]
                tar_np_file_name = tar_file_name.replace(img_suffix, 'npy')

                tar_vid_name = target_name[0][0].split('/')[-2]
                tar_type_dir_name = target_name[0][0].split('/')[-3]

                tar_ht_dir_name = tar_type_dir_name + '_heatmap'
                
                tar_ht_vid_path = os.path.dirname(target_name[0][0])
                tar_ht_vid_path = tar_ht_vid_path.replace(tar_type_dir_name, tar_ht_dir_name)
                tar_ht_np_path = os.path.join(tar_ht_vid_path, tar_np_file_name)

                #print('#################################')
                #print('tar_file_name: %s ' % (tar_file_name))
                #print('tar_np_file_name: %s ' % (tar_np_file_name))
                #print('tar_vid_name: %s ' % (tar_vid_name))
                #print('tar_type_dir_name: %s ' % (tar_type_dir_name))
                #print('tar_ht_dir_name: %s ' % (tar_ht_dir_name))
                #print('tar_ht_vid_path: %s ' % (tar_ht_vid_path))
                #print('tar_ht_np_path: %s ' % (tar_ht_np_path))

                if not os.path.exists(tar_ht_np_path):
                    if not os.path.exists(tar_ht_vid_path):
                        os.makedirs(tar_ht_vid_path)

                    with torch.no_grad():
                        cur_target_heatmap = self.get_heatmap(target_img)
                    
                    cur_target_heatmap_np = cur_target_heatmap.data.cpu().numpy()
                    np.save(tar_ht_np_path, cur_target_heatmap_np)
                    #print('Saved file %s' % (tar_ht_np_path))
                else:
                    cur_target_heatmap_np = np.load(tar_ht_np_path)
                    cur_target_heatmap = torch.tensor(cur_target_heatmap_np)

                    if self.cuda:
                        cur_target_heatmap = cur_target_heatmap.cuda()

                    #print('Loaded file %s' % (tar_ht_np_path))

                #print('#################################')

                # print('Cal L2 loss between %s and %s!!' % (source_name, target_name))
                l2_loss = self.get_l2_loss(cur_source_heatmap, cur_target_heatmap)

                if tar_idx == 0:
                    min_l2_loss = l2_loss
                    best_result_img = target_img
                    best_result_name = target_name

                if l2_loss < min_l2_loss:
                    min_l2_loss = l2_loss
                    best_result_img = target_img
                    best_result_name = target_name

            is_Normalized = train_dataloader.dataset.transform_params['normalize']
            cur_syn_img = best_result_img
            cur_syn_img_result = self.tensor2im(cur_syn_img[0, :, :, :],
                                                is_Normalized)
            cur_syn_img_result = np.transpose(cur_syn_img_result, (1, 2, 0))

            video_name = source_name[0][0].split('/')[-2]
            video_path = os.path.join(source_result_dir, video_name)
            if not os.path.exists(video_path):
                os.mkdir(video_path)

            img_file_name = 'syn_' + source_name[0][0].split('/')[-1]
            img_file_path = os.path.join(video_path, img_file_name)

            end_time = time.time()
            time_per_img = end_time - start_time

            from PIL import Image
            im = Image.fromarray(cur_syn_img_result)
            im.save(img_file_path)
            #self.logger.info('Saved file %s !!!!' % img_file_path)
            print('Time: %.3f - Saved file %s to %s!!!!' % (time_per_img, best_result_name, img_file_path))











