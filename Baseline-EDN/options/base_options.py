### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import argparse
import os
from util import util
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):    
        # experiment specifics
        self.parser.add_argument('--name', type=str, default='subject1', help='name of the experiment. It decides where to store samples and models')        
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')                       
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')        
        self.parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')

        # input/output sizes       
        self.parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=512, help='then crop to this size')
        self.parser.add_argument('--label_nc', type=int, default=3*2, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

        # for setting inputs
        self.parser.add_argument('--dataset', type=str, default='edn', help='chooses which target. [edn | edn_1000 | mixamo | Stefani | others]') 
        self.parser.add_argument('--dataroot', type=str, default='../Motion-Datasets/EDN-subs') 
        self.parser.add_argument('--target_name', type=str, default='Andromeda', help='chooses which target. [Andromeda | Liam | Remy | Stefani]')
        self.parser.add_argument('--source_name', nargs='+', help='chooses which target. [Andromeda | Liam | Remy | Stefani | others]')
        self.parser.add_argument('--resize_or_crop', type=str, default='scale_width', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')        
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation') 
        self.parser.add_argument('--nThreads', default=4, type=int, help='# threads for loading data')                
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

        # for displays
        self.parser.add_argument('--display_winsize', type=int, default=512,  help='display window size')
        self.parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')

        # for generator
        self.parser.add_argument('--netG', type=str, default='global', help='selects model to use for netG')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--n_downsample_global', type=int, default=4, help='number of downsampling layers in netG') 
        self.parser.add_argument('--n_blocks_global', type=int, default=9, help='number of residual blocks in the global generator network')
        self.parser.add_argument('--n_blocks_local', type=int, default=3, help='number of residual blocks in the local enhancer network')
        self.parser.add_argument('--n_local_enhancers', type=int, default=1, help='number of local enhancers to use')        
        self.parser.add_argument('--niter_fix_global', type=int, default=0, help='number of epochs that we only train the outmost local enhancer')        

        # for instance-wise features
        self.parser.add_argument('--no_instance', action='store_true', help='if specified, do *not* add instance map as input')        
        self.parser.add_argument('--instance_feat', action='store_true', help='if specified, add encoded instance features as input')
        self.parser.add_argument('--label_feat', action='store_true', help='if specified, add encoded label features as input')        
        self.parser.add_argument('--feat_num', type=int, default=3, help='vector length for encoded features')        
        self.parser.add_argument('--load_features', action='store_true', help='if specified, load precomputed feature maps')
        self.parser.add_argument('--n_downsample_E', type=int, default=4, help='# of downsampling layers in encoder') 
        self.parser.add_argument('--nef', type=int, default=16, help='# of encoder filters in the first conv layer')        
        self.parser.add_argument('--n_clusters', type=int, default=10, help='number of clusters for features')

        # for face discriminator
        self.parser.add_argument('--face_discrim', action='store_true', help='if specified, add a face discriminator')
        self.parser.add_argument('--niter_fix_main', type=int, default=0, help='number of epochs that we only train the face discriminator')        

        #for face generator
        self.parser.add_argument('--face_generator', action='store_true', help='if specified, add a face residual prediction generator')
        self.parser.add_argument('--faceGtype', type=str, default='unet', help='selects architecture to use for face generator, choose from a UNet generator or global generator [unet|global]')

        # for gestures, only do 64 frame segments
        self.parser.add_argument('--gestures', action='store_true', help='for gestures project 64 frames')


        self.initialized = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        if self.opt.dataset == 'edn':
            self.opt.dataroot = '../Motion-Datasets/EDN-subs'
            if self.opt.phase == 'train':
                self.opt.input_data_dir = 'fixmid_train_poseImg'
                self.opt.gt_data_dir = 'fixmid_train'
            elif self.opt.phase == 'test':
                self.opt.input_data_dir = 'fixmid_test_poseImg'
                self.opt.gt_data_dir = 'fixmid_test'
        elif self.opt.dataset == 'edn_1000':
            self.opt.dataroot = '../Motion-Datasets/EDN-subs'
            if self.opt.phase == 'train':
                self.opt.input_data_dir = 'fixmid_train_poseImg_1000'
                self.opt.gt_data_dir = 'fixmid_train_1000'
            elif self.opt.phase == 'test':
                self.opt.input_data_dir = 'fixmid_test_poseImg_1000'
                self.opt.gt_data_dir = 'fixmid_test_1000'
        elif self.opt.dataset == 'edn_10000':
            self.opt.dataroot = '../Motion-Datasets/EDN-subs'
            if self.opt.phase == 'train':
                self.opt.input_data_dir = 'fixmid_train_poseImg_10000'
                self.opt.gt_data_dir = 'fixmid_train_10000'
            elif self.opt.phase == 'test':
                self.opt.input_data_dir = 'fixmid_test_poseImg_10000'
                self.opt.gt_data_dir = 'fixmid_test_10000'
        elif self.opt.dataset == 'mixamo':
            self.opt.dataroot = '../Motion-Datasets/mixamo'
            if self.opt.phase == 'train':
                self.opt.input_data_dir = 'Mixamo-2D-train_fixratio_cropedPoseImg'
                self.opt.gt_data_dir = 'Mixamo-2D-train_fixratio_cropedOriImg'
            elif self.opt.phase == 'test':
                self.opt.input_data_dir = 'Mixamo-2D-test_fixratio_cropedPoseImg'
                self.opt.gt_data_dir = 'Mixamo-2D-test_fixratio_cropedOriImg'
        elif self.opt.dataset == 'mpi':
            self.opt.dataroot = '../Motion-Datasets/mpi_inf_3dhp'
            if self.opt.phase == 'train':
                self.opt.input_data_dir = 'mpi_train'
                self.opt.gt_data_dir = 'mpi_train'
            elif self.opt.phase == 'test':
                self.opt.input_data_dir = 'mpi_test'
                self.opt.gt_data_dir = 'mpi_test'

        self.opt.name = self.opt.target_name

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        
        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk        
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        if save and not self.opt.continue_train:
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
