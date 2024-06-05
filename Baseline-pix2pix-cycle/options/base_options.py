import argparse
import os
from util import util
import torch
import models
import data


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        # dataroot = '../Motion-Datasets-new/Support-Data'
        # parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        # parser.add_argument('--dataroot', type=str, default='../Motion-Datasets-mixamo')
        # parser.add_argument('--input_data_dir', type=str, default='Mixamo-2D-train_fixratio_cropedPoseImg')
        # parser.add_argument('--gt_data_dir', type=str, default='Mixamo-2D-train_fixratio_cropedOriImg')
        parser.add_argument('--dataset', type=str, default='edn',
                            help='chooses which target. [edn | mpi | mixamo | Stefani | others]') 
        # parser.add_argument('--dataroot', type=str, default='../Motion-Datasets/mpi_inf_3dhp') 
        # parser.add_argument('--input_data_dir', type=str, default='mpi_train')
        # parser.add_argument('--gt_data_dir', type=str, default='mpi_train')
        parser.add_argument('--target_name', type=str, default='Andromeda',
                            help='chooses which target. [Andromeda | Liam | Remy | Stefani]')
        # parser.add_argument('--source_name', type=str, default='Andromeda',
        #                     help='chooses which target. [Andromeda | Liam | Remy | Stefani | others]')
        parser.add_argument('--source_name',
                            nargs='+',
                            help='chooses which target. [Andromeda | Liam | Remy | Stefani | others]')

        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        # model parameters
        parser.add_argument('--model', type=str, default='pix2pix', help='chooses which model to use. [cycle_gan | pix2pix | test | colorization]')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--ngf', type=int, default=32, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=32, help='# of discrim filters in the first conv layer')
        parser.add_argument('--netD', type=str, default='basic', help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
        parser.add_argument('--netG', type=str, default='unet_128', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        # dataset parameters
        # parser.add_argument('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
        parser.add_argument('--dataset_mode', type=str, default='motion', help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
        parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
        parser.add_argument('--load_size', type=int, default=256, help='scale images to this size') #286
        parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name + '_' + opt.netG)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        if opt.isTrain and opt.model.find('cycle') != -1:
            opt.lr = opt.lr * 0.1

        if opt.dataset == 'edn':
            opt.dataroot = '../Motion-Datasets/EDN-subs'
            if opt.phase == 'train':
                opt.input_data_dir = 'fixmid_train_poseImg'
                opt.gt_data_dir = 'fixmid_train'
            elif opt.phase == 'test':
                opt.input_data_dir = 'fixmid_test_poseImg'
                opt.gt_data_dir = 'fixmid_test'
        elif opt.dataset == 'edn_1000':
            opt.dataroot = '../Motion-Datasets/EDN-subs'
            if opt.phase == 'train':
                opt.input_data_dir = 'fixmid_train_poseImg_1000'
                opt.gt_data_dir = 'fixmid_train_1000'
            elif opt.phase == 'test':
                opt.input_data_dir = 'fixmid_test_poseImg_1000'
                opt.gt_data_dir = 'fixmid_test_1000'
        elif opt.dataset == 'edn_10000':
            opt.dataroot = '../Motion-Datasets/EDN-subs'
            if opt.phase == 'train':
                opt.input_data_dir = 'fixmid_train_poseImg_10000'
                opt.gt_data_dir = 'fixmid_train_10000'
            elif opt.phase == 'test':
                opt.input_data_dir = 'fixmid_test_poseImg_10000'
                opt.gt_data_dir = 'fixmid_test_10000'
        elif opt.dataset == 'mixamo':
            opt.dataroot = '../Motion-Datasets/mixamo'
            if opt.phase == 'train':
                opt.input_data_dir = 'Mixamo-2D-train_fixratio_cropedPoseImg'
                opt.gt_data_dir = 'Mixamo-2D-train_fixratio_cropedOriImg'
            elif opt.phase == 'test':
                opt.input_data_dir = 'Mixamo-2D-test_fixratio_cropedPoseImg'
                opt.gt_data_dir = 'Mixamo-2D-test_fixratio_cropedOriImg'
        elif opt.dataset == 'mpi':
            opt.dataroot = '../Motion-Datasets/mpi_inf_3dhp'
            if opt.phase == 'train':
                opt.input_data_dir = 'mpi_train'
                opt.gt_data_dir = 'mpi_train'
            elif opt.phase == 'test':
                opt.input_data_dir = 'mpi_test'
                opt.gt_data_dir = 'mpi_test'

        opt.name = opt.target_name + '_' + opt.model

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
