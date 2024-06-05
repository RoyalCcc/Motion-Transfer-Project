import torch
import os
import cv2
from PIL import Image
import argparse
from importlib import util
import xlwt
import xlrd
from xlutils.copy import copy
import socket

import PerceptualSimilarity
import PerceptualSimilarity.models as models
from PerceptualSimilarity.util import util as LPIPS_util
from SSIM import *
import numpy as np

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

def calc_scores(real_dir, syn_dir, alex_lpips,
                vgg_lpips, model_ssim, enable_cuda):
    print('Scores between: \n %s \n %s' % (real_dir, syn_dir))

    mse_score = 0.0
    psnr_score = 0.0
    ssim_score = 0.0
    model_ssim_score = 0.0
    alex_lpips_scores = 0.0
    vgg_lpips_scores = 0.0

    # real_data_names = sorted(os.listdir(real_dir))
    # real_data_names = [name for name in real_data_names if name != '.DS_Store']

    syn_data_names = sorted(os.listdir(syn_dir))
    syn_data_names = [name for name in syn_data_names if name != '.DS_Store']

    num_frames = len(syn_data_names)
    # real_data_names = real_data_names[-num_frames:]

    num_invaild = 0
    for idx in range(num_frames):
        # real_data_name = real_data_names[idx]
        syn_data_name = syn_data_names[idx]
        real_data_name = syn_data_name.split('_')[-1]
        # real_img_path = os.path.join(real_dir, real_data_name)
        real_img_path = os.path.join(real_dir, real_data_name)
        if not os.path.exists(real_img_path):
            num_invaild += 1
            continue
        real_img = Image.open(real_img_path).convert('RGB')
        # real_img = cv2.imread(real_img_path)

        syn_img_path = os.path.join(syn_dir, syn_data_name)
        syn_img = Image.open(syn_img_path).convert('RGB')
        # syn_img = cv2.imread(syn_img_path)

        # size = syn_img.shape[:2]
        size = syn_img.size

        real_img = real_img.resize(size, Image.BILINEAR)
        real_img = np.array(real_img)
        syn_img = np.array(syn_img)
        # real_img1 = cv2.resize(real_img, size)

        cur_mse = cal_mse(real_img, syn_img)
        cur_psnr = cal_psnr(real_img, syn_img)
        cur_ssim = calc_ssim(real_img, syn_img)

        mse_score += cur_mse
        psnr_score += cur_psnr
        ssim_score += cur_ssim

        # Load images
        real_img = LPIPS_util.load_image(real_img_path)  # RGB image from [-1,1]
        syn_img = LPIPS_util.load_image(syn_img_path)

        size = syn_img.shape[:2]
        real_img = cv2.resize(real_img, size, interpolation=cv2.INTER_LINEAR)

        real_img = LPIPS_util.im2tensor(real_img)  # RGB image from [-1,1]
        syn_img = LPIPS_util.im2tensor(syn_img)

        if enable_cuda:
            real_img = real_img.cuda()
            syn_img = syn_img.cuda()

        # Compute distance
        cur_alex_lpips = alex_lpips.forward(real_img, syn_img)
        cur_alex_lpips = cur_alex_lpips.detach().cpu().numpy()
        cur_alex_lpips = np.squeeze(cur_alex_lpips)
        # print('Distance: %.3f' % cur_lpips)
        alex_lpips_scores += cur_alex_lpips

        # Compute distance
        cur_vgg_lpips = vgg_lpips.forward(real_img, syn_img)
        cur_vgg_lpips = cur_vgg_lpips.detach().cpu().numpy()
        cur_vgg_lpips = np.squeeze(cur_vgg_lpips)
        # print('Distance: %.3f' % cur_lpips)
        vgg_lpips_scores += cur_vgg_lpips

        cur_model_ssim = model_ssim.forward(real_img, syn_img)
        cur_model_ssim = cur_model_ssim.detach().cpu().numpy()
        cur_model_ssim = np.squeeze(cur_model_ssim)
        model_ssim_score += cur_model_ssim

    num_frames -= num_invaild
    mse_score /= float(num_frames)
    psnr_score /= float(num_frames)
    ssim_score /= float(num_frames)
    model_ssim_score /= float(num_frames)
    alex_lpips_scores /= float(num_frames)
    vgg_lpips_scores /= float(num_frames)

    #lpips_scores = 1.0 - lpips_scores

    print('Toral %d Frames:' % num_frames)
    print('MSE Score: %f' % mse_score)
    print('Alex LPIPS Score: %f' % alex_lpips_scores)
    print('VGG LPIPS Score: %f' % vgg_lpips_scores)
    print('PSNR Score: %f' % psnr_score)
    print('SSIM Score: %f' % ssim_score)
    print('Model SSIM Score: %f' % model_ssim_score)

    return num_frames, mse_score, alex_lpips_scores, \
           vgg_lpips_scores, psnr_score, ssim_score, model_ssim_score

def dict2str(inputs):
    result = ''
    for key, val in inputs.items():
        if isinstance(val, dict):
            result += key + ': ' + '\n' + dict2str(val) + '\n'
        else:
            result += key + ': ' + str(val) + '\n'

    return result

if __name__ == "__main__":
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test

    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    # dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options

    config = dict()
    config['test_gpu_ids'] = opt.gpu_ids

    CUDA_VISIBLE_DEVICES = str(config['test_gpu_ids'][0])
    for gpu_index in range(1, len(config['test_gpu_ids'])):
        CUDA_VISIBLE_DEVICES += ", " + str(config['test_gpu_ids'][gpu_index])

    print('Current GPU index: ' + CUDA_VISIBLE_DEVICES)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

    enable_cuda = False
    if -1 not in config['test_gpu_ids'] and torch.cuda.is_available():  # enable cuda
        print("Using CUDA: " + str(torch.cuda.is_available()))
        enable_cuda = True

    ## Initializing the model
    alex_lpips = models.PerceptualLoss(model='net-lin', net='alex', use_gpu=enable_cuda)

    vgg_lpips = models.PerceptualLoss(model='net-lin', net='vgg', use_gpu=enable_cuda)

    model_ssim = models.PerceptualLoss(model='ssim', colorspace='RGB', use_gpu=enable_cuda)

    results_dict = dict()

    ######## Excel file ########
    file_name_xls = opt.target_name + '_' + opt.model + '_' + opt.netG + '_scores.xls'

    if os.path.exists(file_name_xls):
        workbook = xlrd.open_workbook(file_name_xls)
        workbook = copy(workbook)
    else:
        workbook = xlwt.Workbook()

    for target_name in opt.source_name:
        data_dir = os.path.join(opt.dataroot, opt.gt_data_dir)
        cur_video_names = sorted(os.listdir(data_dir))
        cur_video_names = [name for name in cur_video_names if name != '.DS_Store']
        video_names = []
        for cur_video_name in cur_video_names:
            if cur_video_name.find(target_name) != -1:
                video_names.append(cur_video_name)

        results_dict[target_name] = dict()
        results_dict[target_name]['num_frames'] = 0

        results_dict[target_name]['min_MSE'] = 0.0
        results_dict[target_name]['min_alex_LPIPS'] = 0.0
        results_dict[target_name]['min_vgg_LPIPS'] = 0.0
        results_dict[target_name]['max_PSNR'] = 0.0
        results_dict[target_name]['max_SSIM'] = 0.0
        results_dict[target_name]['max_Model_SSIM'] = 0.0

        results_dict[target_name]['min_MSE_epoch'] = 0
        results_dict[target_name]['min_alex_LPIPS_epoch'] = 0
        results_dict[target_name]['min_vgg_LPIPS_epoch'] = 0
        results_dict[target_name]['max_PSNR_epoch'] = 0
        results_dict[target_name]['max_SSIM_epoch'] = 0
        results_dict[target_name]['max_Model_SSIM_epoch'] = 0

        sheet_name_xls = target_name

        sheet = workbook.add_sheet(sheet_name_xls)

        host_name = socket.gethostname()
        value_title = ['MSE', 'Alex_LPIPS', 'VGG_LPIPS',
                        'PSNR', 'SSIM', 'Model_SSIM']

        sheet.write(0, 0, host_name)
        for j in range(len(value_title)):
            sheet.write(1, j + 1, value_title[j])

        row = 2

        syn_result_name = opt.target_name + '2' + target_name + '_' + opt.model + '_' + opt.netG + '_syn'
        syn_result_dir = os.path.join(opt.checkpoints_dir, syn_result_name)

        # for epoch in range(1, 5):
        for epoch in range(opt.epoch_start, opt.epoch_end, opt.epoch_in):
        # for epoch in range(10, 101, 10):
            epoch_result_dir = os.path.join(syn_result_dir, 'syn_epoch' + str(epoch))
            if not os.path.exists(epoch_result_dir):
                os.mkdir(epoch_result_dir)

            mse_score = 0.0
            alex_lpips_score = 0.0
            vgg_lpips_score = 0.0
            psnr_score = 0.0
            ssim_score = 0.0
            model_ssim_score = 0.0
            total_frames = 0
            for video_name in video_names:
                train_video_name = video_name.replace(target_name, opt.target_name)
                real_video_dir = os.path.join(data_dir, train_video_name)

                # real_video_dir = os.path.join(data_dir, video_name)
                syn_video_dir = os.path.join(epoch_result_dir, video_name)

                print('\n')
                print('The results of %s:' % opt.model)
                num_frames, cur_mse_score, cur_alex_lpips_score, cur_vgg_lpips_score,\
                cur_psnr_score, cur_ssim_score, cur_model_ssim_score = \
                    calc_scores(real_video_dir, syn_video_dir, alex_lpips,
                                vgg_lpips, model_ssim, enable_cuda)

                mse_score += num_frames * cur_mse_score
                alex_lpips_score += num_frames * cur_alex_lpips_score
                vgg_lpips_score += num_frames * cur_vgg_lpips_score
                psnr_score += num_frames * cur_psnr_score
                ssim_score += num_frames * cur_ssim_score
                model_ssim_score += num_frames * cur_model_ssim_score
                total_frames += num_frames

            mse_score /= float(total_frames)
            alex_lpips_score /= float(total_frames)
            vgg_lpips_score /= float(total_frames)
            psnr_score /= float(total_frames)
            ssim_score /= float(total_frames)
            model_ssim_score /= float(total_frames)

            print('\n')
            print('Toral %d Frames:' % total_frames)
            print('MSE Score: %f' % mse_score)
            print('Alex LPIPS Score: %f' % alex_lpips_score)
            print('VGG LPIPS Score: %f' % vgg_lpips_score)
            print('PSNR Score: %f' % psnr_score)
            print('SSIM Score: %f' % ssim_score)
            print('Model SSIM Score: %f' % model_ssim_score)

            epoch_str = 'Epoch ' + str(epoch)
            value_row = [epoch_str, mse_score, alex_lpips_score, vgg_lpips_score,
                         psnr_score, ssim_score, model_ssim_score]

            for j in range(len(value_row)):
                sheet.write(row, j, value_row[j])
            row += 1

            if epoch == opt.epoch_start:
                results_dict[target_name]['num_frames'] = total_frames

                results_dict[target_name]['min_MSE'] = mse_score
                results_dict[target_name]['min_MSE_epoch'] = epoch

                results_dict[target_name]['min_alex_LPIPS'] = alex_lpips_score
                results_dict[target_name]['min_alex_LPIPS_epoch'] = epoch

                results_dict[target_name]['min_vgg_LPIPS'] = vgg_lpips_score
                results_dict[target_name]['min_vgg_LPIPS_epoch'] = epoch

                results_dict[target_name]['max_PSNR'] = psnr_score
                results_dict[target_name]['max_PSNR_epoch'] = epoch

                results_dict[target_name]['max_SSIM'] = ssim_score
                results_dict[target_name]['max_SSIM_epoch'] = epoch

                results_dict[target_name]['max_Model_SSIM'] = model_ssim_score
                results_dict[target_name]['max_Model_SSIM_epoch'] = epoch

            if mse_score < results_dict[target_name]['min_MSE']:
                results_dict[target_name]['min_MSE'] = mse_score
                results_dict[target_name]['min_MSE_epoch'] = epoch

            if alex_lpips_score < results_dict[target_name]['min_alex_LPIPS']:
                results_dict[target_name]['min_alex_LPIPS'] = alex_lpips_score
                results_dict[target_name]['min_alex_LPIPS_epoch'] = epoch

            if vgg_lpips_score < results_dict[target_name]['min_vgg_LPIPS']:
                results_dict[target_name]['min_vgg_LPIPS'] = vgg_lpips_score
                results_dict[target_name]['min_vgg_LPIPS_epoch'] = epoch

            if psnr_score > results_dict[target_name]['max_PSNR']:
                results_dict[target_name]['max_PSNR'] = psnr_score
                results_dict[target_name]['max_PSNR_epoch'] = epoch

            if ssim_score > results_dict[target_name]['max_SSIM']:
                results_dict[target_name]['max_SSIM'] = ssim_score
                results_dict[target_name]['max_SSIM_epoch'] = epoch

            if model_ssim_score > results_dict[target_name]['max_Model_SSIM']:
                results_dict[target_name]['max_Model_SSIM'] = model_ssim_score
                results_dict[target_name]['max_Model_SSIM_epoch'] = epoch

        attr_str = 'Best results'
        value_row = [attr_str, results_dict[target_name]['min_MSE'],
                     results_dict[target_name]['min_alex_LPIPS'],
                     results_dict[target_name]['min_vgg_LPIPS'],
                     results_dict[target_name]['max_PSNR'],
                     results_dict[target_name]['max_SSIM'],
                     results_dict[target_name]['max_Model_SSIM']]

        for j in range(len(value_row)):
            sheet.write(row, j, value_row[j])

        #######################
        results_str = dict2str(results_dict)

        print('\n')
        print(results_str)

        workbook.save(file_name_xls)
        print('\n')
        print('Save results to %s' % file_name_xls)

        ######### Calculate Transfer Videos ############
        # transfer_opt = config['transfer_opt']
        # real_videos_dir = os.path.join(transfer_opt['root_dir'], transfer_opt['data_dir'])
        # videos_names = sorted(os.listdir(real_videos_dir))
        # videos_names = [name for name in videos_names if name != '.DS_Store']
        #
        # for video_name in videos_names:
        #     real_video_dir = os.path.join(real_videos_dir, video_name)
        #
        #     syn_video_dir_name = video_name + '_syn_epoch' + str(epoch)
        #     syn_video_dir = os.path.join(config['syn_exp_dir'], syn_video_dir_name)
        #     calc_scores(real_video_dir, syn_video_dir, dis_model, enable_cuda)


