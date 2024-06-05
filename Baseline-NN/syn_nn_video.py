from __future__ import print_function
import argparse
import os
import torch
from importlib import util
from Data.dataloaders import DataloaderFactory
from Models.experiments_trainer import Experiments_Trainer
import copy

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, default='11',
                        help='config file with parameters of the experiment.')
    args_opt = parser.parse_args()

    exp_config_file = os.path.join('.', 'Config', args_opt.config + '.py')
    # exp_directory = os.path.join('.', 'Experiments', args_opt.config)
    # print('Launching experiment: %s' % exp_config_file)

    # Load the configuration params of the experiment
    config_module_child_name = args_opt.config
    config_spec = util.spec_from_file_location(config_module_child_name, exp_config_file)
    config_module = util.module_from_spec(config_spec)
    config_spec.loader.exec_module(config_module)
    config = config_module.config

    # the place where logs, models, and other stuff will be stored
    config['exp_dir'] = os.path.join('.', 'Experiments', config['exp_dir'])

    print("Loading experiment %s from file: %s" % (args_opt.config, exp_config_file))
    print("Generated logs, snapshots, and model files will be stored on %s" % (config['exp_dir']))

    return config

if __name__ == "__main__":
    config = load_config()
    # config['test_gpu_ids'][0] = 3

    CUDA_VISIBLE_DEVICES = str(config['test_gpu_ids'][0])
    for gpu_index in range(1, len(config['test_gpu_ids'])):
        CUDA_VISIBLE_DEVICES += ", " + str(config['test_gpu_ids'][gpu_index])

    print('Current GPU index: ' + CUDA_VISIBLE_DEVICES)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

    train_opt = config['train_opt']
    train_opt['phase'] = 'test'
    train_opt['batch_size'] = 1

    syn_transfer = False

    config['phase'] = 'syn'

    #print('1--' + str(torch.cuda.memory_allocated(device=torch.cuda.current_device()) / 1048576))

    print('\n')
    print('Syn Videos for Baseline-NN: ')

    config['exp_dir'] = config['syn_exp_dir']
    config['exp_dir'] = os.path.join('.', 'Experiments', config['exp_dir'])

    experiments_trainer = Experiments_Trainer(config)
    # train_dataloader = DataloaderFactory(train_opt).load_dataloader()
    train_opt['target_names'] = train_opt['target_names'][0]
    train_dataloader = DataloaderFactory(train_opt).load_dataloader()

    test_opt = config['test_opt']
    target_names = copy.deepcopy(test_opt['target_names'])
    for target_name in target_names:
        test_opt['target_names'] = target_name

        test_dataloader = DataloaderFactory(test_opt).load_dataloader()
        #print('2--' + str(torch.cuda.memory_allocated(device=torch.cuda.current_device()) / 1048576))

        experiments_trainer.synVideo(train_dataloader, test_dataloader, target_name)

    test_opt['target_names'] = target_names


