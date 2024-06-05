from torch.nn import CrossEntropyLoss, BCELoss
import os

config = dict()

config['gpu_ids'] = [0] #default='-1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
config['test_gpu_ids'] = [0]

config['train_interval'] = 10
config['phase'] = 'train' #'syn' or 'train'

config['exp_dir'] = 'subject5'
config['syn_exp_dir'] = 'subject5'

config['enable_pafs'] = True
config['img_type'] = 'fixmid_1000' #'fixsize', 'ori'

config['pose_img_loss'] = False
config['in_static_encoder'] = 1
config['in_dyc_encoder'] = 1

##################################
transform_optionsF = dict()
transform_optionsF['scale_type'] = 'resize'
transform_optionsF['load_width'] = 256#256#1024#512
transform_optionsF['load_height'] = 256#128#512#288
transform_optionsF['normalize'] = True
transform_optionsF['toTensor'] = True

############ train opt ################
train_opt = dict()
train_opt['data_name'] = 'united_source'
train_opt['target_names'] = ['subject5']
train_opt['enable_pose_img'] = config['pose_img_loss']

train_opt['root_dir'] = '../Motion-Datasets/EDN-subs'
if config['img_type'] == 'fixmid_1000':
    train_opt['data_dir'] = 'fixmid_train_1000'
    train_opt['pose_data_dir'] = 'fixmid_train_1000'

train_opt['num_clip_len'] = config['in_dyc_encoder']
train_opt['phase'] = 'test'
train_opt['batch_size'] = 1
train_opt['num_workers'] = 0
train_opt['transform_params'] = transform_optionsF
config['train_opt'] = train_opt

############ test opt ################
test_opt = dict()
test_opt['data_name'] = 'united_source'
test_opt['enable_pose_img'] = config['pose_img_loss']
test_opt['target_names'] = ['subject5']

test_opt['root_dir'] = '../Motion-Datasets/EDN-subs'
if config['img_type'] == 'fixmid_1000':
    test_opt['data_dir'] = 'fixmid_test_1000'
    test_opt['pose_data_dir'] = 'fixmid_test_1000'

test_opt['num_clip_len'] = config['in_dyc_encoder']
test_opt['phase'] = 'test'
test_opt['batch_size'] = 1
test_opt['num_workers'] = 0
test_opt['transform_params'] = transform_optionsF
config['test_opt'] = test_opt

###############################
networks = dict()
optimizers = dict()

########### OpenPose ################
net_optionsF = dict()
net_optionsF['requires_grad'] = False

networks['openpose'] = dict()
networks['openpose']['net_name'] = 'openpose'
networks['openpose']['net_params'] = net_optionsF
networks['openpose']['pretrained_path'] = os.path.join('Openpose',
                                                       'network',
                                                       'pose_model.pth')

############# Criterions ###################
criterions = dict()

config['criterions'] = criterions
config['networks'] = networks
config['optimizers'] = optimizers



