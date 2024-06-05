### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import numpy as np

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        if opt.phase == 'train':
            self.per_name = opt.target_name
        else:
            self.per_name = opt.source_name

        ### label maps    
        # self.dir_label = os.path.join(opt.dataroot, opt.phase + '_label')           
        self.dir_label = os.path.join(opt.dataroot, opt.input_data_dir, self.per_name)
        self.label_paths = sorted(make_dataset(self.dir_label))

        ### real images
        if opt.isTrain:
            # self.dir_image = os.path.join(opt.dataroot, opt.phase + '_img')  
            self.dir_image = os.path.join(opt.dataroot, opt.gt_data_dir, self.per_name)    
            self.image_paths = sorted(make_dataset(self.dir_image))

        ### load face bounding box coordinates size 128x128
        if opt.face_discrim or opt.face_generator:
            self.dir_facetext = os.path.join(opt.dataroot, opt.phase + '_facetexts128')
            print('----------- loading face bounding boxes from %s ----------' % self.dir_facetext)
            self.facetext_paths = sorted(make_dataset(self.dir_facetext))


        self.dataset_size = len(self.label_paths) 
      
    def __getitem__(self, index):        
        ### label maps
        # index = len(self) - 1
        paths = self.label_paths
        label_path = paths[index]              
        label = Image.open(label_path).convert('RGB')        
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label)
        original_label_path = label_path

        # image_tensor = next_label = next_image = face_tensor = 0
        image_tensor = next_label = next_image = face_tensor = torch.zeros(label_tensor.size())
        # face_tensor = torch.tensor(face_tensor)
        ### real images 
        if self.opt.isTrain:
            image_path = self.image_paths[index]   
            image = Image.open(image_path).convert('RGB')    
            transform_image = get_transform(self.opt, params)     
            image_tensor = transform_image(image).float()

        is_next = index < len(self) - 1
        if self.opt.gestures:
            is_next = is_next and (index % 64 != 63)

        """ Load the next label, image pair """
        if is_next:

            paths = self.label_paths
            label_path = paths[index+1]              
            label = Image.open(label_path).convert('RGB')        
            params = get_params(self.opt, label.size)          
            transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            next_label = transform_label(label).float()
            
            if self.opt.isTrain:
                image_path = self.image_paths[index+1]   
                image = Image.open(image_path).convert('RGB')
                transform_image = get_transform(self.opt, params)      
                next_image = transform_image(image).float()

        """ If using the face generator and/or face discriminator """
        if self.opt.face_discrim or self.opt.face_generator:
            facetxt_path = self.facetext_paths[index]
            facetxt = open(facetxt_path, "r")
            face_tensor = torch.IntTensor(list([int(coord_str) for coord_str in facetxt.read().split()]))

        # numel_1 = label_tensor.float().numel()
        # numel_2 = image_tensor.numel()
        # numel_3 = face_tensor.numel()
        # numel_4 = next_label.numel()
        # numel_5 = next_image.numel()
        input_dict = {'label': label_tensor.float(), 'image': image_tensor, 
                      'path': original_label_path, 'face_coords': face_tensor.float(),
                      'next_label': next_label.float(), 'next_image': next_image.float() }
        return input_dict

    def __len__(self):
        return len(self.label_paths)

    def name(self):
        return 'AlignedDataset'