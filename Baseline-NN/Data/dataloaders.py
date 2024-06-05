from __future__ import print_function

import torch
import torch.utils.data as data
from Data.datasets import TargetDataset, SupportDataset,\
    SourceDataset, MultiTargetDataset, TransferDataset,\
    UnitedTargetDataset, UnitedSourceDataset

def CreateDataset(opt):
    data_name = opt['data_name']
    dataset = None
    if data_name == 'united_source':
        dataset = UnitedSourceDataset(opt)
        print("united source dataset [%s] was created" % dataset.target_name)
    else:
        raise ValueError('Not supported or recognized data_name', data_name)
    return dataset

class DataloaderFactory(object):
    def __init__(self, opt):
        self.dataset = CreateDataset(opt)
        self.phase = self.dataset.phase
        self.batch_size = opt['batch_size']
        self.num_workers = opt['num_workers']
        self.is_eval_mode = (self.phase=='test')

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=(False if self.is_eval_mode else True),
            drop_last=True,
            num_workers=int(self.num_workers))

    def load_data(self):
        return self.dataset

    def load_dataloader(self):
        return self.dataloader