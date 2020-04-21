import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from src.factories import DataAugmentationFactory, DataSamplerFactory, DatasetFactory
import ipdb

class BaseDataLoaderFactory():
    def __init__(self, dataset_name, dataset_params, train_params, base_loader_params):
        # Copy some parameters
        self.dataset_name = dataset_name
        self.dataset_params = dataset_params
        self.train_params = train_params
        self.base_loader_params = base_loader_params
        # Generate factories
        self.data_augment_fact = DataAugmentationFactory()
        self.data_fact = DatasetFactory()
        self.sampler_fact = DataSamplerFactory()
    
        # Init
        self.ld_dict = {'train_val': {}, 'test': {}}
        self.get_data_split()
        self.build_base_parameters()
        self.gen_data_augmentation()
        self.gen_data_sampler()

        self.fix_dataset_params('datalst_pth')
        self.fix_dataset_params('spldata_dir')

    def get_data_split(self):
        self.ld_dict['train_val'] = {k:{} for k in self.dataset_params['datalst_pth']['train_val'].keys()}
        self.ld_dict['test'] = {k:{} for k in self.dataset_params['datalst_pth']['test'].keys()}

    def set_all_train_val(self, key, val):
        for k in self.ld_dict['train_val']:
            self.ld_dict['train_val'][k][key] = val

    def set_all_test(self, key, val):
        for k in self.ld_dict['test']:
            self.ld_dict['test'][k][key] = val 

    def set_all(self, key, val):
        self.set_all_train_val(key,val)
        self.set_all_test(key,val)

    def build_base_parameters(self):
        self.set_all("ld_params", dict(self.base_loader_params))
        self.set_all("transform", None)
        self.set_all("sampler", None)
  
    def gen_data_augmentation(self):
        if 'transforms' not in self.train_params:
            return
        else:
            #base transform for all sets
            if ("base" in self.train_params['transforms']):
                self.set_all("transform", 
                    self.train_params['transforms']['base'])
            for targ in self.train_params['transforms']:
                if targ == "base":
                    continue
                else:
                    trsf = self.train_params['transforms'][targ]
                    tm = targ.split('/')
                    self.ld_dict[tm[0]][tm[1]]['transform'] = \
                    {
                        # **self.ld_dict[tm[0]][tm[1]]['transform'],
                        **trsf
                    }
            
    def gen_data_sampler(self):
        if 'samplers' not in self.train_params:
            return
        else:
            for targ in self.train_params['samplers']:
                tm = targ.split('/')
                self.ld_dict[tm[0]][tm[1]]['sampler'] = \
                        self.train_params['samplers'][targ] #train_val/train
    
    def fix_dataset_params(self,dict_key):
        new_dict = {}
        data_dict = self.dataset_params[dict_key]
        for group in data_dict.keys():
            for mem in data_dict[group]:
                new_key = group + "/" + mem
                new_dict[new_key] = data_dict[group][mem]
        self.dataset_params[dict_key] = new_dict

    def get_sampler(self, sampler_config, dataset):
        if (sampler_config == None):
            return None
        sampler_params = list(sampler_config.values())[0]
        sampler_params['dataset'] = dataset
        return self.sampler_fact.generate(
            list(sampler_config.keys())[0],
            **sampler_params
        )
    def get_transform(self, transform_config):
        composed_transforms = transforms.Compose([
        self.data_augment_fact.generate(
            i, transform_config[i])
        for i in transform_config.keys()])
        return composed_transforms

    def build_group_loaders(self, group):
        return_dict = {}
        for mem in self.ld_dict[group]:
            mode = group + "/" + mem
            _ld_dict = self.ld_dict[group][mem]
            #Set shuffle and drop_last
            _shuffle, _drop_last = False, False
            if mem in ['train']:
                _shuffle, _drop_last = True, True
            
            #Create transform
            _transform = self.get_transform(_ld_dict['transform'])
            self.dataset_params["transform"] = _transform
            
            # Create dataset
            _dataset = self.data_fact.generate(
            self.dataset_name, mode=mode, **self.dataset_params)

            #Create sampler
            _sampler = self.get_sampler(_ld_dict['sampler'], _dataset)

            if (_sampler is not None):
                _loader = DataLoader(_dataset, batch_sampler=_sampler, 
                num_workers= _ld_dict["ld_params"]['num_workers'])
            else:
                _loader = DataLoader(_dataset, shuffle=_shuffle, drop_last=_drop_last, 
                **_ld_dict["ld_params"])
            return_dict[mem] = _loader
        return return_dict
    
    def train_val_loaders(self):
        return self.build_group_loaders('train_val')
    
    def test_loaders(self):
        return self.build_group_loaders('test')

            
        
                

    
        



