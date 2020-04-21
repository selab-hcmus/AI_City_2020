"""AICity20 VehicleType dataset"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import os
import xml.etree.ElementTree as ET
import numpy as np
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import pandas as pd
from src.datasets.img_base_dataset import ImageBaseDataset
import ipdb
class AIC20_VEHI_REID(ImageBaseDataset):
    """AICity20 Vehicle Re-ID dataset"""
    def get_support_modes(self):
        """This funnnction return support modes for the current dataset
        """
        return ['train', 'gallery', 'query']

    def get_all_labels(self):
        """This function constructs a dictionary of image's label. 
        Each element should looks like: 
        lbl_dict[str(`0000123')] = 1 """
        
        # xml_train_lbl = ET.parse(lbl_fname, parser=ET.XMLParser(encoding='iso-8859-5'))
        # root = xml_train_lbl.getroot()
        lbl_dict = {}
        self.ist2idx = {} 
        c = 0
        img2id = {}
        df = pd.read_csv(self.datalst_pth)  
        # Convert instance to instance index: e.g. 1->453 (333 insts) => 0 -> 332
        # for child in root.iter("Item"):
        #     imgId = child.attrib["imageName"]
        #     vehId = int(child.attrib["vehicleID"])
        #     img2id[imgId.split('.')[0]] = vehId
        for i, imgName in enumerate(df["image_id"]):
            vehId = int(df["vehicle_id"][i])
            img2id[imgName.split('.')[0]] = vehId
        for imgId in self.imglst:   
            vehId = img2id[imgId]
            if vehId not in self.ist2idx:
                self.ist2idx[vehId] = c
                c+=1
            lbl_dict[imgId] = self.ist2idx[vehId]
        return lbl_dict

    
    def get_img_list(self):
        df = pd.read_csv(self.datalst_pth) 
        print(self.datalst_pth)
        return df["image_id"].to_list()


    def get_data_label(self, idx):
        # Test mode does not have label
        if (self.mode == 'test'):
            return -1
        img_id = self.imglst[idx]
        return self.label_dict[img_id]

    def get_nclasses(self):
        """This function returns the number of vehicle instances"""
        return len(self.ist2idx)
    
    def get_list_of_labels(self):
        """This function returns all labels"""
        return np.array([self.get_data_label(idx) for idx in range(len(self.img_fname_lst))])
    
    def get_unique_labels(self):
        """This funnction returns uniques labels only"""
        return np.arange(0, self.get_nclasses())
    
    def get_inst2imgs_dict(self):
        """This function returns a dictionary with the following structure:
            inst2imgs[<instance ID>] = list([img_idx1, img_idx2, ...])
        """
        self.inst2imgs = {}
        for idx in range(len(self.img_fname_lst)):
            lbl = self.get_data_label(idx)
            if lbl not in self.inst2imgs:
                self.inst2imgs[lbl] = []
            self.inst2imgs[lbl].append(idx)
        return self.inst2imgs
    
    def get_img_names(self):
        """This function returns a list of image names (str):
            e.g: '000123', '000234'
        """
        return self.imglst