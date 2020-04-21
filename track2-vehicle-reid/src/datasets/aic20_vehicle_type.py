"""AICity20 VehicleType dataset"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from src.datasets.img_base_dataset import ImageBaseDataset

class AIC20_VEHI_TYPE(ImageBaseDataset):
    """AICity20 VehicleType dataset"""
    def get_all_img_fname(self):
        super().get_all_img_fname()
        self.img_fname_lst = [
            str(img).zfill(6) + ".jpg"
            for img in self.img_fname_lst
        ]