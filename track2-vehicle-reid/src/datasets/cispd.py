"""CIS-PD dataset"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from src.datasets.imu_base_dataset import IMUBaseDataset
import numpy as np

class CispdDataset(IMUBaseDataset):
    """CIS-PD dataset"""
    def get_data_sample(self, msr_id):
        data = np.load(os.path.join(self.data_root, msr_id+'.npy')).astype(np.float32)
        return data