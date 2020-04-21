"""IMU Base dataset"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

import numpy as np
import pandas
from torch.utils.data import Dataset
import ipdb

class IMUBaseDataset(Dataset):
    """Base dataset to inherit from"""

    def __init__(self, mode, datalst_pth, data_root, lbl_fname, lbl_type,
                 seq_len=-1, transform = None):
        """Initialize the dataset

        Args:
            mode: `train`, `val`, or `test` mode
            datalst_pth: a dictionary of paths wrt to different modes
            data_root: root directory of the dataset
            lbl_fname: path to the label file
            lbl_type: type of the label to retrieve:
                `on_off`, `dyskinesia`, `tremor`, or `all` (for everything)
            seq_len: the samples will be padded/cropped to this length.
                -1 to keep the original length
            transform: transform object to apply random data augmentation
        """
        assert mode in ['train', 'val', 'test'], 'Unsupported mode: {}'.format(mode)
        self.mode = mode
        self.datalst_pth = datalst_pth[mode]
        assert os.path.isfile(self.datalst_pth), 'Not found: {}'.format(self.datalst_pth)
        self.datalst = open(self.datalst_pth, 'r').read().splitlines()

        assert os.path.isdir(data_root), 'Not found: {}'.format(data_root)
        self.data_root = data_root

        assert os.path.isfile(lbl_fname), 'Not found: {}'.format(lbl_fname)
        self.label_dict = pandas.read_csv(lbl_fname)

        assert lbl_type in ['on_off', 'dyskinesia', 'tremor', 'all'], \
            'Unsupported label type: {}'.format(lbl_type)
        self.lbl_type = lbl_type

        assert (seq_len == -1) or (seq_len > 0), \
            'Bad seq_len: {}'.format(seq_len)
        self.seq_len = seq_len


        self.transform = transform

    def __len__(self):
        return len(self.datalst)

    def get_data_sample(self, msr_id):
        data = pandas.read_csv(os.path.join(self.data_root, msr_id+'.csv'))
        # Convert data to numpy
        data = data.to_numpy().astype(np.float32)
        return data
    
    def crop_pad_data(self, data):
        true_len = len(data)
        if self.seq_len > 0:
            if self.seq_len < true_len:
                data = data[:self.seq_len]
            if self.seq_len > true_len:
                delta = self.seq_len - true_len
                data = np.pad(data, ((0, delta), (0, 0)), 'edge')
        return data

    def __getitem__(self, idx):
        """Get item wrt a given index

        Args:
            idx: sample index

        Returns:
            data: (numpy array of shape (N, 4)) t, x, y, z mesurements
            lbl: (float) label
        """
        # Retrieve measurement ID wrt to the given index
        msr_id = self.datalst[idx]

        # Read data and labels correponding to the measurement ID
        data = self.get_data_sample(msr_id)
        
        # Crop/pad the data if required
        data = self.crop_pad_data(data)

        # If transform is no none, apply data augmentation strategy
        if self.transform:
            data = self.transform(data)     

        # Test mode does not have labels
        if (self.mode == 'test'):
            lbl = -1
        else:
            lbl = self.label_dict.loc[self.label_dict['measurement_id'] == msr_id]
            
            # Retrieve the correct label wrt the given lbl_type
            if self.lbl_type == 'all':
                lbl = lbl.to_numpy()[0][2:]
                lbl = lbl.astype(np.float32)
            else:
                lbl = float(lbl[self.lbl_type])
                lbl = np.array(lbl).astype(np.float32)

        return data, lbl, msr_id

    def get_subject_id(self, msr_id):
        """Get subject id from measurement id (observation)

        Args:
            msr_id: measurement id

        Return:
            subject id as string
        """
        row = self.label_dict[self.label_dict['measurement_id'] == msr_id]
        subject_id = row['subject_id'].values[0]
        return str(subject_id)

    def get_subject_id_list(self, msr_id_lst):
        """Get a list of subject id from a list of measurement ids

        Args:
            msr_id_lst: list of measurement ids

        Return:
            list of subject ids as strings
        """
        return [self.get_subject_id(msr_id) for msr_id in msr_id_lst]
