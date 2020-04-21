"""Make data list that contains no NaN labels
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from torch.utils.data import DataLoader

from src.utils.load_cfg import ConfigLoader
from src.factories import DatasetFactory
from src.utils.misc import MiscUtils


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-d', '--dataset_cfg', type=str,
        help='Path to the dataset config filename')
    parser.add_argument(
        '-m', '--mode', type=str,
        choices=['train', 'val', 'test'],
        help='mode to load the dataset')
    parser.add_argument(
        '-t', '--lbl_type', type=str, default='all',
        choices=['on_off', 'dyskinesia', 'tremor', 'all'],
        help='Type of label to load to observe. `all` means all of them')
    parser.add_argument(
        '-o', '--output_pth', type=str, default=None,
        help='Path to the output file')

    args = parser.parse_args()
    assert args.output_pth is not None, 'No output_pth given'

    return args


def make_nonan(loader, output_pth):
    """
    """
    pbar = MiscUtils.gen_pbar(
        max_value=len(loader), msg='Scanning {}: ')

    # Scan through dataset
    nonan_lst = []
    for i, (_, labels, msr_ids) in enumerate(loader):
        # Retrieve data
        # samples = samples.numpy().squeeze()
        labels = labels.numpy().squeeze()

        if np.isnan(labels):
            continue

        nonan_lst.append(msr_ids[0])

        # Update progress
        pbar.update(i+1)
    pbar.finish()

    with open(output_pth, 'w') as fout:
        fout.write('\n'.join(nonan_lst))


def main():
    """Main function"""
    # Load input arguments
    args = parse_args()

    # Create dataset and data loader
    dataset_name, dataset_params = ConfigLoader.load_dataset_cfg(args.dataset_cfg)

    dataset_factory = DatasetFactory()
    dataset = dataset_factory.generate(
        dataset_name, mode=args.mode, lbl_type=args.lbl_type, **dataset_params)

    loader = DataLoader(
        dataset, shuffle=False, drop_last=False, batch_size=1, num_workers=8)

    make_nonan(loader, args.output_pth)
    return 0


if __name__ == '__main__':
    sys.exit(main())
