"""Check the distribution of the datasets
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
        '-e', '--exp_name', type=str, default=None,
        help='Name of the experiment')
    parser.add_argument(
        '-o', '--out_dir', type=str,
        help='Output directory to store the visualizations')
    parser.add_argument(
        '-a', '--append', action='store_true',
        help='Whether to append the output file or not')

    args = parser.parse_args()

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    if not os.path.isdir(os.path.join(args.out_dir, 'imgs')):
        os.makedirs(os.path.join(args.out_dir, 'imgs'))

    # Generate the exp_name as current time stamp if not provided
    if args.exp_name is None:
        args.exp_name = str(time.time())

    return args


def viz_length_distribution(all_n_samples, all_duration, args):
    """Plot length ditribution

    Args:
        all_n_samples: list of number of samples
        all_duration: list of duration of samples (in seconds)
        args: input arguments

    Return:
        img_fname: relative path to image filename to write to html file
    """
    fig, axes = plt.subplots(3, 1, figsize=(15, 7))
    msgs = [''] * 3

    axes[0].hist(all_n_samples, bins=100)
    msgs[0] = 'n_samples\nmean={:.3f}\nstd={:.3f}'.format(
        all_n_samples.mean(), all_n_samples.std())

    axes[1].hist(all_duration, bins=100)
    msgs[1] = 'seconds\nmean={:.3f}\nstd={:.3f}'.format(
        all_duration.mean(), all_duration.std())

    all_sampling_rates = 1. * all_n_samples / all_duration
    axes[2].hist(all_sampling_rates, bins=100)
    msgs[2] = 'Hz\nmean={:.3f}\nstd={:.3f}'.format(
        all_sampling_rates.mean(), all_sampling_rates.std())

    for i in range(3):
        axes[i].set_ylabel(msgs[i])

    fig.suptitle('Distribution of sample length')
    img_fname = os.path.join(args.out_dir, 'imgs', '{}_len.png'.format(args.exp_name))
    fig.savefig(img_fname)
    return img_fname.replace(args.out_dir, '.')


def viz_label_distribution(all_labels, args):
    """Plot label distribution

    Args:
        all_labels: list of labels as percentages
        args: input arguments

    Return:
        img_fname: relative path to image filename to write to html file
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    names = ['on_off', 'dyskinesia', 'tremor']
    for i in range(3):
        axes[i].pie(
            all_labels[i], labels=[0, 1, 2, 3, 4, 'NaN'], autopct='%1.0f%%',
            shadow=False, startangle=0, pctdistance=0.7, labeldistance=1.1)
        axes[i].set_title(names[i])

    fig.suptitle('Distribution of labels')
    img_fname = os.path.join(args.out_dir, 'imgs', '{}_lbl.png'.format(args.exp_name))
    fig.savefig(img_fname)
    return img_fname.replace(args.out_dir, '.')


def check_distribution(loader, args):
    """Check data distribution

    Args:
        loader: initialized data loader
        args: input arguments
    """
    # Allocate memory
    all_n_samples = np.zeros(len(loader))
    all_duration = np.zeros(len(loader))
    all_labels = np.zeros([3, 6], dtype=np.int)

    # Prepare environment
    if args.append:
        fout = open(os.path.join(args.out_dir, 'viz.html'), 'a')
    else:
        fout = open(os.path.join(args.out_dir, 'viz.html'), 'w')

    fmt = '<img src="{}"></img><br>\n'
    pbar = MiscUtils.gen_pbar(
        max_value=len(loader), msg='Scanning {}: '.format(args.exp_name))

    # Scan through dataset
    for i, (samples, labels, msr_ids) in enumerate(loader):
        # Retrieve data
        samples = samples.numpy().squeeze()
        labels = labels.numpy().squeeze()

        # Collect length information
        all_n_samples[i] = samples.shape[0]
        all_duration[i] = samples[-1, 0] - samples[0, 0]

        # Collect labels information
        for j in range(3):
            if np.isnan(labels[j]):
                all_labels[j, -1] += 1
            else:
                all_labels[j, int(labels[j])] += 1

        # Update progress
        pbar.update(i+1)
    pbar.finish()

    all_labels = all_labels / len(loader)

    # Write to html file
    fout.write('{}<br>\n'.format(args.exp_name))

    img_fname = viz_length_distribution(all_n_samples, all_duration, args)
    fout.write(fmt.format(img_fname))

    img_fname = viz_label_distribution(all_labels, args)
    fout.write(fmt.format(img_fname))

    fout.write('<hr>\n')

    fout.close()


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
        dataset, shuffle=False, drop_last=True, batch_size=1, num_workers=8)

    # Check distribution
    sns.set()
    check_distribution(loader, args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
