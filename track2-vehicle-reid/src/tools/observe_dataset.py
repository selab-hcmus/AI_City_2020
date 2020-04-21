"""Observe dataset as raw time series and log-scale spectrogram
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
from scipy.signal import stft
from scipy.ndimage import gaussian_filter
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
        '-n', '--n_samples', type=int,
        help='Number of observation to visualize')
    parser.add_argument(
        '--min_band', type=float, default=None,
        help='Minimum frequency of the frequency band to plot (in Hz)')
    parser.add_argument(
        '--max_band', type=float, default=None,
        help='Maximum frequency of the frequency band to plot (in Hz)')
    parser.add_argument(
        '-o', '--out_dir', type=str,
        help='Output directory to store the visualizations')

    args = parser.parse_args()

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    if args.min_band is not None:
        assert args.max_band is not None, 'Both mix_band and max_band must be given'
    if args.max_band is not None:
        assert args.min_band is not None, 'Both mix_band and max_band must be given'
    return args


def parse_labels(lbls, lbl_type):
    """Parse the labels for figure title

    Args:
        lbls: numpy array if lbl_type is `all`, otherwise a floating point
        lbl_type: label type
    """
    if lbl_type != 'all':
        return '{} = {}'.format(lbl_type[0], lbls)
    return 'on_off = {}, dyskinesia = {}, tremor = {}'.format(
        lbls[0], lbls[1], lbls[2])


def viz_time_series(samples, ax):
    """Plot the time series measurement

    Args:
        samples: (N, 4) numpy array where the columns are (t, x, y, z) data
        ax: axis handle
    """
    names = ['t', 'x', 'y', 'z']
    for j in range(1, 4):
        ax.plot(samples[:, j], alpha=0.8, label=names[j])

    ax.set_title('Time sequence')
    ax.set_xlabel('n_samples')
    ax.set_ylabel('m/s^2')
    ax.legend()
    ax.set_xlim(0, samples.shape[0])


def viz_gradient(samples, ax, sigma=10):
    """Plot the gradient of the measurements

    Args:
        samples: (N, 4) numpy array where the columns are (t, x, y, z) data
        ax: axis handle
        sigma: standard deviation of Gaussian filter
    """
    names = ['t', 'x', 'y', 'z']
    for j in range(1, 4):
        grad = (np.abs(np.gradient(gaussian_filter(samples[:, j], sigma=sigma))))
        ax.plot(grad, alpha=0.8, label='Gradient '+names[j])

    ax.set_title('Gradient')
    ax.set_xlabel('n_samples')
    ax.set_ylabel('m/s^3')
    ax.legend()
    ax.set_xlim(0, samples.shape[0])


def viz_spectrogram(samples, sampling_freq, axes, fig,
                    min_band=None, max_band=None):
    """Plot the spectrogram in log scale using STFT

    Args:
        samples: (N, 4) numpy array where the columns are (t, x, y, z) data
        sampling_freq: sampling frequency of samples
        axes: list of axis handles corresponding to x, y, and z
        fig: figure handle
        min_band: minimum frequency of the frequency band to plot (in Hz)
        max_band: maximum frequency of the frequency band to plot (in Hz)
    """
    names = ['t', 'x', 'y', 'z']
    for j in range(1, 4):
        stft_freq, stft_time, stft_resp = stft(samples[:, j], fs=sampling_freq)
        stft_resp = (np.log(np.abs(stft_resp)))  # log scale of real part

        # Zooming a specific range of frequency if specified
        if (min_band is not None) and (max_band is not None):
            min_idx = int(min_band / sampling_freq * 2 * len(stft_freq))
            max_idx = int(max_band / sampling_freq * 2 * len(stft_freq))
            stft_freq = stft_freq[min_idx: max_idx+1]
            stft_resp = stft_resp[min_idx: max_idx+1, :]

        # im_handl = axes[j+1].imshow(stft_resp, origin='lower', aspect='auto')
        # fig.colorbar(im_handl, ax=axes[j+1])
        im_handl = axes[j-1].pcolormesh(stft_time, stft_freq, stft_resp)
        fig.colorbar(im_handl, ax=axes[j-1])
        axes[j-1].grid(False)

        axes[j-1].set_title('Log-scale STFT - {}-axis'.format(names[j]))
        axes[j-1].set_xlabel('seconds')
        axes[j-1].set_ylabel('Hz')


def viz(loader, args):
    """Visualize data

    Args:
        loader: initialized data loader
        args: input arguments
    """
    # Prepare the environments
    img_dir = os.path.join(args.out_dir, 'imgs')
    if not os.path.isdir(img_dir):
        os.makedirs(img_dir)
    out_fname = os.path.join(args.out_dir, 'viz.html')
    fout = open(out_fname, 'w')
    fmt = '<img src="{}"></img><br>'
    pbar = MiscUtils.gen_pbar(max_value=args.n_samples, msg='Plotting: ')

    # Loop through n_samples and plot
    for i, (samples, labels, msr_ids) in enumerate(loader):
        # Retrieve data
        samples = samples.numpy().squeeze()
        labels = labels.numpy().squeeze()
        n_samples = len(samples)
        duration = samples[-1, 0] - samples[0, 0]
        sampling_freq = 1. * n_samples / duration

        # Plot the figure
        fig, axes = plt.subplots(5, 1, figsize=(12, 12), constrained_layout=True)

        viz_time_series(samples, axes[0])
        viz_gradient(samples, axes[1])
        viz_spectrogram(samples, sampling_freq, axes[2:5], fig,
                        args.min_band, args.max_band)
        fig.suptitle(parse_labels(labels, args.lbl_type))

        # Save the figure
        img_fname = os.path.join(img_dir, msr_ids[0]+'.png')
        fig.savefig(img_fname)
        plt.close(fig)

        # Add to html
        fout.write(msr_ids[0] + '<br>')
        fout.write('sampling frequency = {} <br>'.format(sampling_freq))
        img_fname = img_fname.replace(args.out_dir, '.')
        fout.write(fmt.format(img_fname))
        fout.write('<hr>')

        # Update progress
        pbar.update(i+1)
        if i >= args.n_samples-1:
            break
    pbar.finish()

    # Close the html file
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
        dataset, shuffle=True, drop_last=True, batch_size=1, num_workers=8)

    # Vizualize
    sns.set()
    viz(loader, args)

    return 0


if __name__ == '__main__':
    sys.exit(main())
