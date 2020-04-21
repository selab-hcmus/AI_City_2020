"""Misc utilities as static methods"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import glob

import progressbar
import torch
import src.utils.logging as logging

logger = logging.get_logger(__name__)


class MiscUtils:
    @staticmethod
    def save_progress(model, optimizer, logdir, epoch):
        """Save the training progress for model and optimizer

        Data are saved as: [logdir]/epoch_[epoch].[extension]

        Args:
            model: model to save
            optimizer: optimizer to save
            logdir: where to save data
            epoch: the current epoch
        """
        prefix = os.path.join(logdir, 'epoch_{:05d}'.format(epoch))
        logger.info('Saving to: %s' % prefix)

        model.save_model(prefix+'.model')
        torch.save(optimizer.state_dict(), prefix+'.opt')
        torch.save(torch.get_rng_state(), prefix+'.rng')
        torch.save(torch.cuda.get_rng_state(), prefix+'.curng')

    @staticmethod
    def load_progress(model, optimizer, prefix):
        """Load the training progress for model and optimizer

        Data are loaded from: [prefix].[extension]

        Args:
            model: model to load
            optimizer: optimizer to load
            prefix: prefix with the format [logdir]/epoch_[epoch]

        Return:
            lr: loaded learning rate
            next_epoch: id of the nex epoch
        """
        logger.info('Loading from: %s' % prefix)

        model.load_model(prefix+'.model')
        optimizer.load_state_dict(torch.load(prefix+'.opt'))
        torch.set_rng_state(torch.load(prefix+'.rng'))
        torch.cuda.set_rng_state(torch.load(prefix+'.curng'))

        lr = optimizer.param_groups[0]['lr']
        tmp = os.path.basename(prefix)
        next_epoch = int(tmp.replace('epoch_', '')) + 1
        return lr, next_epoch

    @staticmethod
    def loss_widgets(msg=''):
        """Generate a default widgets for progress bar with loss information

        Args:
            msg: a string to show the message at the beginning of the bar

        Return:
            widgets: list of items for the progress bar widgets
        """
        widgets = [
            msg, progressbar.Percentage(),
            ' ', progressbar.Bar(marker='#', left='[', right=']'),
            ' ', progressbar.DynamicMessage('loss', width=10, precision=7),
            ' ', progressbar.ETA(),
        ]
        return widgets

    @staticmethod
    def gen_pbar(max_value, msg='', widgets=None):
        """Generate a progress bar

        Args:
            max_value: maximum value of the bar
            msg: a string to show the message at the beginning of the bar
            widgets: list of items for the progress bar widgets. If None, will
                use the default loss_widgets

        Return:
            pbar: the generated progress bar
        """
        if widgets is None:
            widgets = MiscUtils.loss_widgets(msg)
        pbar = progressbar.ProgressBar(widgets=widgets, max_value=max_value)
        return pbar

    @staticmethod
    def get_lastest_checkpoint(logdir, regex='epoch_*.model'):
        """Get the latest checkpoint in a logdir

        For example, for the logdir with:
            logdir/
                epoch_00000.model
                epoch_00001.model
                ...
                epoch_00099.model
        The function will return `logdir/epoch_00099`

        Args:
            logdir: log directory to find the latest checkpoint
            regex: regular expression to describe the checkpoint

        Return:
            prefix: prefix with the format [logdir]/epoch_[epoch]
        """
        assert os.path.isdir(logdir), 'Not a directory: {}'.format(logdir)

        save_lst = glob.glob(os.path.join(logdir, regex))
        save_lst.sort()
        prefix = save_lst[-1].replace('.model', '')
        return prefix
