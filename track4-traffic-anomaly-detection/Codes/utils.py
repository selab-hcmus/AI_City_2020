import glob
import os
from collections import OrderedDict

import cv2
import numpy as np
import tensorflow as tf

from constant import const

rng = np.random.RandomState(2017)


def np_load_frame(filename, resize_height, resize_width):
    """
    Load image path and convert it to numpy.ndarray. Notes that the color channels are BGR and the color space
    is normalized from [0, 255] to [-1, 1].

    :param filename: the full path of image
    :param resize_height: resized height
    :param resize_width: resized width
    :return: numpy.ndarray
    """
    image_decoded = cv2.imread(filename)
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    image_resized = image_resized.astype(dtype=np.float32)
    image_resized = (image_resized / 127.5) - 1.0
    return image_resized


class DataLoader(object):
    def __init__(self, video_folder, resize_height=256, resize_width=256, phase='train'):
        self.dir = video_folder
        self.phase = phase
        self.videos = OrderedDict()
        self._resize_height = resize_height
        self._resize_width = resize_width
        self.setup()

    def __call__(self, batch_size, time_steps, num_pred=1):
        video_info_list = list(self.videos.values())
        num_videos = len(video_info_list)
        print(num_videos)
        clip_length = time_steps + num_pred
        resize_height, resize_width = self._resize_height, self._resize_width

        def video_clip_generator():
            v_id = -1
            while True:
                v_id = (v_id + 1) % num_videos

                video_info = video_info_list[v_id]
                try:
                    start = rng.randint(0, video_info['length'] - clip_length)
                except:
                    print(video_info)
                    print(clip_length)
                    raise ValueError()
                video_clip = []
                if const.BLEND_MOTION == 0:
                    for frame_id in range(start, start + clip_length):
                        video_clip.append(np_load_frame(video_info['frame'][frame_id], resize_height, resize_width))
                else:
                    blend_im = video_info['frame'][start].replace('origin', 'blend')
                    video_clip.append(np_load_frame(blend_im, resize_height, resize_width))
                    video_clip.append(np_load_frame(video_info['frame'][start], resize_height, resize_width))
                    video_clip.append(np_load_frame(video_info['frame'][start + 1], resize_height, resize_width))
                video_clip = np.concatenate(video_clip, axis=2)

                yield video_clip

        # video clip paths
        dataset = tf.data.Dataset.from_generator(generator=video_clip_generator,
                                                 output_types=tf.float32,
                                                 output_shapes=[resize_height, resize_width, clip_length * 3])
        print('generator dataset, {}'.format(dataset))
        dataset = dataset.prefetch(buffer_size=const.PREFETCH)
        dataset = dataset.shuffle(buffer_size=const.PREFETCH).batch(batch_size)
        print('epoch dataset, {}'.format(dataset))

        return dataset

    def __getitem__(self, video_name):
        assert video_name in self.videos.keys(), 'video = {} is not in {}!'.format(video_name, self.videos.keys())
        return self.videos[video_name]

    def setup(self):
        videos = sorted(glob.glob(os.path.join(self.dir, 'origin/*')))
        for video in sorted(videos):
            video_name = video.split('/')[-1]
            self.videos[video_name] = {}
            if self.phase == 'train':
                self.videos[video_name]['path'] = video
            elif self.phase == 'test':
                self.videos[video_name]['path'] = os.path.join(video, 'images')
            else:
                raise NameError(self.phase)
            self.videos[video_name]['frame'] = glob.glob(os.path.join(self.videos[video_name]['path'], '*.jpg'))
            self.videos[video_name]['frame'].sort()
            self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])

    def get_video_clips(self, video, start, end):
        # assert video in self.videos, 'video = {} must in {}!'.format(video, self.videos.keys())
        # assert start >= 0, 'start = {} must >=0!'.format(start)
        # assert end <= self.videos[video]['length'], 'end = {} must <= {}'.format(video, self.videos[video]['length'])
        batch = []
        if const.BLEND_MOTION == 0:
            for i in range(start, end):
                image = np_load_frame(self.videos[video]['frame'][i], self._resize_height, self._resize_width)
                batch.append(image)
        else:
            blend_path = self.videos[video]['frame'][start].replace('origin', 'blend')
            image = np_load_frame(blend_path, self._resize_height, self._resize_width)
            batch.append(image)

            image = np_load_frame(self.videos[video]['frame'][start], self._resize_height, self._resize_width)
            batch.append(image)

            image = np_load_frame(self.videos[video]['frame'][start + 1], self._resize_height, self._resize_width)
            batch.append(image)

        return np.concatenate(batch, axis=2)


def log10(t):
    """
    Calculates the base-10 log of each element in t.

    @param t: The tensor from which to calculate the base-10 log.

    @return: A tensor with the base-10 log of each element in t.
    """

    numerator = tf.log(t)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def psnr_error(gen_frames, gt_frames):
    """
    Computes the Peak Signal to Noise Ratio error between the generated images and the ground
    truth images.

    @param gen_frames: A tensor of shape [batch_size, height, width, 3]. The frames generated by the
                       generator model.
    @param gt_frames: A tensor of shape [batch_size, height, width, 3]. The ground-truth frames for
                      each frame in gen_frames.

    @return: A scalar tensor. The mean Peak Signal to Noise Ratio error over each frame in the
             batch.
    """
    shape = tf.shape(gen_frames)
    num_pixels = tf.to_float(shape[1] * shape[2] * shape[3])
    gt_frames = (gt_frames + 1.0) / 2.0
    gen_frames = (gen_frames + 1.0) / 2.0
    square_diff = tf.square(gt_frames - gen_frames)

    batch_errors = 10 * log10(1 / ((1 / num_pixels) * tf.reduce_sum(square_diff, [1, 2, 3])))
    return tf.reduce_mean(batch_errors)


def diff_mask(gen_frames, gt_frames, min_value=-1, max_value=1):
    # normalize to [0, 1]
    delta = max_value - min_value
    gen_frames = (gen_frames - min_value) / delta
    gt_frames = (gt_frames - min_value) / delta

    gen_gray_frames = tf.image.rgb_to_grayscale(gen_frames)
    gt_gray_frames = tf.image.rgb_to_grayscale(gt_frames)

    diff = tf.abs(gen_gray_frames - gt_gray_frames)
    return diff


def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')

def set_up_matplotlib():
    """Set matplotlib up."""
    import matplotlib
    # Use a non-interactive backend
    matplotlib.use('Agg')
