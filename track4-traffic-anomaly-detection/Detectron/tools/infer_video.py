#!/usr/bin/env python

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import numpy as np
import glob
import logging
import os
import sys
import time
import pickle
from tqdm import tqdm

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for saving detections(default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--video-ext',
        dest='video_ext',
        help='video file name extension (default: mp4)',
        default='mp4',
        type=str
    )
    parser.add_argument(
        '--output-frame-rate',
        dest='out_fps',
        help='output frame rate (default: 10)',
        default=10,
        type=float
    )
    parser.add_argument(
        '--vis-vid',
        dest='vis_vid',
        help='visualize video or just save bboxes and other outputs',
        action='store_true'
    )
    parser.add_argument(
        '--average-frame',
        dest='average_frame',
        help='allow averaging frames before inference to catch still objects',
        action='store_true'
    )
    parser.add_argument(
        '--alpha',
        dest='alpha',
        default=0.01,
        type=float,
        help='moving average coefficient used for computing average frame.'
    )
    parser.add_argument(
        'vi_or_folder', help='video or folder of videos', default=None
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger = logging.getLogger(__name__)

    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)

    assert not cfg.MODEL.RPN_ONLY, \
        'RPN models are not supported'
    assert not cfg.TEST.PRECOMPUTED_PROPOSALS, \
        'Models that require precomputed proposals are not supported'

    model = infer_engine.initialize_model_from_cfg(args.weights)

    if os.path.isdir(args.vi_or_folder):
        vi_list = glob.iglob(os.path.join(args.vi_or_folder, '*.' + args.video_ext))
    else:
        vi_list = [args.vi_or_folder]

    for i, vi_name in enumerate(vi_list):
        logger.info('Processing {}'.format(vi_name))
        cap = cv2.VideoCapture(vi_name)

        if cap.isOpened():
            num_frms, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
            detections = {'video_path': vi_name, 
                          'original_fps': fps, 'original_num_frames': num_frms, 'out_fps': args.out_fps,
                          'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                          'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                          'bboxes': [], 'segments': [], 'keypoints': []}

            timers = defaultdict(Timer)
            t = time.time()
            time_stride = int(fps / args.out_fps)

            print('Frame averaging mode: ', args.average_frame)
            if args.average_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                _, average_im = cap.read()
                first_step = 1
            else:
                first_step = 0

            print('Number of total frames: ', int(num_frms))
            for frm_id in tqdm(range(first_step, int(num_frms))):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frm_id)
                _, im = cap.read()

                if args.average_frame:
                    average_im = cv2.addWeighted(average_im, 1 - args.alpha, im ,args.alpha, 0)
                    im = average_im

                if frm_id % time_stride == 0:
                    with c2_utils.NamedCudaScope(0):
                        cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                            model, im, None, timers=timers)
                        for key, val in zip(['bboxes', 'segments', 'keypoints'], [cls_boxes, cls_segms, cls_keyps]):
                            detections[key].append(val)

            with open(os.path.join(args.output_dir, os.path.basename(vi_name) + '.pkl'), 'wb') as f:
                pickle.dump(detections, f)
                logger.info('Saved results to %s' % os.path.join(args.output_dir, os.path.basename(vi_name) + '.pkl'))
            logger.info('Inference time: {:.3f}s'.format(time.time() - t))


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
