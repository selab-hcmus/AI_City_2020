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

"""Script for visualizing results saved in a .pkl file."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import cv2
import numpy as np
import os
import sys
import pickle
import glob
from tqdm import tqdm

import detectron.utils.vis as vis_utils
import detectron.datasets.dummy_datasets as dummy_datasets

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'detection_dirs',
        help='file saving detections of videos',
        nargs='+',
        type=str
    )
    parser.add_argument(
        '--thresh',
        dest='thresh',
        help='detection prob threshold',
        default=0.9,
        type=float
    )
    parser.add_argument(
        '--kp-thresh',
        dest='kp_thresh',
        help='Threshold for visualizing keypoints',
        default=2.0,
        type=float
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        default='/tmp/result_videos/',
        type=str
    )
    parser.add_argument(
        '--average-frame',
        dest='average_frame',
        help='Allowing averaging frames by moving average algorithm',
        action='store_true'
    )
    parser.add_argument(
        '--alpha',
        dest='alpha',
        help='Coefficient for moving average algorithm if average-frame is fired',
        default=0.01,
        type=float
    )
    parser.add_argument(
        '--save-as-images',
        dest='save_as_images',
        help='Save result videos as separate image for each frame.',
        action='store_true'
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


def vis(detection_dirs, output_dir, thresh, kp_thresh, average_frame, alpha, save_as_images):
    assert all([os.path.isdir(detect_dir) for detect_dir in detection_dirs]), 'All input paths should be directories!'

    detection_files = sorted(os.listdir(detection_dirs[0]))
    assert all([sorted(os.listdir(detect_dir)) == detection_files for detect_dir in detection_dirs])

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    for detect_file in detection_files:
        all_detections = []
        for detect_dir in detection_dirs:
            with open(os.path.join(detect_dir, detect_file), 'rb') as f:
                all_detections.append(pickle.load(f))

        dummy_coco_dataset = dummy_datasets.get_coco_dataset_with_classes(['__background__', 'car'])

        boxes_seq, segms_seq, keyps_seq = [[vid_detections[key] for vid_detections in all_detections]
                                                                for key in ['bboxes', 'segments', 'keypoints']]
        vid_path = all_detections[0]['video_path']
        orig_num_frms, orig_fps, out_fps, w, h = [all_detections[0][kw] for kw in 
                                                    ['original_num_frames', 'original_fps',
                                                    'out_fps', 'width', 'height']]
        time_stride = int(orig_fps / out_fps)
        out_vid_path = os.path.join(output_dir, os.path.basename(vid_path))

        vid_cap = cv2.VideoCapture(vid_path)
        if save_as_images:
            if not os.path.isdir(out_vid_path):
                os.mkdir(out_vid_path)
        else:
            vid_writer = cv2.VideoWriter(out_vid_path, 1983148141, out_fps, (w, h))
        
        if average_frame:
            vid_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            _, average_im = vid_cap.read()
            first_step = time_stride
        else:
            first_step = 0

        assert vid_cap.isOpened(), 'Cannot open input video.'
        if not save_as_images:
            assert vid_writer.isOpened(), 'Cannot create output video.'

        for out_frm_id, orig_frm_id in tqdm(enumerate(range(first_step, int(orig_num_frms), time_stride)), total=int((orig_num_frms-first_step)//time_stride)):
            vid_cap.set(cv2.CAP_PROP_POS_FRAMES, orig_frm_id)
            _, im = vid_cap.read()

            if average_frame:
                average_im = np.around((1 - alpha) * average_im + alpha * im).astype(np.uint8)
                im = average_im

            for i in range(len(all_detections)):
                im = vis_utils.vis_one_image_opencv(im,
                    boxes_seq[i][out_frm_id],
                    segms_seq[i][out_frm_id],
                    keyps_seq[i][out_frm_id],
                    dataset=dummy_coco_dataset,
                    show_class=True,
                    show_box=True,
                    thresh=thresh,
                    kp_thresh=kp_thresh,)

            if save_as_images:
                cv2.imwrite(os.path.join(out_vid_path, str(out_frm_id) + '.jpg'), im)
            else:
                vid_writer.write(im)

        vid_cap.release()

        if not save_as_images:
            vid_writer.release()


if __name__ == '__main__':
    opts = parse_args()
    vis(
        opts.detection_dirs,
        opts.output_dir,
        opts.thresh,
        opts.kp_thresh,
        opts.average_frame,
        opts.alpha,
        opts.save_as_images
    )
