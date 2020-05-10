from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2
from tqdm import tqdm
import pickle
import time

from opts import opts
from detectors.detector_factory import detector_factory
from utils.debugger import Debugger

def demo(opt):
    # creat folder save results
    os.mkdir('../Detection/bboxes_{}'.format(opt.arch))

    # class_map = {1: 1, 2: 2} # color for boundingbox
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    # opt.debug = max(opt.debug, 1)
    Detector = detector_factory[opt.task]
    detector = Detector(opt)

    assert os.path.isdir(opt.demo), 'Need path to videos directory.'
    video_paths = [
        os.path.join(opt.demo, video_name) for video_name in os.listdir(opt.demo) if video_name.split('.')[-1] == 'mp4'
    ]
    
    # video_paths = [
    #     os.path.join(opt.demo, 'cam_2.mp4')
    # ]

    # debugger = Debugger(dataset=opt.dataset, theme=opt.debugger_theme)

    for video_path in sorted(video_paths):
        bboxes = []
        video = cv2.VideoCapture(video_path)
        width, height = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # bbox_video = cv2.VideoWriter(
        #     filename='/home/leducthinh0409/centernet_visualize_{}/'.format(opt.arch) + os.path.basename(video_path),
        #     fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
        #     fps=float(30),
        #     frameSize=(width, height),
        #     isColor=True
        # )
     
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in tqdm(range(num_frames)):
            # skip_frame
            if opt.skip_frame > 0:
                if i % opt.skip_frame == 0:
                    continue

            _, img = video.read()
            
            ret = detector.run(img)
            bboxes.append(ret['results'])

            # debugger.add_img(img, img_id='default')
            # for class_id in class_map.keys():
            #     for bbox in ret['results'][class_id]:
            #         if bbox[4] > opt.vis_thresh:
            #             debugger.add_coco_bbox(bbox[:4], class_map[class_id], bbox[4], img_id='default')
            # bbox_img = debugger.imgs['default']
            # bbox_video.write(bbox_img)

        with open('../Detection/bboxes_{}'.format(opt.arch) + os.path.basename(video_path) + '.pkl', 'wb') as f:
            pickle.dump(bboxes, f)

if __name__ == '__main__':
    start = time.time()
    opt = opts().init()
    print('*skip_frame = ', opt.skip_frame)
    demo(opt)
    end = time.time()
    print('total_time inference = ', end - start)