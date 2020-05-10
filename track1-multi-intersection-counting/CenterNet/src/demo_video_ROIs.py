from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import numpy as np
import os
import cv2
from tqdm import tqdm
import pickle

from opts import opts
from detectors.detector_factory import detector_factory
from utils.debugger import Debugger

def demo(opt):
    class_map = {1: 1, 2: 2} # color for boundingbox
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
    debugger = Debugger(dataset=opt.dataset, theme=opt.debugger_theme)

    for video_path in sorted(video_paths):
        print('video_name = ', video_path)

        bboxes = []
        video = cv2.VideoCapture(video_path)
        width, height = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # pointer
        pts = []
        arr_name = os.path.basename(video_path).split('.')[0].split('_')
        cam_name = arr_name[0]+'_'+arr_name[1]
        print('cam_name = ',cam_name)
        with open('../ROIs/{}.txt'.format(cam_name)) as f:
          for line in f:
            pts.append([int(x) for x in line.split(',')])
        pts = np.array(pts)  
      
        # make mask
        mask = np.zeros((height, width), np.uint8)
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
        
        bbox_video = cv2.VideoWriter(
            filename='/home/leducthinh0409/centernet_visualize_{}/'.format(opt.arch) + os.path.basename(video_path),
            fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
            fps=float(30),
            frameSize=(width, height),
            isColor=True
        )
     
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in tqdm(range(num_frames)):
            _, img_pre = video.read()

            ## do bit-op
            dst = cv2.bitwise_and(img_pre, img_pre, mask=mask)
            ## add the white background
            bg = np.ones_like(img_pre, np.uint8)*255
            cv2.bitwise_not(bg, bg, mask=mask)
            img = bg + dst

            ret = detector.run(img)
            bboxes.append(ret['results'])
            debugger.add_img(img, img_id='default')
            for class_id in class_map.keys():
                for bbox in ret['results'][class_id]:
                    if bbox[4] > opt.vis_thresh:
                        debugger.add_coco_bbox(bbox[:4], class_map[class_id], bbox[4], img_id='default')
            bbox_img = debugger.imgs['default']
            bbox_video.write(bbox_img)

        with open('/home/leducthinh0409/bboxes_{}/'.format(opt.arch) + os.path.basename(video_path) + '.pkl', 'wb') as f:
            pickle.dump(bboxes, f)

if __name__ == '__main__':
    opt = opts().init()
    demo(opt)
