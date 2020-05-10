from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import json
import os
import cv2

from opts import opts
from detectors.detector_factory import detector_factory

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
valid_ids = [1, 2, 3]

def to_float(x):
  return float("{:.2f}".format(x))

def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 1)
  Detector = detector_factory[opt.task]
  detector = Detector(opt)

  ls_video = os.listdir(opt.demo)
  for video in sorted(ls_video):
    new_demo = os.path.join(opt.demo, video)
    if os.path.isdir(new_demo):
      image_names = []
      ls = os.listdir(new_demo)
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(new_demo, file_name))
    else:
      image_names = [new_demo]
    
    detections = []
    for (image_name) in image_names:
      ret = detector.run(image_name)
      time_str = ''
      for stat in time_stats:
        time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
      print(time_str)
      # save file
      
      for cls_ind in ret['results']:
        category_id = valid_ids[cls_ind - 1]
        for bbox in ret['results'][cls_ind]:
          bbox[2] -= bbox[0]
          bbox[3] -= bbox[1]
          score = bbox[4]
          bbox_out  = list(map(to_float, bbox[0:4]))
          detection = {
              "image_name":image_name,
              "category_id": int(category_id),
              "bbox": bbox_out,
              "score": float("{:.2f}".format(score))
          }
          if len(bbox) > 5:
              extreme_points = list(map(to_float, bbox[5:13]))
              detection["extreme_points"] = extreme_points
          detections.append(detection)
    json.dump(detections,open('/home/leducthinh0409/json_frames_{}/{}.json'.format(opt.arch, new_demo.split('/')[-1]), 'w'))

if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
