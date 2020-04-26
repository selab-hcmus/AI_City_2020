import numpy as np
import pickle
import os
import json
import time
from tracker.iou_tracker import *
# path to video data
path_video = "./data/AIC20_track1/Dataset_A"
# path to different detected bbox results
path_bbox_dla = "./dla_backbone/bboxes_dla_34"

def format_bbox(video_name, file_name):
    ''' prepare formatted bbox for tracking'''

    file_content = open(os.path.join(path_bbox_dla,file_name),'rb')
    content = pickle.load(file_content)
    print("Processing:",video_name)
    data = []

    for fr_id, fr_content in enumerate(content):
        dets = []
        c_bboxes = fr_content[1]
        t_bboxes = fr_content[2]
        for bb in c_bboxes:
            dets.append({'bbox': (bb[0], bb[1], bb[2], bb[3]), 'score': bb[4], 'class': 1})
        for bb in t_bboxes:
            dets.append({'bbox': (bb[0], bb[1], bb[2], bb[3]), 'score': bb[4], 'class': 2})
        data.append(dets)
    file_content.close()
    return data

if __name__ == "__main__":
    duration = time.time()
    print("Running tracking on ", path_bbox_dla)
    for file_name in os.listdir(path_bbox_dla):
        vid_name = file_name[:-8]
        if vid_name == "cam_3":
            data = format_bbox(vid_name, file_name)
            content_video_path = os.path.join(path_video, vid_name+".mp4")
            # results = track_viou_edited(content_video_path, data, 0.3, 0.7, 0.6, 13, 6, "KCF", 1.0)
            results = track_iou_edited(vid_name, data, 0.3, 0.7, 0.15, 10)
    # run_tracking_bbox_dla()
    duration = time.time() - duration 
    print("Total tracking time:", duration)
