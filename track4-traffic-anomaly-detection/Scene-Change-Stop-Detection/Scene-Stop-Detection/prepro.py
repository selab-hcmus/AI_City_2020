import json
import glob
import os
import cv2

cuts_dir = '../aic19-track3-test-data/stop_cuts'

dataset_cuts = {}
cuts_files = sorted(list(glob.glob(os.path.join(cuts_dir, '*.mp4.json'))), key=lambda x: int(os.path.basename(x).split('.')[0]))
for cuts_file in cuts_files:
    with open(cuts_file, 'r') as f:
        cuts = json.load(f)
    dirname = os.path.dirname(cuts_file)
    basename = os.path.basename(cuts_file)

    vid = cv2.VideoCapture(os.path.join(dirname[:-4], basename[:-5]))
    num_frms = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    vid.release()

    cur_frm = 0

    dataset_cuts[basename.split('.')[0]] = []

    i = 0
    while i < len(cuts):
        cut = cuts[i] / 30
        start_cut = cut

        duration = 0
        while i + 1 < len(cuts):
            end_cut = cuts[i+1] / 30
            if end_cut > cut + 1:
                break
            duration += 1
            i += 1
            cut = end_cut
        print basename, start_cut, duration
        
        if duration > 0:
            dataset_cuts[basename.split('.')[0]].append((start_cut-1, start_cut + duration))
        i += 1 
        
    print basename.split('.')[0], len(cuts), dataset_cuts[basename.split('.')[0]]

with open(os.path.join(cuts_dir, 'stop_scene_periods.json'), 'w') as f:
    json.dump(dataset_cuts, f)
