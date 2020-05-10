import json
import glob
import os
import cv2

cuts_dir = '../aic19-track3-test-data/cuts'

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
    for cut in cuts:
        #print cur_frm, cut
        if cur_frm < cut - 5*30:
            dataset_cuts[basename.split('.')[0]].append((cur_frm, cut - 30))
        cur_frm = cut + 30
    if cur_frm < num_frms - 5*30:
        dataset_cuts[basename.split('.')[0]].append((cur_frm, num_frms))
    print basename.split('.')[0], len(cuts), dataset_cuts[basename.split('.')[0]]

with open(os.path.join(cuts_dir, 'unchanged_scene_periods.json'), 'w') as f:
    json.dump(dataset_cuts, f)
