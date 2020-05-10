import numpy as np
import imageio
import os
import matplotlib.pyplot as plt
import glob
from pathlib import Path

alpha = 0.01

data_path = 'F:\\Workspace\\aicity2020\\AIC20_track4\\test-data'
output_path = Path(r"F:\Workspace\tuanbi97\AICityChallenge\average_image")

for vid_id in range(1,101):
    fname = os.path.join(data_path, '%d.mp4' %vid_id)
    reader = imageio.get_reader(fname, 'ffmpeg')
    meta_data = reader.get_meta_data()
    length = int(meta_data['duration']*meta_data['fps'])

    frame = reader.get_data(0)
    
    average = frame
    #temp = np.zeros_like(frame)

    for i in range(1, length):
        prev = frame
        frame = reader.get_data(i)
        if (i % 3600 == 0):
            print(i) #print process

        ## calculating average image
        average = (1 - alpha)*average + alpha*frame
        if (i % 30 == 0):
            second = i // 30
            path = output_path / str(vid_id)
            path.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(path / "average{}.jpg".format(second), average.astype(np.uint8))


    
