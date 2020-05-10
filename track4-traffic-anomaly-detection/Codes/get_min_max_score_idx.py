from __future__ import print_function
import numpy as np
# video_names = [1, 32, 35, 46, 64, 68, 82]
video_names = [32, 35]
for video_name in video_names:
    file_name = '../images/ai_city_' + str(video_name) + '/scores.txt'
#     print(file_name)
    f = open(file_name, 'r').readlines()
    tmp = [string.replace('\n', '') for string in f if string != ""]
    results = np.array(list(map(float, f)))
    idx_min = np.argmin(results)
    idx_max = np.argmax(results)
    # print(video_name, idx_min + 1, idx_max + 1)
    print(video_name, idx_min + 1)
    