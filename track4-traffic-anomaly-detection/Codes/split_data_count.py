from __future__ import print_function
import os
import cv2
import numpy as np

def sigmoid(x):
    return (1 / (1 + np.exp(-x))).astype(np.float32)

def diff(frame1, frame2):
    prev = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    sub = np.abs(gray.astype(np.float32) - prev.astype(np.float32))
    #print('sum diff 1: ', np.sum(sub), np.max(sub))
    # if (frame_id > 210):
    #     plt.imshow(sub, cmap='gray')
    #     plt.show()
    sub = sigmoid(sub - 40)
    #print('sum diff 2: ', np.sum(sub), np.max(sub))
    # if (frame_id > 210):
    #     plt.imshow(sub, cmap='gray')
    #     plt.show()
    diff = int(np.sum(sub))
    return (diff > 10)

# data_path = '../Data/ai_city/training/videos/origin/aic19-track3-test-data'
data_path = '../../AIC20/AIC20_track4/test-data/'

# #mini training
# # output_path = '../Data/ai_city/training/frames/mini'
# output_path = '../data/training/frames/mini'
# videos = [1, 32, 35, 46, 64, 68, 82]
# interval = {
#     1: [600, 1200],
#     32: [19200, 20700],
#     35: [2*60*30, 3*60*30],
#     46: [0, (1*60 + 40)*30],
#     64: [6*60*30, 8*60*30],
#     68: [60*30, 2*60*30],
#     82: [0, 1* 60 * 30]
# }

# alpha = 0.5

# for video_id in videos:
#     video_path = os.path.join(data_path, str(video_id) + '.mp4')
#     print(video_path)
#     save_path = os.path.join(output_path, 'blend', str(video_id))
#     origin_path = os.path.join(output_path, 'origin', str(video_id))
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#     if not os.path.exists(origin_path):
#         os.makedirs(origin_path)
#     cap = cv2.VideoCapture(video_path)
#     reserve = 600
#     cap.set(cv2.CAP_PROP_POS_FRAMES, interval[video_id][0] - reserve)
#     frame_id = 0
#     true_id = interval[video_id][0] - reserve
#     ret, ave = cap.read()
#     ave = np.array(ave, dtype=np.float32)
#     prev_frame = ave
#     while (True):
#         true_id += 1
#         if (true_id > interval[video_id][1]):
#             break
#         ret, frame = cap.read()
#         if ret == False:
#             break
#         if not diff(frame, prev_frame):
#             continue
#         ave = (1 - alpha) * ave + alpha * frame
#         if true_id > interval[video_id][0]:
#           frame_id += 1
#           cv2.imwrite(os.path.join(save_path, 'frame%05d.jpg' % frame_id), ave)
#           cv2.imwrite(os.path.join(origin_path, 'frame%05d.jpg' % frame_id), frame)
#         prev_frame = frame
#     cap.release()

# #mini testing
# # output_path = '../Data/ai_city/testing/frames/mini-test'
# output_path = '../data/testing/frames/mini-test'
# videos = [8, 15, 16, 23, 33, 45]
# interval = {
#     8: [3*60*30, 4*60*30],
#     15: [2*60*30, 3*60*30],
#     16: [2*60*30, 3*60*30],
#     23: [2*60*30, 3*60*30],
#     33: [2*60*30, 3*60*30],
#     45: [1*60*30, 2*60*30],
# }

# alpha = 0.5

# for video_id in videos:
#     video_path = os.path.join(data_path, str(video_id) + '.mp4')
#     save_path = os.path.join(output_path, 'blend', str(video_id))
#     origin_path = os.path.join(output_path, 'origin', str(video_id))
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#     if not os.path.exists(origin_path):
#         os.makedirs(origin_path)
#     cap = cv2.VideoCapture(video_path)
#     reserve = 600
#     cap.set(cv2.CAP_PROP_POS_FRAMES, interval[video_id][0] - reserve)
#     frame_id = 0
#     true_id = interval[video_id][0] - reserve
#     ret, ave = cap.read()
#     ave = np.array(ave, dtype=np.float32)
#     prev_frame = ave
#     while (True):
#         true_id += 1
#         if (true_id > interval[video_id][1]):
#             break
#         ret, frame = cap.read()
#         if ret == False:
#             break
#         if not diff(frame, prev_frame):
#             continue
#         ave = (1 - alpha) * ave + alpha * frame
#         if true_id > interval[video_id][0]:
#           frame_id += 1
#           cv2.imwrite(os.path.join(save_path, 'frame%05d.jpg' % frame_id), ave)
#           cv2.imwrite(os.path.join(origin_path, 'frame%05d.jpg' % frame_id), frame)
#         prev_frame = frame
#     cap.release()

#mini challenge
# output_path = '../Data/ai_city/testing/frames/mini-challenge'
output_path = '../data/testing/frames/mini-challenge'
videos = [1, 32, 35, 46, 64, 68, 82]
interval = {
    1: [30 * 30, (1 * 60 + 40) * 30],
    32: [(12 * 60 + 20) * 30, (12 * 60 + 50) * 30],
    35: [(4 * 60 + 20)*30, (4 * 60 + 50)*30],
    46: [(1 * 60 +50)*30, (2*60 + 15)*30],
    64: [(8*60 + 20)*30, (8*60 + 40)*30],
    68: [(2*60 + 35)*30, 3*60*30],
    82: [(1 * 60 + 10) * 30, (1* 60 + 30) * 30]
}

alpha = 0.5

for video_id in videos:
    video_path = os.path.join(data_path, str(video_id) + '.mp4')
    save_path = os.path.join(output_path, 'blend', str(video_id))
    origin_path = os.path.join(output_path, 'origin', str(video_id))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(origin_path):
        os.makedirs(origin_path)
    cap = cv2.VideoCapture(video_path)
    reserve = 600
    cap.set(cv2.CAP_PROP_POS_FRAMES, interval[video_id][0] - reserve)
    frame_id = 0
    true_id = interval[video_id][0] - reserve
    ret, ave = cap.read()
    prev_frame = ave
    while (True):
        true_id += 1
        if (true_id > interval[video_id][1]):
            break
        ret, frame = cap.read()
        if ret == False:
            break
        if not diff(frame, prev_frame):
            continue
        ave = (1 - alpha) * ave + alpha * frame
        if (true_id > interval[video_id][0]):
            frame_id += 1
            cv2.imwrite(os.path.join(save_path, 'frame%05d.jpg' % frame_id), ave)
            cv2.imwrite(os.path.join(origin_path, 'frame%05d.jpg' % frame_id), frame)
            print(video_id, frame_id, true_id)
        prev_frame = frame
    cap.release()
