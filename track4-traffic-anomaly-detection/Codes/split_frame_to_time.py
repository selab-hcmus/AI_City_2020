id_s_frames = open('id_s_frames.txt', 'r').readlines()
split_data_count = open('split_data_count.txt', 'r').readlines()

arr_s = []
arr_frames = []
video_name = 0
for tmp in id_s_frames:
    video_name = video_name + 1
    tmp = id_s_frames[video_name - 1].replace('\n', '').split(' ')
    tmp = [string for string in tmp if string != ""]
    s, frames = float(tmp[1]), int(tmp[2])
    arr_s.append(s)
    arr_frames.append(frames)
    
split_data = [None]*100
for tmp in split_data_count:
    tmp = tmp.replace('\n', '').split(' ')
    tmp = [string for string in tmp if string != ""]
    vid, split_frame, real_frame = int(tmp[0]), int(tmp[1]), int(tmp[2])
    if split_data[vid-1] == None:
        split_data[vid-1] = [None]*1000
    split_data[vid-1][split_frame-1] = real_frame

second_only=False

def frame_to_s(video_id, detected_frame):
    s_detected = split_data[video_id-1][detected_frame-1]/arr_frames[video_id - 1]*arr_s[video_id - 1]
    if second_only == True:
        return s_detected
    _m = s_detected // 60
    _s = (s_detected - _m*60)
    return (s_detected, _m, _s)

# @@@
f = open('get_min_max_score_idx.txt').readlines()
for line in f:
    tmp = line.replace('\n', '').split(' ')
    tmp = [string for string in tmp if string != ""]
    video_name = int(tmp[0])
    f_min = int(tmp[1])
    if f_min == 0:
        f_min = 1
#     f_max = int(tmp[2])
#     print(video_name, frame_to_s(video_name, f_min), frame_to_s(video_name, f_max))
    print(video_name, frame_to_s(video_name, f_min))