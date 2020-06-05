import numpy as np
import os
import matplotlib.path as mplPath
import cv2
import json
PATH_ROI = "../data/AIC20_track1/ROIs"
PATH_MOI = "./MOIs/"
PATH_SCREENSHOT = "./screen_shot_with_roi_and_movement"
PATH_RESULTS = "../dla_backbone/info_counting"
PATH_VIDEO = "../data/AIC20_track1/Dataset_A"
PATH_TRACKING = "../dla_backbone/info_tracking"
# PATH_TRACKING = "../Center_net/info_split"
PATH_VIDEO_OUT = "../dla_backbone/counting_visualized"
VISUALIZED = True

def load_roi():
    roi_list = {}
    for file_name in os.listdir(PATH_ROI):
        if(file_name.endswith(".txt")):
            full_path_name = os.path.join(PATH_ROI, file_name)
            file_p = open(full_path_name, "r")
            content_list = file_p.read().splitlines()
            content_list = [(int(point.split(',')[0]), int(point.split(',')[1])) for point in content_list]
            roi_list[file_name[:-4]] = content_list
    return roi_list

def load_moi():
    moi_list = {}
    '''
        moi_list is a dictionary, each key is video name and its corresponding mask
        each value in each key is also a dictionary.This dictionary has key as movement_id in
        video and value is the the mask of the movement in binary 
    '''
    print("Extracting MOI")
    for folder_name in os.listdir(PATH_MOI):
        moi_list[folder_name] = {}
        for file_name in os.listdir(os.path.join(PATH_MOI, folder_name)):
            if file_name.endswith(".npy"):
                movement_id = file_name.split("_")[-1][:-4]
                full_path_name = os.path.join(PATH_MOI, folder_name, file_name)
                content = np.load(full_path_name)
                moi_list[folder_name][movement_id] = content
    return moi_list

def out_of_roi(center, poly):
    path_array = []
    for poly_point in poly:
        path_array.append([poly_point[0], poly_point[1]])
    path_array = np.asarray(path_array)
    polyPath = mplPath.Path(path_array)

    return polyPath.contains_point(center, radius = 0.5)

def validate_center(center, use_off_set, roi_list):
    const = 5
    off_set = [(const, const), (const, -const), (-const, const), (-const, -const)]

    if not use_off_set:
        return  out_of_roi(center, roi_list)

    for each_off_set in off_set:
        center_change = (center[0] + each_off_set[0], center[1] + each_off_set[1])
        if out_of_roi(center_change, roi_list):
            return False
    return True

def out_of_range_bbox(tracking_info, width, height, off_set):
    x_min = int(tracking_info[4])
    y_min = int(tracking_info[5])
    x_max = int(tracking_info[6])
    y_max = int(tracking_info[7])
    return ((x_min-off_set <=0) or (y_min-off_set<=0) or (x_max+off_set>=width) or (y_max+off_set>=height))


def find_latest_object_and_vote_direction(frame_id_list, cur_fr_id, tracking_info, delta_fix, target_obj_id, roi_list, width, height):
    exist_latest_obj = False
    count_out = 0
    count_in = 0
    offset = 10
    for delta in range(1, delta_fix):
        pre_index = np.where(frame_id_list == (cur_fr_id - delta))[0]
        for each_pre_index in pre_index:
            if tracking_info[each_pre_index][3]==target_obj_id:
                exist_latest_obj = True
                pre_obj_center = center_box(tracking_info[each_pre_index][4:])
                if out_of_range_bbox(tracking_info[each_pre_index], width, height, offset):
                    count_out += 1
                else:
                    if validate_center(pre_obj_center, False, roi_list):
                        count_in += 1
                    else:
                        count_out += 1
    return count_out, count_in, exist_latest_obj

def center_box(cur_box):
    return (int((cur_box[0]+cur_box[2])/2), int((cur_box[1]+cur_box[3])/2))

def draw_roi(roi_list, image):
    start_point = roi_list[0]
    for end_point in roi_list[1:]:
        cv2.line(image, start_point, end_point, (0,0,255), 2)
        start_point = end_point
    return image

def draw_moi(annotated_frame, vid_name):
    vid_name_json = vid_name.split("_")
    vid_name_json = vid_name_json[0]+"_"+vid_name_json[1]+".json"
    json_file = open(os.path.join(PATH_SCREENSHOT, vid_name_json))
    content = json.load(json_file)[u'shapes']
    for element in content:
        list_points = element['points']
        label = element['label']
        # draw arrow
        for index, point in enumerate(list_points):
            start_point = (list_points[index][0], list_points[index][1])
            end_point = (list_points[index+1][0], list_points[index+1][1])
            if index+2 == len(list_points):
                cv2.arrowedLine(annotated_frame, start_point, end_point, (255, 0, 0), 5)
                break
            else:
                cv2.line(annotated_frame, start_point, end_point, (255,0,0), 5)
        cv2.putText(annotated_frame, label, end_point, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2,cv2.LINE_AA)
    return annotated_frame

def voting(point, vote_movement, obj_id, moi_list):
    # one vote for each time point lying inside MOI
    exist_MOI = False
    if obj_id not in vote_movement:
        vote_movement[obj_id] = {}
    for moi_id in moi_list:
        moi_content = moi_list[moi_id]
        if moi_content[int(point[1])][int(point[0])] == True:
            if moi_id not in vote_movement[obj_id]:
                vote_movement[obj_id][moi_id] = 0
            vote_movement[obj_id][moi_id] += 1
            exist_MOI = True
    return vote_movement, exist_MOI

def build_text_name_dict(moi_list):
    results = {}
    for moi_id in moi_list:
        text_name_car = moi_id+"_"+"car"
        text_name_truck = moi_id+"_"+"truck"
        results[text_name_car] = 0
        results[text_name_truck] = 0
    results = dict(sorted (results.items()))
    return results
def randomize_color():
    return 
def draw_path(annotate_fr, offset_fr, cur_obj_id, tracking_info, frame_id_list, cur_fr_id):
    COLORS = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
    start_point = []
    for delta in range(0, offset_fr):
        pre_index = np.where(frame_id_list == (cur_fr_id - delta))[0]
        for each_pre_index in pre_index:
            if tracking_info[each_pre_index][3]==cur_obj_id:
                end_point = center_box(tracking_info[each_pre_index][4:])
                if len(start_point)!=0:
                    cv2.line(annotate_fr, start_point, end_point, COLORS, 10)
                start_point = end_point
    return annotate_fr

def draw_text_summarize(annotate_fr, text_name_dict, width, height):
    x_coor = 20
    y_coor = 20
    for text_name in text_name_dict:
        str_write = text_name+":"+str(text_name_dict[text_name])+"."
        cv2.putText(annotate_fr, str_write, (x_coor, y_coor), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2,cv2.LINE_AA)
        x_coor += 150
        if x_coor + 150 >=width:
            x_coor = 20
            y_coor += 40

    return annotate_fr

def car_counting(vid_name, roi_list, moi_list):
    video_name = vid_name+".mp4"
    print("Processing", vid_name)
    tracking_info = np.load(PATH_TRACKING + '/info_' + video_name + '.npy', allow_pickle = True)
    N = tracking_info.shape[0]
    frame_id = tracking_info[:, 1].astype(np.int).reshape(N)
    delta_fix = 10
    obj_id = tracking_info[:, 3].astype(np.int).reshape(N)
    results = []
    num_car_out = 0

    input = cv2.VideoCapture(PATH_VIDEO + '/' + vid_name + '.mp4')
    width = int(input.get(3)) # get width
    print("Visualizing ", vid_name)
    height = int(input.get(4)) #get height

    num_truck_out = 0
    already_count = []
    vote_movement = {} # each key is obj_id, each value is A dictionary. Each key in A is movement_id, each value in A is count vote for that object to the corresponding 
    num_car_out = {} # each key is mv_id, each value count number of car 
    num_truck_out = {} # each key is mv_id, each value count number of truck


    for fr_id in range(1, max(frame_id)+1):
        index_cur_fr = np.where(frame_id==fr_id)[0]
        for index_box in index_cur_fr:

            cur_box = tracking_info[index_box][4:]
            cur_center = center_box(cur_box)
            cur_obj_id = tracking_info[index_box][3]

            is_inside_roi = validate_center(cur_center, False, roi_list)
            vote_movement, inside_MOI = voting(cur_center, vote_movement, cur_obj_id, moi_list)
            if not inside_MOI:
                continue 
            if not is_inside_roi:# current car is outside roi
                count_out, count_in, is_ok = find_latest_object_and_vote_direction(frame_id, fr_id, tracking_info, 10, cur_obj_id, roi_list, width, height)
                if is_ok: #exist object
                    # pre_obj_center = center_box(latest_obj[4:])
                    # is_inside_roi_pre_obj = validate_center(pre_obj_center, False, roi_list)
                    if count_in>=count_out and cur_obj_id not in already_count: # previous car lies inside roi
                        already_count.append(cur_obj_id)
                        max_movement_id = max(vote_movement[cur_obj_id], key=vote_movement[cur_obj_id].get)
                        if tracking_info[index_box][0] == 1:
                            if max_movement_id not in num_car_out:
                                num_car_out[max_movement_id] = 0
                            num_car_out[max_movement_id] += 1
                            num_object_out = num_car_out[max_movement_id]
                            class_type = "car"
                        else:
                            if max_movement_id not in num_truck_out:
                                num_truck_out[max_movement_id] = 0
                            num_truck_out[max_movement_id] += 1
                            num_object_out = num_truck_out[max_movement_id]
                            class_type = "truck"
                        results.append([fr_id, num_object_out, cur_center[0], cur_center[1], max_movement_id, class_type])
            else : # using offset to refine again
                is_out = out_of_range_bbox(tracking_info[index_box], width, height, 2)
                if is_out:
                    count_out, count_in, is_ok = find_latest_object_and_vote_direction(frame_id, fr_id, tracking_info, 10, cur_obj_id, roi_list, width, height)
                    if is_ok: #exist object
                        # pre_obj_center = center_box(latest_obj[4:])
                        # is_inside_roi_pre_obj = validate_center(pre_obj_center, False, roi_list)
                        if count_in>=count_out and cur_obj_id not in already_count: # previous car lies inside roi
                            already_count.append(cur_obj_id)
                            max_movement_id = max(vote_movement[cur_obj_id], key=vote_movement[cur_obj_id].get)
                            
                            if tracking_info[index_box][0] == 1:
                                if max_movement_id not in num_car_out:
                                    num_car_out[max_movement_id] = 0
                                num_car_out[max_movement_id] += 1
                                num_object_out = num_car_out[max_movement_id]
                                class_type = "car"
                            else:
                                if max_movement_id not in num_truck_out:
                                    num_truck_out[max_movement_id] = 0
                                num_truck_out[max_movement_id] += 1
                                num_object_out = num_truck_out[max_movement_id]
                                class_type = "truck"
                            results.append([fr_id, num_object_out, cur_center[0], cur_center[1], max_movement_id, class_type])
    if VISUALIZED:
        output = cv2.VideoWriter(PATH_VIDEO_OUT + '/' + vid_name + '.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), 5.0, (width, height))
        idx = 0
        results = np.array(results)
        N = len(results)
        print(N)
        frame_id = results[:, 0].astype(np.int).reshape(N)
        text_summary = build_text_name_dict(moi_list)

        while (input.isOpened()):
            ret, frame = input.read()
            if not ret:
                break
            idx += 1
            indx_cur_fr = np.where(frame_id == idx)[0]
            annotate_fr = draw_roi(roi_list, frame)
            if len(indx_cur_fr)!=0:
                for result_id in indx_cur_fr:
                    cur_annotate = results[result_id] 
                    count_object = cur_annotate[1]
                    cv2.putText(annotate_fr, cur_annotate[-1]+"-"+str(count_object).zfill(5), (int(cur_annotate[2]), int(cur_annotate[3])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2,cv2.LINE_AA) 
            annotate_fr = draw_roi(roi_list, frame)
            annotate_fr = draw_moi(annotate_fr, vid_name)
            #draw text results
            if len(indx_cur_fr)!=0:
                for result_id in indx_cur_fr:
                    cur_annotate = results[result_id] 
                    count_object = cur_annotate[1]
                    moi_id = cur_annotate[4]
                    class_type = cur_annotate[5]
                    text_summary[moi_id+"_"+class_type] = count_object
                    cv2.putText(annotate_fr, cur_annotate[4]+"-"+str(count_object).zfill(5), (int(cur_annotate[2]), int(cur_annotate[3])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2,cv2.LINE_AA) 
            annotate_fr = draw_text_summarize(annotate_fr, text_summary, width, height)
            output.write(annotate_fr)
#             if idx == 900:
#                 break
        input.release()
        output.release()
    np.save(PATH_RESULTS + '/info_' + vid_name+".mp4", results)
    return results

if __name__ == "__main__":
    moi_list = load_moi()
    roi_list = load_roi()
    for video_name in os.listdir(PATH_VIDEO):
        if video_name .endswith(".mp4"):
            roi_vid_name = video_name[:-4].split("_")[0]+ "_" + video_name[:-4].split("_")[1]
            results = car_counting(video_name[:-4], roi_list[roi_vid_name], moi_list[roi_vid_name])
