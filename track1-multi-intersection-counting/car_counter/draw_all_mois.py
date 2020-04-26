import numpy as np 
import os
import cv2
import json
import copy 
path_screenshot = "./screen_shot_with_roi_and_movement"
path_raw_image = "./screenshot_raw"
moi_path = "./MOIs"
def draw_moi(annotated_frame, vid_name, movement_id):
    vid_name_json = vid_name+".json"
    json_file = open(os.path.join(path_screenshot, vid_name_json))
    content = json.load(json_file)[u'shapes']
    for element in content:
        list_points = element['points']
        label = element['label']
        if int(label) == int(movement_id):
        # draw arrow
            for index, point in enumerate(list_points):
                start_point = (list_points[index][0], list_points[index][1])
                end_point = (list_points[index+1][0], list_points[index+1][1])
                if index+2 == len(list_points):
                    cv2.arrowedLine(annotated_frame, start_point, end_point, (255, 0, 0), 5)
                    break
                else:
                    cv2.line(annotated_frame, start_point, end_point, (255,0,0), 5)
            cv2.putText(annotated_frame, label, end_point, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5,cv2.LINE_AA)
    return annotated_frame
def draw():
    path = "./MOIs"
    for cam_folder in os.listdir(path):
        print("Process", cam_folder)
        path_folder = os.path.join(path, cam_folder)
        mask_list = []
        first = False
        for file_name in os.listdir(path_folder):
            if file_name.endswith(".npy"):
                mask = np.load(os.path.join(path_folder, file_name))
                if first == False:
                    original = mask
                    first = True
                else:
                    original = np.logical_or(original, mask)
        results=  np.array(original)+0
        cv2.imwrite("./all_mois_in_one/"+str(cam_folder)+".jpg", results*255)

def process_image(image, mask, mov_id, vid_name):
    image = draw_moi(image, vid_name, mov_id)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if mask[i][j] == False:
                image[i][j] = [0,0,0]
    return image

def draw_each_mask():
    for cam_screenshot in os.listdir(path_screenshot):
        if cam_screenshot[-4:] != ".jpg":
            continue
        print(os.path.join(path_raw_image,cam_screenshot))
        image = cv2.imread(os.path.join(path_raw_image,cam_screenshot))
        print("Process cam", cam_screenshot[:-4])
        full_path = os.path.join(moi_path, cam_screenshot[:-4])
        for file_name in os.listdir(full_path):
            tmp_image = copy.copy(image)
            if file_name.endswith(".npy"):
                mov_id = file_name.split("_")[-1][:-4]
                mask = np.load(os.path.join(full_path, file_name))
                tmp_image = process_image(tmp_image, mask, mov_id, cam_screenshot[:-4])
                if cam_screenshot[:-4] not in os.listdir("./blend_each_movement"):
                    os.makedirs("./blend_each_movement/"+cam_screenshot[:-4])
                cv2.imwrite("./blend_each_movement/"+cam_screenshot[:-4]+"/"+file_name[:-4]+".jpg", tmp_image)
if __name__ == "__main__":
    draw_each_mask()