import json
import os
import pickle

import colour
import cv2
import matplotlib.pyplot as plt
import numpy as np

import Config
from Misc import BoundingBox, Image

#input detected bounding boxes (text file)
#get result by function detect

class DetectorDay:

    def __init__(self, result_file, result_file_nclas):
        self.name = 'day'
        self.result = {}
        self.initialize(self.result, result_file)
        self.result_nclas = {}
        self.initialize(self.result_nclas, result_file_nclas)
        self.need_nclass = [6, 11, 46, 78, 86, 96]

    def initialize(self, result, result_file):
        f = open(result_file, 'r')
        lines = f.readlines()
        for i in range(0, len(lines)):
            split = lines[i].split(' ')
            name = split[0]
            video_id = int(name.split('/')[0])
            frame_id = int(name.split('/')[1][:-4][7:])
            if not video_id in result.keys():
                result[video_id] = {}
            result[video_id][frame_id] = []
            for j in range(1, len(split) - 1, 5):
                box = BoundingBox(float(split[j + 0]), float(split[j + 1]), float(split[j + 2]), float(split[j + 3]),
                                  float(split[j + 4]))
                result[video_id][frame_id].append(box)

        for i in result.keys():
            ave_imgs = os.listdir(Config.data_path + '/average_image/' + str(i))
            for j in range(1, len(ave_imgs) + 1):
                if j not in result[i].keys():
                    result[i][j] = []

    def detect(self, video_id, frame_id):
        if (video_id in self.need_nclass):
            return self.result_nclas[video_id][frame_id]
        else:
            return self.result[video_id][frame_id]

class DetectorNight:

    def __init__(self, result_path):
        self.name = 'night'
        self.result_path = result_path
        self.night_videos = np.zeros(101)
        self.result = {}
        data_names = os.listdir(self.result_path)
        for data_name in data_names:
            video_id = int(data_name.split('.')[0])
            self.night_videos[video_id] = 1
            self.result[video_id] = {}
            with open(self.result_path + '/' + data_name, 'rb') as file:
                data = pickle.load(file, encoding='latin1')
                for i in range(0, len(data['bboxes'])):
                    self.result[video_id][i + 1] = []
                    for j in range(0, len(data['bboxes'][i][1])):
                        box = data['bboxes'][i][1][j]
                        self.result[video_id][i + 1].append(BoundingBox(box[0], box[1], box[2], box[3], box[4]))

    def detect(self, video_id, frame_id):
        return self.result[video_id][frame_id]

    def checkNight(self, video_id):
        return (self.night_videos[video_id] == 1)

class DayNightDetector:

    def __init__(self):
        self.night_videos = np.zeros(101, np.int)
        self.video_path = Config.data_path + '/average_image'
        self.temp_initialize()

    def temp_initialize(self):
        nights = [2, 4, 7, 10, 22, 36, 37, 42, 44, 50, 53, 59, 61, 62, 66, 73, 74, 77, 94]
        for video_id in nights:
            self.night_videos[video_id] = 1

    def checkNight(self, video_id):
        return (self.night_videos[video_id] == 1)

    def initialize(self):
        f = open('gt.txt', 'r')
        cc = []
        for video_id in range(1, 101):
            print(video_id)
            image = cv2.cvtColor(Image.load(self.video_path + '/' + str(video_id) + '/average10.jpg'), cv2.COLOR_BGR2RGB)
            image = image[: int(image.shape[0] / 2), :]
            w = image.shape[1] // 5
            h = image.shape[0] // 5
            image = cv2.resize(image, (w, h))
            hsvIm = np.array(image)
            for i in range(0, image.shape[0]):
                for j in range(0, image.shape[1]):
                    rgb = image[i][j] / 255.0
                    hsv = colour.RGB_to_HSV(rgb)
                    hsv = [int(hsv[0] * 360), int(hsv[1] * 255), int(hsv[2] * 255)]
                    hsvIm[i][j] = hsv

            hBin = 360
            vBin = 255
            hHist, hX = np.histogram(hsvIm[:, :, 0].flatten(), hBin, (0, 360))
            vHist, vX = np.histogram(hsvIm[:, :, 2].flatten(), vBin, (0, 255))

            nH = (np.sum(hHist[0: 72]) + np.sum(hHist[288: ])) / (h * w)
            nV = np.sum(vHist[124:]) / (h * w)


            print(nH, nV)

            if nV < 0.289:
                result = 'night'
            else:
                result = 'day'
            gt = f.readline()[:-1]
            print(result, gt)
            cc.append((nH, nV, gt))

        dayList = [x for x in cc if (x[2] == 'day')]
        nightList = [x for x in cc if (x[2] == 'night')]
        plt.scatter([x[0] for x in dayList], [x[1] for x in dayList], c = 'r')
        plt.scatter([x[0] for x in nightList], [x[1] for x in nightList], c = 'b')
        plt.show()

class DetectorDay5fps:

    def __init__(self, result_file):
        self.name = 'day'
        f = open(result_file, 'r')
        lines = f.readlines()

        self.result = {}
        for i in range(0, len(lines)):
            split = lines[i].split(' ')
            name = split[0]
            video_id = int(name.split('/')[0])
            frame_id = int(name.split('/')[1][:-4][7:])
            if not video_id in self.result.keys():
                self.result[video_id] = {}
            self.result[video_id][frame_id] = []
            for j in range(1, len(split) - 1, 5):
                box = BoundingBox(float(split[j + 0]), float(split[j + 1]), float(split[j + 2]), float(split[j + 3]),
                                  float(split[j + 4]))
                self.result[video_id][frame_id].append(box)

        for i in self.result.keys():
            ave_imgs = os.listdir(Config.data_path + '/average_image_10fps/' + str(i))
            for j in range(1, len(ave_imgs) + 1):
                if j not in self.result[i].keys():
                    self.result[i][j] = []

    def detect(self, video_id, frame_id):
        return self.result[video_id][frame_id]

class JapDetector:
    category = [{"id": 1, "name": "pedestrian", "type": "thing", "supercategory": "person"},
                {"id": 2, "name": "rider", "type": "thing", "supercategory": "person"},
                {"id": 3, "name": "car", "type": "thing", "supercategory": "vehicle"},
                {"id": 4, "name": "truck", "type": "thing", "supercategory": "vehicle"},
                {"id": 5, "name": "bus", "type": "thing", "supercategory": "vehicle"},
                {"id": 6, "name": "motorcycle", "type": "thing", "supercategory": "vehicle"},
                {"id": 7, "name": "bicycle", "type": "thing", "supercategory": "vehicle"}]
    def __init__(self, file_path, all_path):
        self.name = 'both'
        with open(file_path, 'r') as f:
            self.boxes = json.load(f)
        with open(all_path, 'r') as f:
            self.paths = json.load(f)
        print(len(self.boxes))
        print(self.paths.keys())
        print(self.paths['images'])
        print(self.paths['videos'])
        print(self.boxes[0])
        self.result = {}
        for box in self.boxes:
            x, y, w, h = box['bbox']
            rbox = BoundingBox(x, y, x + w, y + h, box['score'])
            if box['category_id'] > 0:
                print(box['image_id'])
                image_info = self.paths['images'][box['image_id']]
                file_name = image_info['file_name']
                split = file_name.split('/')
                video_id = int(split[1])
                frame_id = int(split[2][7:-4])
                if not video_id in self.result.keys():
                    self.result[video_id] = {}
                if not frame_id in self.result[video_id].keys():
                    self.result[video_id][frame_id] = []
                self.result[video_id][frame_id].append(rbox)

        for i in self.result.keys():
            ave_imgs = os.listdir(Config.data_path + '/average_image/' + str(i))
            for j in range(1, len(ave_imgs) + 1):
                if j not in self.result[i].keys():
                    self.result[i][j] = []

    def detect(self, video_id, frame_id):
        return self.result[video_id][frame_id]

if __name__ == '__main__':

    detectorDay = DetectorDay(Config.data_path + '/32.txt', Config.data_path + '/32_result_8_3_3_nclas.txt')
    detector = detectorDay
    f = open(Config.data_path + '/32.txt')
    lines = f.readlines()
    for lines in lines:
        im_name = lines.split(' ')[0]
        video_id = int(im_name.split('/')[0])
        frame_id = int(im_name.split('/')[1][7:-4])
        img_paths = os.listdir(Config.data_path + '/average_image/' + str(video_id))
        print(video_id, frame_id)
        im = cv2.cvtColor(cv2.imread(Config.data_path + '/average_image/' + str(video_id) +'/average' + str(frame_id) + '.jpg'), cv2.COLOR_BGR2RGB)
        boxes = detector.detect(video_id, frame_id)
        for j in range(0, len(boxes)):
            if (boxes[j].score > 0):
                im = cv2.rectangle(im, (boxes[j].x1, boxes[j].y1), (boxes[j].x2, boxes[j].y2), (0, 255, 0), 3)
                im = cv2.putText(im, "%.2f" % (boxes[j].score), (boxes[j].x1, boxes[j].y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                plt.imshow(im)
        plt.show()
    f.close()