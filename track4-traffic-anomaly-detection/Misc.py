import cv2
import Config
import numpy as np

class BoundingBox:
    def __init__(self, x1, y1, x2, y2, score):
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)
        self.score = score

    def applyMask(self, mask):
        self.score = self.score * np.mean(mask[self.y1: self.y2, self.x1: self.x2])

    def area(self):
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def expand(self, box):
        self.x1 = min(self.x1, box.x1)
        self.y1 = min(self.y1, box.y1)
        self.x2 = max(self.x2, box.x2)
        self.y2 = max(self.y2, box.y2)
        self.score = 1.0 - (1.0 - self.score) * (1.0 - box.score)

class Image:

    @staticmethod
    def load(img_path):
        im = cv2.imread(img_path)
        #im = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        return im

    @staticmethod
    def calcConfident(boxes):
        conf = 1.0
        for box in boxes:
            if box.score > Config.box_threshold:
                conf *= (1.0 - box.score)
        conf = 1 - conf
        return conf

    @staticmethod
    def calcFrameConfident(events):
        conf = 1.0
        for key in events.keys():
            event = events[key]
            if event.status:
                conf *= (1.0 - event.region.score)
        conf = 1.0 - conf
        return conf

    @staticmethod
    def addBoxes(im, boxes):
        im = np.array(im)
        for box in boxes:
            if (box.score > Config.box_threshold):
                im = cv2.rectangle(im, (box.x1, box.y1), (box.x2, box.y2), (0, 255, 0), 2)
                im = cv2.putText(im, "%.2f" % (box.score), (box.x1, box.y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        return im

    @staticmethod
    def save(im, path):
        cv2.imwrite(path, im)

