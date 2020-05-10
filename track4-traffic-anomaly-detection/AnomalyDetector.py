import numpy as np
import cv2
from Misc import BoundingBox, Image
import Config

class AnomalyEvent:
    def __init__(self, box, time, id):
        self.id = id
        self.region = box
        self.start_time = time
        self.latest_update = time
        self.boxes = [box] #list of bounding box in anomaly event
        self.count = 1 #freq
        self.status = 0 #0 / 1 = suspect / anomaly

    def getConf(self):
        return self.region.score

    def boxIntersect(self, box1, box2):
        xmax = max(box1.x1, box2.x1)
        xmin = min(box1.x2, box2.x2)
        ymax = max(box1.y1, box2.y1)
        ymin = min(box1.y2, box2.y2)
        if (xmax > xmin) or (ymax > ymin):
            return 0
        else:
            return (xmin - xmax) * (ymin - ymax)

    def overlapRatio(self, box):
        return self.boxIntersect(self.region, box) / box.area()

    def IoU(self, box1, box2):
        intersect = self.boxIntersect(box1, box2)
        union = box1.area() + box2.area() - intersect
        return intersect / union

    def radiusRestrict(self, pivot, box):
        x1 = (pivot.x1 + pivot.x2) / 2.0
        y1 = (pivot.y1 + pivot.y2) / 2.0
        x2 = (box.x1 + box.x2) / 2.0
        y2 = (box.y1 + box.y2) / 2.0
        dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        radius = ((((pivot.x2 - pivot.x1) ** 2 + (pivot.y2 - pivot.y1) ** 2) ** 0.5) / 2 \
                + (((box.x2 - box.x1) ** 2 + (box.y2 - box.y1) ** 2) ** 0.5) / 2) * (4 / 5)
        if dist <= radius: return True, dist
        return False, dist

    def checkContains(self, box):
        # method 1
        # if (self.overlapRatio(box) > Config.aevent_overlap_ratio) or (self.radiusRestrict(box)):
        #     return True
        # else:
        #     return False

        # method 2
        # isInEvent = False
        # min_dist = 1000000000.0
        # for event_box in self.boxes:
        #     allow, dist = self.radiusRestrict(event_box, box)
        #     if allow:
        #         isInEvent = True
        #         if dist < min_dist: min_dist = dist
        # return isInEvent, min_dist

        # method 3
        # isInEvent = False
        # max_dist = -1.0
        # for event_box in self.boxes:
        #     IoU = self.IoU(event_box, box)
        #     if IoU > Config.aevent_iou:
        #         isInEvent = True
        #         if IoU > max_dist: max_dist = IoU
        # return isInEvent, max_dist

        #method 4
        isInEvent = False
        max_dist = -1.0
        for event_box in self.boxes:
            overlapRatio = self.boxIntersect(event_box, box) / box.area()
            if overlapRatio > Config.aevent_overlap_ratio:
                isInEvent = True
                if overlapRatio > max_dist: max_dist = overlapRatio
        return isInEvent, max_dist

    def expandRegion(self, box):
        needNew = True
        maxBox = None
        maxIoU = 0
        for abox in self.boxes:
            iou = self.IoU(abox, box)
            if iou > Config.threshold_join_box:
                needNew = False
                if iou > maxIoU:
                    maxIoU = iou
                    maxBox = abox
        if needNew:
            self.boxes.append(box)
        else:
            maxBox.expand(box)
        self.region.x1 = min(self.region.x1, box.x1)
        self.region.y1 = min(self.region.y1, box.y1)
        self.region.x2 = max(self.region.x2, box.x2)
        self.region.y2 = max(self.region.y2, box.y2)
        self.region.score = Image.calcConfident(self.boxes)

    def addBox(self, box, time):
        self.expandRegion(box)
        self.latest_update = time
        self.count += 1

class AnomalyDetector:
    def __init__(self):
        self.nextId = 0
        self.events = {}  # list of anomaly event
        self.prevEvents = {}

    def addBoxes(self, boxes, time):
        #print('before: ',self.events.keys())
        #print('len boxes: ', len(boxes))
        for box in boxes:
            #print('box_score:', box.score)
            if box.score < Config.box_threshold: continue
            #print('adding: ', box.x1, box.y1)
            lc = 0
            max_dist = -1.0
            pevent = None
            for key in self.events.keys():
                event = self.events[key]
                allow, dist = event.checkContains(box)
                if allow:
                    lc = 1
                    if max_dist < dist:
                        max_dist = dist
                        pevent = event

            if lc == 1:
                pevent.addBox(box, time)
            if lc == 0:
                self.events[self.nextId] = AnomalyEvent(box, time, self.nextId)
                self.nextId += 1
        #print('after: ', self.events.keys())

    def examineEvents(self, video_id, scene_id, time, isEnd, file):
        ret = []
        keys = [key for key in self.events.keys()]
        temp = {}
        for key in keys:
            event = self.events[key]
            if (time - event.latest_update > Config.threshold_anomaly_finish \
                    or (time > event.start_time and event.count / (time - event.start_time) < Config.threshold_anomaly_freq)) and isEnd == False:
                if event.status == 1: #anomaly event
                    #format: video_id scene_id start_time end_time confident
                    file.write(str(video_id) + ' ' + str(scene_id) + ' ' + str(event.start_time) + ' ' + str(time) + ' ' + str(event.getConf()) + '\n')
                self.events.pop(key)
            else:
                if time - event.latest_update < Config.threshold_anomaly_most_idle:
                    if event.status == 0 and time - event.start_time < Config.threshold_anomaly_least_time and time - event.start_time > Config.threshold_proposal_least_time:
                        minStartTime = event.start_time
                        for pKey in self.prevEvents.keys():
                            pEvent = self.prevEvents[pKey]
                            if event.start_time - pEvent.latest_update < Config.threshold_proposal_merge:
                                minStartTime = min(minStartTime, pEvent.start_time)
                        event.start_time = minStartTime
                    if time - event.start_time > Config.threshold_anomaly_least_time:
                        event.status = 1
                        ret.append(event)


            if isEnd:
                if event.status == 1: #anomaly event
                    #format: video_id scene_id start_time end_time confident
                    file.write(str(video_id) + ' ' + str(scene_id) + ' ' + str(event.start_time) + ' ' + str(time) + ' ' + str(event.getConf()) + '\n')
                else:
                    #update previous events for next scene
                    minStartTime = event.start_time
                    for pKey in self.prevEvents.keys():
                        pEvent = self.prevEvents[pKey]
                        if event.start_time - pEvent.latest_update < Config.threshold_proposal_merge:
                            minStartTime = min(minStartTime, pEvent.start_time)
                    event.start_time = minStartTime
                    if time - event.start_time > Config.threshold_anomaly_least_time:
                        event.status = 1
                        ret.append(event)
                    else:
                        temp[event.id] = event

                self.events.pop(key)

        if isEnd:
            self.prevEvents = temp

        currentConf = Image.calcFrameConfident(self.events)
        return ret, currentConf

    def drawEvents(self, im):
        #print(self.events.keys())
        for key in self.events.keys():
            event = self.events[key]
            if event.status == 0:
                #draw proposal
                im = cv2.rectangle(im, (event.region.x1, event.region.y1), (event.region.x2, event.region.y2),
                                   (255, 153, 51), 3)
                im = cv2.putText(im, "%d proposal" % (key), (event.region.x2 + 10, event.region.y1),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 153, 51), 2, cv2.LINE_AA)
            else:
                if event.status == 1:
                    im = cv2.rectangle(im, (event.region.x1, event.region.y1), (event.region.x2, event.region.y2), (255, 0, 0), 3)
                    im = cv2.putText(im, "%d %.2f" % (key, event.region.score), (event.region.x2 + 10, event.region.y1),
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
                    im = cv2.putText(im, "start: %.2f" % (event.start_time), (event.region.x1, event.region.y2 + 20),
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        return im