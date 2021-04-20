import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import Config
from AnomalyDetector import AnomalyDetector
from Detectors import DetectorDay, DetectorNight, DayNightDetector
from MaskList import MaskList
from Misc import Image
from StableFrameList import StableFrameList

#Initialize detector
print('Parse detector result ...')
dayNightDetector = DayNightDetector()
detectorDay = DetectorDay(Config.data_path + '/result_8_3_3_clas.txt', Config.data_path + '/result_8_3_3_nclas.txt')
detectorNight = DetectorNight(Config.data_path + '/extracted-bboxes-dark-videos')
anomalyDetector = AnomalyDetector()
stableList = StableFrameList(Config.data_path + '/unchanged_scene_periods.json')
maskList = MaskList(Config.data_path + '/masks_refine_v3')

for video_id in range(1, 101):
    print("Processing video ", video_id)
    detector = detectorDay
    if dayNightDetector.checkNight(video_id):
        detector = detectorNight

    stableIntervals = stableList[video_id]
    print(stableIntervals)
    confs = {}
    print(detector.name)

    #anomaly save file
    if not os.path.exists(Config.output_path + '/' + str(video_id)):
        os.makedirs(Config.output_path + '/' + str(video_id))
    f = open(Config.output_path + '/' + str(video_id) + '/anomaly_events.txt', 'w')

    #loop all stable intervals
    for scene_id in range(1, len(stableIntervals) + 1):
        l, r = stableIntervals[scene_id - 1]
        sl = int(l / Config.fps) + 1
        sr = int(r / Config.fps)
        sceneMask = maskList[(video_id, scene_id)]

        #create output folder
        if not os.path.exists(Config.output_path + '/' + str(video_id) + '/' + str(scene_id)):
            os.makedirs(Config.output_path + '/' + str(video_id) + '/' + str(scene_id))

        # output folder: output / video_id / scene_id / stuffs
        # output: average + boxes, gray_boxes before, gray_boxes after mask

        for frame_id in range(sl, sr):
            ave_im = Image.load(Config.data_path + '/average_image/' + str(video_id) + '/average' + str(frame_id) + '.jpg')
            boxes = detector.detect(video_id, frame_id)
            for box in boxes: box.applyMask(sceneMask)

            box_im = Image.addBoxes(ave_im, boxes)

            if detector.name == 'night':
                Image.save(box_im, Config.output_path + '/' + str(video_id) + '/' + str(scene_id) + '/night_average' + format(frame_id, '03d') + '.jpg')

            else:
                Image.save(box_im, Config.output_path + '/' + str(video_id) + '/' + str(scene_id) + '/day_average' + format(frame_id, '03d') + '.jpg')

            #detect anomaly event in scene
            anomalyDetector.addBoxes(boxes, frame_id) #input detected boxes => list of anomaly event
            detectedAnomalyEvents, conf = anomalyDetector.examineEvents(video_id, scene_id, frame_id, frame_id == sr - 1, f)

            event_im = anomalyDetector.drawEvents(box_im)

            Image.save(event_im, Config.output_path + '/' + str(video_id) + '/' + str(scene_id) + '/events' + format(frame_id, '03d') + '.jpg')
            confs[frame_id] = conf

    f.close()
    #output anomaly graph text before, anomaly_graph_after, anomaly_graph before, anomaly_graph after, result metric
    print(confs)
    f = plt.figure()
    plt.plot(list(confs.keys()), list(confs.values()), lw=4)
    plt.xlabel('Frame')
    plt.xlim(left=0)
    plt.ylabel('Confidence')
    plt.ylim(bottom=0)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
    f.savefig(Config.output_path + '/' + str(video_id) + '/' + str(video_id) + '_anomaly.pdf', bbox_inches='tight')
    plt.close(f)
    f = open(Config.output_path + '/' + str(video_id) + '/' + str(video_id) + '_anomaly.txt', 'w')
    for frame_id, conf in confs.items():
        f.write(str(frame_id) + ' ' + str(conf) + '\n')
    f.close()
