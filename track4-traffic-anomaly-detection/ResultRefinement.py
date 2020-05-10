import Config
import numpy as np
import json

class Interval:

    def __init__(self, scene_id, l, r, score):
        self.scene_id = scene_id
        self.l = l
        self.r = r
        self.score = score

    def expand(self, interval):
        self.l = min(self.l, interval.l)
        self.r = max(self.r, interval.r)

    def overlap(self, interval):
        if interval.r >= self.l: return True
        return False

    def overlapInterval(self, l, r):
        if r >= self.l: return True
        return False

def refineResult(output_path):
    #final result
    #Merge overlapped region
    with open(Config.data_path + '/stop_scene_periods.json', 'r') as f:
        stops = json.load(f)
    f = open(output_path + '/result_all.txt', 'w')
    for video_id in range(1, 100):
        g = open(output_path + '/' + str(video_id) + '/anomaly_events.txt')
        lines = g.readlines()
        intervals = []


        #merge in scene
        for line in lines:
            vid, scene_id, l, r, score = (float(x) for x in line.split(' '))
            intervals.append(Interval(scene_id, l, r, score))

        check = np.zeros(len(intervals))
        for i in range(0, len(intervals)):
            for j in range(0, i):
                if (intervals[i].overlap(intervals[j])):
                    intervals[i].expand(intervals[j])
                    check[j] = 1
                else:
                    if intervals[i].scene_id != intervals[j].scene_id:
                        if (intervals[i].overlapInterval(intervals[j].l, intervals[j].r + Config.threshold_anomaly_merge)):
                            intervals[i].expand(intervals[j])
                            check[j] = 1

        minv = 1000.0
        scorev = 0
        for i in range(0, len(intervals)):
            if (check[i] == 0):
                lc = 0
                for j in range(0, len(stops[str(video_id)])):
                    stop = stops[str(video_id)][j]
                    if intervals[i].l > stop[0] - 20 and intervals[i].l < stop[1] + 20:
                        lc = 1
                        break
                if lc == 0:
                    if minv > max(intervals[i].l - Config.start_offset, 0):
                        minv = max(intervals[i].l - Config.start_offset, 0)
                        scorev = intervals[i].score

        if (minv < 1000):
            f.write("%d %.2f %.2f\n" % (video_id, minv, scorev))

    f.close()

if __name__=='__main__':
    refineResult(Config.output_path)