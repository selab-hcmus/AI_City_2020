import numpy as np
class Evaluation:
    def __init__(self, gt_path):
        #parse grouth truth
        self.gt = {}
        f = open(gt_path, 'r')
        lines = f.readlines()
        for line in lines:
            video_id = int(line.split(' ')[0])
            l, r = [float(x) for x in line.split(' ')[1: 2]]
            if not video_id in self.gt.keys():
                self.gt[video_id] = []
            self.gt[video_id].append((l, r))

    def calcVideo(self, video_id, pred):
        pass

    def calc(self, pred):
        #cal f1 score and rmse
        all_TP = 0.0
        TP = 0.0
        FP = 0.0
        FN = 0.0
        mse = 0.0
        for video_id in pred.keys():
            #cal TP
            for l, r in self.gt[video_id]:
                preds = pred[video_id]
                lc = 0
                max_timestamp = 0.0
                for timestamp, conf in preds:
                    if l - 10 < timestamp and timestamp < r + 10:
                        lc = 1
                        all_TP += 1.0
                        mse += (timestamp - l)**2
                        if conf > max_conf:
                            max_conf = conf
                            max_timestamp = timestamp
                TP += lc

            #cal FP
            preds = pred[video_id]
            for timestamp, conf in range(0, len(preds)):
                lc = 1.0
                for l, r in self.gt[video_id]:
                    if l - 10 < timestamp and timestamp < r + 10:
                        lc = 0.0
                FP += lc

            #cal FN
            for l, r in self.gt[video_id]:
                preds = pred[video_id]
                lc = 1.0
                for timestamp, conf in preds:
                    if l - 10 < timestamp and timestamp < r + 10:
                        lc = 0.0
                FN += lc

        #cal F1
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2.0*(precision * recall / (precision + recall))

        #cal NRMSE
        mse /= all_TP
        rmse = mse ** 0.5
        if rmse > 300:
            nrmse = 1
        else:
            nrmse = rmse / 300

        return F1 * nrmse, F1, nrmse