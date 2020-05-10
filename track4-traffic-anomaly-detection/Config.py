#data_path = 'E:/Thesis/BackupPlan/Data'
#data_path = 'E:/APCS/Thesis/Codes/Data'
#data_path1 = '/media/tuanbi97/Tuanbi97/APCS/Thesis/Codes/AICityChallenge2018_rank2/py-faster-rcnn/data'
data_path = 'F:\\Workspace\\aicity2020\\preprocessed_data'
dataset_path = 'F:\\Datasets\\AIC20_track4\\test-data'
#dataset_path = 'E:/Datasets/aic19-track3-test-data'
fps = 30
output_path = data_path + '\\output_demo'

box_threshold = 0.5
aevent_overlap_ratio = 0.2
aevent_iou = 0.2
threshold_anomaly_finish = 23
threshold_anomaly_most_idle = 10
threshold_anomaly_freq = 0.4
threshold_anomaly_least_time = 60
threshold_join_box = 0.7

threshold_proposal_least_time = 30
threshold_proposal_merge = 20
threshold_anomaly_merge = 30

start_offset = 10