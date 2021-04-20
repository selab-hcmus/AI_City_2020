dataset_path = '~/Datasets/AI city challenge/AIC20_track4/test-data'
data_path = '~/AI_City_2020/track4-traffic-anomaly-detection/preprocessed_data'
output_path = data_path + '/output_demo'

fps = 30
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