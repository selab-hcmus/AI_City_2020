import utils
utils.set_up_matplotlib()
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
import os
import time
import numpy as np
import pickle
import cv2


from models import generator
from utils import DataLoader, load, save, psnr_error, diff_mask
from constant import const
import evaluate

def calculate_score(psnrs):
   scores = np.array([], dtype=np.float32) 
   distance = psnrs
   if (distance.max() - distance.min())!=0:
      distance = (distance - distance.min()) / (distance.max() - distance.min())
   else:
      distance = (distance - 0) / (distance.max() - 0)
   scores = np.concatenate((scores[:], distance[DECIDABLE_IDX:]), axis=0)
   return scores

def visualize(inputs, gt_frame, pred_frame, scores, frame_order):
   frame_index = frame_order[-1]
   length = len(scores)
   threshold = 0.5
   gs = gridspec.GridSpec(3, 4)
   ax1 = plt.subplot(gs[0, 0])
   ax2 = plt.subplot(gs[0, 1])
   ax3 = plt.subplot(gs[1, 0])
   ax4 = plt.subplot(gs[1, 1])
   ax5 = plt.subplot(gs[:2, 2:4])
   ax6 = plt.subplot(gs[2:, :])

   #input blend
   motion = (inputs[..., (num_his-2)*3: (num_his-1)*3][0] + 1) / 2.0 
   ax1.imshow((inputs[..., (num_his-2)*3: (num_his-1)*3][0] + 1) / 2.0)
   ax1.set_axis_off()
   ax1.set_title('motion')
   #input current
   normal = (inputs[..., (num_his-1)*3: num_his*3][0] + 1) / 2.0
   ax2.imshow((inputs[..., (num_his-1)*3: num_his*3][0] + 1)/2.0)
   ax2.set_axis_off()
   ax2.set_title('normal')
   #pred
   predict = (pred_frame[0] + 1) / 2.0
   ax3.imshow((pred_frame[0]+1)/2.0)
   ax3.set_axis_off()
   ax3.set_title('predict')
   #gt
   gt = (gt_frame + 1) / 2.0
   ax4.imshow((gt_frame +1)/ 2.0)
   ax4.set_axis_off()
   ax4.set_title('gt')

   img1 = (gt_frame+1)/2.0
   img2 = (pred_frame[0]+1)/2.0
   error_r = np.fabs(np.subtract(img2[:,:,0], img1[:,:,0]))
   error_g = np.fabs(np.subtract(img2[:,:,1], img1[:,:,1]))
   error_b = np.fabs(np.subtract(img2[:,:,2], img1[:,:,2]))
   lum_img = np.maximum(np.maximum(error_r, error_g), error_b)
   # Uncomment the next line to turn the colors upside-down
   #lum_img = np.negative(lum_img);
   ax5.imshow(lum_img)
   ax5.set_axis_off()
   ax5.set_title('diff')

   #compute scores

   ax6.plot(frame_order, scores[0: len(frame_order)], label="scores")
   ax6.axis([0, length, 0, 1])
   cv2.imwrite('../images/{}_{}/{}_gt.png'.format(dataset_name, video_name, '%04d'%(frame_index)), gt * 255)
   cv2.imwrite('../images/{}_{}/{}_motion.png'.format(dataset_name,video_name,'%04d'%(frame_index)), motion * 255)
   cv2.imwrite('../images/{}_{}/{}_predict.png'.format(dataset_name, video_name, '%04d'%(frame_index)), predict * 255)
#    plt.savefig('../images/{}_{}/{}.png'.format(dataset_name,video_name,'%04d'%(frame_index)), dpi=200)
   plt.clf()

slim = tf.contrib.slim

os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = const.GPU

dataset_name = const.DATASET
test_folder = const.TEST_FOLDER

num_his = const.NUM_HIS
height, width = const.HEIGHT, const.WIDTH

snapshot_dir = const.SNAPSHOT_DIR
psnr_dir = const.PSNR_DIR
evaluate_name = const.EVALUATE
scores = np.array([], dtype=np.float32)
# gt_loader = evaluate.GroundTruthLoader()
# gt_loader.AI_CITY_VIDEO_START = 20 * 30
# gt = evaluate.get_gt(dataset=dataset_name)
print(const)
# define dataset
with tf.name_scope('dataset'):
    test_video_clips_tensor = tf.placeholder(shape=[1, height, width, 3 * (num_his + 1)],
                                             dtype=tf.float32)
    test_inputs = test_video_clips_tensor[..., 0:num_his*3]
    test_gt = test_video_clips_tensor[..., -3:]
    print('test inputs = {}'.format(test_inputs))
    print('test prediction gt = {}'.format(test_gt))

# define testing generator function and
# in testing, only generator networks, there is no discriminator networks and flownet.
with tf.variable_scope('generator', reuse=None):
    print('testing = {}'.format(tf.get_variable_scope().name))
    test_outputs = generator(test_inputs, layers=const.LAYERS, output_channel=3)
    test_psnr_error = psnr_error(gen_frames=test_outputs, gt_frames=test_gt)
    diff_mask_tensor = diff_mask(test_outputs, test_gt)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    # dataset
    data_loader = DataLoader(test_folder, height, width)

    # initialize weights
    sess.run(tf.global_variables_initializer())
    print('Init global successfully!')

    # tf saver
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=None)

    restore_var = [v for v in tf.global_variables()]
    loader = tf.train.Saver(var_list=restore_var)
    load(loader, sess, snapshot_dir)
    videos_info = data_loader.videos
    num_videos = len(videos_info.keys())
    print('Num videos: ', num_videos)
    total = 0
    psnr_records = []

    if const.BLEND_MOTION == 0:
        DECIDABLE_IDX = const.NUM_HIS
    else:
        DECIDABLE_IDX = 0

    for video_name, video in videos_info.items():
        if video_name != '46':
            continue
        result_dir = '../images/{}_{}/'.format(dataset_name, video_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        length = video['length']
        total += length

        # print(length)
        # assert 1==2

        if const.BLEND_MOTION == 0:
            start = num_his
        else:
            start = 0

        psnrs = np.empty(shape=(length,), dtype=np.float32)

        #calc scores
        for i in range(start, length - 1):
            # break
            video_clip = data_loader.get_video_clips(video_name, i - start, i + 1) # video clip size is (W,H,(4+1)*3)
            psnr = sess.run(test_psnr_error, feed_dict={test_video_clips_tensor: video_clip[np.newaxis, ...]})
            psnrs[i] = psnr
            print('video = {} / {}, i = {} / {}, psnr = {:.6f}'.format(
                    video_name, num_videos, i, length, psnr))

        psnrs[0:num_his] = psnrs[num_his]
        psnr_records.append(psnrs)
        scores = calculate_score(psnrs)

        f = open(result_dir + 'scores.txt','w')
        for score in scores:
            f.write(str(score) + '\n')
        f.close()

        f = open(result_dir + 'psnr.txt','w')
        f.write(str(psnrs) + '\n')
        f.close()
        # psnrs.dump(result_dir + 'psnr.pkl')

        # visualize
        frame_order = []
        for i in range(start, length - 1):
            # if i < 200 or i > 260:
                # continue
            video_clip = data_loader.get_video_clips(video_name, i - start, i + 1)  # video clip size is (W,H,(4+1)*3)
            inputs, pred_frame, diff = sess.run([test_inputs, test_outputs, diff_mask_tensor],
                                                feed_dict={test_video_clips_tensor: video_clip[np.newaxis, ...]})
            frame_order.append(i)
            gt_frame = video_clip[:, :, -3:]
            visualize(inputs=inputs, gt_frame=gt_frame, pred_frame=pred_frame,
                      scores=scores,
                      frame_order=frame_order)
