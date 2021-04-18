import cv2
import numpy as np

import Config
from Misc import Image

class MaskList:
    def __init__(self, file_path):
        self.file_path = file_path

    def __getitem__(self, key):
        video_id, scene_id = key
        mask = np.load(self.file_path + '/mask_' + str(video_id) + '_' + str(scene_id) + '.npy')
        return mask

if __name__ == '__main__':
    list = MaskList(Config.data_path + '/masks_refine_v3')
    video_id = 51
    scene_id = 1
    frame_id = 144
    mask = list[(video_id, 1)]
    im = cv2.cvtColor(Image.load(Config.data_path + '/average_image/' + str(video_id) + '/average' + str(frame_id) +'.jpg' ), cv2.COLOR_BGR2RGB)
    mask = np.dstack((mask, mask, mask))
    im = im * mask
    save_path = './51_mask_image.png'
    cv2.imwrite(save_path, cv2.cvtColor(im, cv2.COLOR_BGR2RGB))