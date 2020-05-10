import cv2
import Config
import matplotlib.pyplot as plt
import numpy as np
import json
import colour

class RoadMask:
    def __init__(self, mask_path, scene_path, im_path):
        self.mask_path = mask_path
        self.scene_path = scene_path
        self.im_path = im_path
        with open(scene_path, 'r') as f:
            self.stableList = json.load(f)
        self.refineMasks()

    def getMask(self, video_id, scene_id):
        mask = np.load(self.mask_path + '/mask_' + str(video_id) + '_' + str(scene_id) + '.npy')
        return mask

    def refineMask(self, im, mask):
        mask = mask > 0.3
        mask = (mask * 255).astype(int)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        im = cv2.GaussianBlur(im, (9, 9), 0)
        im = cv2.bilateralFilter(im, 9, 75, 75)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(im, cmap='gray')
        ax2.imshow(mask, cmap='gray')
        ax3.imshow(cv2.Canny(im, 0, 0), cmap='gray')
        plt.show()


    def refineMasks(self):
        for video_id in range(1, 101):
            stableIntervals = self.stableList[str(video_id)]
            for scene_id in range(1, len(stableIntervals) + 1):
                l, r = stableIntervals[scene_id - 1]
                l = int(l / Config.fps) + 1
                r = int(r / Config.fps)
                mask = self.getMask(video_id, scene_id)
                print(self.im_path + '/' + str(video_id) + '/average' + str(l) + '.jpg')
                im = cv2.cvtColor(cv2.imread(self.im_path + '/' + str(video_id) + '/average' + str(l+5) + '.jpg'), cv2.COLOR_BGR2RGB)
                self.refineMask(im, mask)
                break

        print(self.stableList)

if __name__ == '__main__':
    list = RoadMask(Config.data_path + '/masks', Config.data_path + '/unchanged_scene_periods.json', Config.data_path + '/average_image')

    # video_id = 1
    # mask = list.getMask(video_id, 1)
    # mask = (mask * 255).astype(int)
    # mask = np.dstack((mask, mask, mask))
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.imshow(mask)
    # im = cv2.cvtColor(cv2.imread(Config.data_path + '/average_image/' + str(video_id) + '/average145.jpg'),
    #                   cv2.COLOR_BGR2RGB)
    # im = cv2.addWeighted(im.astype(int), 1, mask, 0.5, 0.0)
    # ax2.imshow(im)
    # plt.show()

    # video_id = 19
    # im = cv2.cvtColor(cv2.imread(Config.data_path + '/average_image/' + str(video_id) + '/average10.jpg'), cv2.COLOR_BGR2RGB)
    # im = cv2.GaussianBlur(im, (9, 9), 0)
    # im = cv2.bilateralFilter(im, 9, 75, 75)
    # plt.imshow(im)
    # plt.show()