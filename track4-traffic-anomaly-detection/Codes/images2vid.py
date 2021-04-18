import glob

import cv2

img_array = []
for filename in glob.glob('/home/datthanh/future_frame/images/*.png'):
    print(filename)
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(filename)

out = cv2.VideoWriter('/home/datthanh/future_frame/images/project.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 15, size)
 
for i in range(len(img_array)):
    image = cv2.imread(img_array[i])
    out.write(image)
out.release()
