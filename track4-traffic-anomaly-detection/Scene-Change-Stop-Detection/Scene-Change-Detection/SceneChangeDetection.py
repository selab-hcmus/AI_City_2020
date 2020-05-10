import argparse
import sys
import numpy as np
import cv2
import json
import os
import glob
from tqdm import tqdm


def LBP(frame):
    for i in range(len(frame)):
        for j in range(len(frame[0])):
            if i==0 or j==0 or i==len(frame)-1 or j==len(frame[0])-1:
                continue
            try: 
                gc=frame[i][j]
                newvalue=0
                if frame[i-1][j-1]>=gc:
                    newvalue+=1
                if frame[i-1][j]>=gc:
                    newvalue+=2
                if frame[i-1][j+1]>=gc:
                    newvalue+=4
                if frame[i][j+1]>=gc:
                    newvalue+=8
                if frame[i+1][j+1]>=gc:
                    newvalue+=16
                if frame[i+1][j]>=gc:
                    newvalue+=32
                if frame[i+1][j-1]>=gc:
                    newvalue+=64
                if frame[i][j-1]>=gc:
                    newvalue+=128

                frame[i][j]=newvalue
            except:
                print i, j
    return frame


def getCuts(file_name, cap):
    begin_id = 0
    background_alpha=0.01
    alpha = 0.1
    thresh = 70000

    cuts = []
    cnt=0
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', 600, 350)
    cv2.namedWindow('background frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('background frame', 600, 350)
    cap.set(cv2.CAP_PROP_POS_FRAMES, begin_id)

    _, background_frame = cap.read()
    background_frame = cv2.cvtColor(background_frame, cv2.COLOR_BGR2YUV)
    background_frame[:, :, 0] = cv2.equalizeHist(background_frame[:, :, 0])
    background_frame = cv2.cvtColor(background_frame, cv2.COLOR_YUV2BGR)
    mean_frame = background_frame.copy()

    for frm_id in range(begin_id+1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        frame[:, :, 0] = cv2.equalizeHist(frame[:, :, 0])
        frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)

        background_frame = cv2.addWeighted(background_frame, 1-background_alpha, frame, background_alpha, 0)
        mean_frame = cv2.addWeighted(mean_frame, 1-alpha, frame, alpha, 0)

        # Our operations on the frame come here
        if frm_id % 30 == 0:
            lbp = LBP(cv2.cvtColor(mean_frame, cv2.COLOR_BGR2GRAY))
            background_lbp = LBP(cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY))

            try:
                lbph = cv2.calcHist([lbp],[0],None,[256],[0,256])
                background_lbph = cv2.calcHist([background_lbp],[0],None,[256],[0,256])

                diff=0
                for i in range(256):
                    diff+=abs(lbph[i]-background_lbph[i])

                print frm_id, diff[0]

                cv2.imshow('frame', mean_frame)
                cv2.imshow('background frame', background_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if diff[0]>=thresh:
                    cnt+=1
                    print "CUT "+str(cnt)+" Detected at frame "+str(frm_id)
                    cuts.append(frm_id)
                    background_frame = frame.copy()

            except UnboundLocalError:
                pass

    print 'Found %d scene changes.' % cnt
    cuts_file = os.path.join(os.path.dirname(file_name), 'cuts', os.path.basename(file_name) + '.json')
    with open(cuts_file, 'w') as f:
        json.dump(cuts, f)

    cap.release()
    #cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detects all scene changing periods in videos.')
    parser.add_argument('vi_or_dir',
                        help='Videos or directory containing videos to be processed.',
                        nargs='+',
                        type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    videos = args.vi_or_dir

    if len(videos) > 1:
        assert any([os.path.isdir(video) for video in videos]), 'Multiple inputs option is only for inputing videos.'
    assert all([os.path.dirname(video) == os.path.dirname(videos[0]) for video in videos]), 'All videos should be placed in the same directory.'

    if len(videos) == 1 and os.path.isdir(videos[0]):
        videos = glob.glob(os.path.join(videos[0], '*.mp4'))

    cuts_dir = os.path.join(os.path.dirname(videos[0]), 'cuts')
    if not os.path.isdir(cuts_dir):
        os.mkdir(cuts_dir)


    for video_name in videos:
        cap = cv2.VideoCapture(video_name)
        print('Processing file name: %s' % video_name)
        getCuts(video_name, cap)
