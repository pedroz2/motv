import os
import pdb
import cv2
import numpy as np
import matplotlib.pyplot as plt
from kalman_filter import KalmanFilter
from sift_model import Sift
from os import path

DOWNLOAD = 'wget https://data.nvision2.eecs.yorku.ca/PIE_dataset/PIE_clips/set01/video_0001.mp4 -P dataset/'
DATASET = 'dataset/video_0001.mp4'

def downloadDataset():
    print("Checking for", DATASET)
    print("Exists:", path.exists(DATASET))

    if not path.exists(DATASET):
        print("Dataset does not exist. Downloading.\n\n")
        os.system(DOWNLOAD)
        

def loadFrames(vid_name):
    return cv2.VideoCapture(vid_name)


if __name__ == "__main__":
    #Download video if not already present.
    downloadDataset()

    print("\n\nDataset ready.")
    print("First frame loaded. User must pick bounds.")

    video = loadFrames(DATASET)
    
    # Loads initial frame into 'frame'
    is_frame_good, frame = video.read()

    # User draws initial frame bounding box here (sx,sy,ex,ey)
    bounding_box = np.array([ [ [850], [670], [1270], [1050]] ]) # hard coded initial bb of white car
    # pdb.set_trace()
    cv2.rectangle(  frame, (bounding_box[0][0][0], bounding_box[0][1][0]), 
                    (bounding_box[0][2][0], bounding_box[0][3][0]),
                    (255,0,0), 2)

    # Initialize KF
    kf = KalmanFilter(initial_bb=bounding_box[0], 
                      dt=1.0, covar=1.0, 
                      proc_noise=4.0, alpha=0.98, 
                      r_1_xy=2.0, r_1_wh=5.0, 
                      r_2_xy=25.0, r_2_wh=50.0)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sift = Sift(gray_frame)

    while is_frame_good:
        cv2.imshow('video 0', frame)
        prev_f = frame
        is_frame_good, frame = video.read()

        # Redraw frame with new bounding box
        cv2.rectangle(  frame, (bounding_box[0][0][0], bounding_box[0][1][0]), 
                    (bounding_box[0][2][0], bounding_box[0][3][0]),
                    (255,0,0), 2)

        """
        Correct filter with new bounding box on previous frame
        Predict bounding box in new frame
        """
        bounding_box[0] = kf.update(bounding_box[0])

        """
        Extract features using sift in previous image
        Match features extracted in previous image with current image (updates bounding box too)
        """
        pdb.set_trace()
        features = sift.extract_features(bounding_box[0])
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        matches = sift.match_features(gray_frame)
        bounding_box[0] = sift.get_new_bounding_box(matches)

        # Update previous frame

        cv2.waitKey(25) # delays next video frams (ms)

    video.release()
    cv2.destroyAllWindows()
