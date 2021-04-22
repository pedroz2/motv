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
    is_frame_good, frame = video.read(0)
    prev_f = frame

    # User draws initial frame bounding box here (x,y,w,h)
    bounding_box = np.array([[0.], [0.], [0.] ,[0.]])

    # Initialize KF
    # kf = KalmanFilter(initial_bb=bounding_box, 
    #                   dt=1.0, covar=1.0, 
    #                   proc_noise=4.0, alpha=0.98, 
    #                   r_1_xy=2.0, r_1_wh=5.0, 
    #                   r_2_xy=25.0, r_2_wh=50.0)
    # sift = Sift(frame)

    while is_frame_good:
        cv2.imshow('video 0', frame)
        is_frame_good, frame = video.read()
        # """
        # Correct filter with new bounding box on previous frame
        # Predict bounding box in new frame
        # """
        # predicted_box = kf.update(bounding_box)

        # """
        # Extract features using sift in previous image
        # Match features extracted in previous image with current image (updates bounding box too)
        # """
        # features = sift.extract_features(prev_f, bounding_box)
        # bounding_box = sift.match_features(f)

        # # Update previous frame
        # prev_f = f

    video.release()
    cv2.destroyAllWindows()
