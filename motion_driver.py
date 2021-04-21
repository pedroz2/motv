import os
import pdb
import cv2
import numpy as np
import matplotlib.pyplot as plt
from kalman_filter import KalmanFilter
from sift_model import Sift

if __name__ == "__main__":
    # TODO: Load in dataset here  
    frames = np.array([])

    # TODO: Load initial frame
    prev_f = frames[0]

    # User draws initial frame bounding box here (x,y,w,h)
    bounding_box = np.array([[0.], [0.], [0.] ,[0.]])

    # Initialize KF
    kf = KalmanFilter(initial_bb=boundin_box, 
                      dt=1.0, covar=1.0, 
                      proc_noise=4.0, alpha=0.98, 
                      r_1_xy=2.0, r_1_wh=5.0, 
                      r_2_xy=25.0, r_2_wh=50.0)
    sift = Sift(initial_img)

    for f in frames:
        """
        Correct filter with new bounding box on previous frame
        Predict bounding box in new frame
        """
        predicted_box = kf.update(bounding_box)

        """
        Extract features using sift in previous image
        Match features extracted in previous image with current image (updates bounding box too)
        """
        features = sift.extract_features(prev_f, bounding_box)
        bounding_box = sift.match_features(f)

        # Update previous frame
        prev_f = f
