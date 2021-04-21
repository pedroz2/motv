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

    # User draws initial frame bounding box here
    bounding_box = np.array([0, 0, 0 ,0])

    # Initialize KF
    kf = KalmanFilter(initial_img, bounding_box)
    sift = Sift(initial_img)

    for f in frames:
        """
        Correct filter with new bounding box on previous frame
        Predict bounding box in new frame
        """
        kf.correct_filter(prev_f, bounding_box)
        predicted_box = kf.predict_filter(f)

        """
        Extract features using sift in previous image
        Match features extracted in previous image with current image (updates bounding box too)
        """
        features = sift.extract_features(prev_f, bounding_box)
        bounding_box = sift.match_features(f)

        # Update previous frame
        prev_f = f