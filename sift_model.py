import numpy as np
import cv2 as cv


class Sift():
    def __init__(self, initial_img):
        self.frame = initial_img

    def extract_features(self, bd_box):
        # Find and save all features in frame (maybe just in the bounding box)
        sift_obj = cv.SIFT_create()
        keypoints = sift_obj.detect(self.frame, bd_box)
        return keypoints
        

    def match_features(self, next_frame): 
        # Match all features in current frame
        sift_obj = cv.SIFT_create()
        kp, desc = sift_obj.detectAndCompute(next_frame)
        return desc