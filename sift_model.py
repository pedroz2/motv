import numpy as np
import cv2
import pdb
from matplotlib import pyplot as plt


class Sift():
    def __init__(self, initial_img):
        self.frame = initial_img
        self.sift_obj = cv2.SIFT_create()
        self.keypoints_prev = None
        self.desc_prev = None
        self.bounding_box_prev = None

    def extract_features(self, bounding_box):
        # Find and save all features in frame (maybe just in the bounding box) 
        print('self.frame', self.frame.shape, self.frame.dtype)
        mask_img = np.zeros_like(self.frame)
        cv2.rectangle(mask_img, (bounding_box[0][0], bounding_box[1][0]), 
                    (bounding_box[2][0], bounding_box[3][0]),
                    (255), -1)
        keypoints, desc = self.sift_obj.detectAndCompute(self.frame, mask=mask_img)

        # Update prev state
        self.keypoints_prev = keypoints
        self.desc_prev = desc
        self.bounding_box_prev = bounding_box
        return keypoints

    def match_features(self, next_frame): 
        # Match all features in current frame
        print('next_frame', next_frame.shape, next_frame.dtype)
        mask_img = np.zeros_like(next_frame)
        cv2.rectangle(mask_img, (self.bounding_box_prev[0][0], self.bounding_box_prev[1][0]), 
                    (self.bounding_box_prev[2][0], self.bounding_box_prev[3][0]),
                    (255), -1)
        # desc = self.sift_obj.detect(next_frame, keypoints = self.keypoints_prev)
        keypoints, desc = self.sift_obj.detectAndCompute(self.frame, mask_img)
        
        pdb.set_trace()
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc, self.desc_prev, k=2)

        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])

        # Uncomment below for visualizing sift matches
        # cv2.drawMatchesKnn expects list of lists as matches.
        # img3 = cv2.drawMatchesKnn(self.frame, self.keypoints_prev, next_frame, keypoints, good, None)
        # plt.imshow(img3)

        # TODO: Make least squares
        bounding_box = []
        
        return matches, bounding_box