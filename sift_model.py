import numpy as np
import cv2
import pdb


class Sift():
    def __init__(self, initial_img):
        self.frame = initial_img
        self.sift_obj = cv2.SIFT_create()
        self.keypoints_prev = None

    def extract_features(self, bounding_box):
        # Find and save all features in frame (maybe just in the bounding box) 
        print('self.frame', self.frame.shape, self.frame.dtype)
        mask_img = np.zeros_like(self.frame)
        cv2.rectangle(mask_img, (bounding_box[0][0], bounding_box[1][0]), 
                    (bounding_box[2][0], bounding_box[3][0]),
                    (255), -1)
        keypoints = self.sift_obj.detect(self.frame, mask=mask_img)
        self.keypoints_prev = keypoints
        return keypoints
        

    def match_features(self, next_frame): 
        # Match all features in current frame
        print('next_frame', next_frame.shape, next_frame.dtype)
        desc = self.sift_obj.compute(next_frame, keypoints = self.keypoints_prev)
        return desc

    def get_new_bounding_box(self, desc):
        # do homogeneous least squares

        pdb.set_trace()
        Ax = np.hstack([np.ones(len(desc)), desc])
        np.hstack([kp1[matches[:,0],:2], kp2[matches[:,1],:2]])

        return new_bounding_box