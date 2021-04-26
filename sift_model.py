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

        # TODO: Make homography
        bounding_box = []
        
        return bounding_box

    def expand_binarize(desc):
        '''
        Explicitly expand packed binary keypoint descriptors like AKAZE and ORB.
        You do not need to modify or worry about this.

        AKAZE and ORB return a descriptor that is binary. Usually one compares
        descriptors using the hamming distance (# of bits that differ). This is
        usually fast since one can do this with binary operators. On Intel
        processors, there's an instruction for this: popcnt/population count.

        On the other hand, this prevents you from actually implementing all the steps
        of the pipeline and requires you writing a hamming distance. So instead, we
        explicitly expand the feature from F packed binary uint8s to (8F) explicit 
        binary 0 or 1 descriptors. The square of the L2 distance of these
        descriptors is the hamming distance.
        
        Converts a matrix where each row is a vector containing F uint8s into their
        explicit binary form.
            
        Input - desc: matrix of size (N,F) containing N 8F dimensional binary
                    descriptors packed into N, F dimensional uint8s
        
        Output - binary_desc: matrix of size (N,8F) containing only 0s or 1s that 
                            expands this to be explicit
        '''
        N, F = desc.shape
        binary_desc = np.zeros((N,F*8))
        for i in range(N):
            for j in range(F):
                binary_desc[i,(j*8):((j+1)*8)] = _bits[desc[i,j]]
        return binary_desc