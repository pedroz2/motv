import os
import time
import pdb
import cv2
import numpy as np
import matplotlib.pyplot as plt
from kalman_filter import KalmanFilter
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

def DrawRectangle(img, bbox, color=(255, 0, 0), thickness=2):
    # Draw rectangle given XYWH bounding box
    X, Y, W, H = bbox[0,0], bbox[1,0], bbox[2,0], bbox[3,0]
    return cv2.rectangle(img, (X, Y), (X+W, Y+H), color, thickness)

def SkipToInteresting(cap):
    # Movement doesn't start until around 27s on video 1
    # 27s * 30fps = frame (27 * 30)
    clear_frames = 27 * 30
    for i in range(clear_frames):
        is_frame_good, frame = video.read()
        if not is_frame_good:
            return

class StaticCapture:
    # Emulates the VideoCapture interface, but on a set of static images
    # Seems to run a bit quicker than loading from the mp4
    def __init__(self):
        self.iteration = 0
    
    def read(self):
        if self.iteration > 500:
            return False, None
        self.iteration += 1
        img = cv2.imread(f'outimgs/out{self.iteration}.jpg')
        return True, img

def DrawBoundingBox(frame):
    # Function to let you quickly draw bounding boxes
    calib_win = 'bbox'
    cv2.namedWindow(calib_win)
    x_val = 0
    y_val = 0
    w_val = 0
    h_val = 0
    def x_cb(val): nonlocal x_val; x_val = val
    def y_cb(val): nonlocal y_val; y_val = val
    def w_cb(val): nonlocal w_val; w_val = val
    def h_cb(val): nonlocal h_val; h_val = val
    cv2.createTrackbar('X', calib_win, 0, frame.shape[0], x_cb)
    cv2.createTrackbar('Y', calib_win, 0, frame.shape[1], y_cb)
    cv2.createTrackbar('W', calib_win, 0, frame.shape[0], w_cb)
    cv2.createTrackbar('H', calib_win, 0, frame.shape[1], h_cb)
    original_frame = frame.copy()
    while True:
        bbox = np.array([ [x_val], [y_val], [w_val], [h_val]])
        frame = original_frame.copy()
        DrawRectangle(frame, bbox)
        cv2.imshow(calib_win, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

class Sift():
    def __init__(self, img_color, bbox):
        self.img_prev = None
        self.keypoints_prev = None
        self.desc_prev = None
        self.bbox_prev = None

        self.img_curr = None
        self.keypoints_curr = None
        self.desc_curr = None
        self.bbox_curr = None
        self.sift_obj = cv2.SIFT_create()

        # Setup initial images
        self.ProcessImg(img_color, bbox)

    def ProcessImg(self, img_color, bbox):
        # Convert to grayscale
        img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        # Store previous img, kp, desc, and bbox
        self.img_prev = self.img_curr
        self.keypoints_prev = self.keypoints_curr
        self.desc_prev = self.desc_curr
        self.bbox_prev = self.bbox_curr

        # Set current img, kp, desc, and bbox
        self.img_curr = img
        self.bbox_curr = bbox
        self.keypoints_curr, self.desc_curr = self.GetKpAndDesc(self.img_curr, bbox)

        if self.img_prev is None:
            return

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(self.desc_curr, self.desc_prev, k=2)
        good = []
        for m, n in matches:
            # TODO: tune this parameter
            if m.distance < 0.8 * n.distance:
                good.append([m])
        # Least Squares
        #  Currently using weighted least squares, biased towards center since 
        #  that seems to help a bit with avoiding the tracking getting 
        #  attached to a building
        num_match = len(good)
        A_x = np.ones((num_match, 2), dtype=np.float64)
        A_y = np.ones((num_match, 2), dtype=np.float64)
        b_x = np.ones((num_match), dtype=np.float64)
        b_y = np.ones((num_match), dtype=np.float64)
        X, Y, W, H = self.bbox_prev[0,0], self.bbox_prev[1,0], self.bbox_prev[2,0], self.bbox_prev[3,0]
        w_vals = np.ones((num_match), dtype=np.float64)
        cx = X + W/2.0
        cy = Y + H/2.0
        for i in range(len(good)):
            m, = good[i]
            #print(keypoints[m.queryIdx].pt, self.keypoints_prev[m.trainIdx].pt)
            b_x[i] = self.keypoints_curr[m.queryIdx].pt[0]
            b_y[i] = self.keypoints_curr[m.queryIdx].pt[1]
            w_vals[i] = (b_x[i] - cx) ** 2.0 + (b_y[i] - cy) ** 2.0
            A_x[i,1] = (self.keypoints_prev[m.trainIdx].pt[0] - X) / float(W)
            A_y[i,1] = (self.keypoints_prev[m.trainIdx].pt[1] - Y) / float(H)
        w_vals = np.max(w_vals) - w_vals
        A_x = A_x * w_vals[:,np.newaxis]
        A_y = A_y * w_vals[:,np.newaxis]
        b_x = b_x * w_vals
        b_y = b_y * w_vals
        # Least squares solutions, [X, W].T, and [Y, H].T
        lst_sln_x = np.linalg.lstsq(A_x, b_x, rcond=None)
        lst_sln_y = np.linalg.lstsq(A_y, b_y, rcond=None)
        pred_bb = np.array([ [lst_sln_x[0][0]], [lst_sln_y[0][0]], [lst_sln_x[0][1]], [lst_sln_y[0][1]]])

        # Draw Matches and show
        #   matched_img2 = np.zeros_like(self.img_curr)
        #   matched_img = cv2.drawMatchesKnn(self.img_curr, self.keypoints_curr, self.img_prev, self.keypoints_prev, good, flags=2, outImg=matched_img2)
        #   cv2.imshow('matched_img', matched_img)

        pred_bb = pred_bb.astype(np.int64)
        return pred_bb

    def GetKpAndDesc(self, img, bbox):
        # TODO: Figure out what to do about bounding box collapse
        #  SIFTING the whole 1080p image takes a very long time,
        #  but the bounding box collapses pretty easily if you only search
        #  inside of it due to the nature of least squares only searching
        #  within the box
        mask_img = np.ones_like(img)
        # Only use bounding box region as mask
        #   X, Y, W, H = bbox[0,0], bbox[1,0], bbox[2,0], bbox[3,0]
        #   margin = 0.0
        #   X = X - margin * W
        #   Y = Y - margin * H
        #   W = W + W * 2.0 * margin
        #   H = H + H * 2.0 * margin
        #   mask_bbox = np.array([[X], [Y], [W], [H]], dtype=np.int64)
        #   mask_img = np.zeros_like(img)
        #   DrawRectangle(mask_img, mask_bbox, color=(255), thickness=-1)
        return self.sift_obj.detectAndCompute(img, mask=mask_img)


if __name__ == '__main__':
    downloadDataset()
    # XYWH
    # bounding_box = np.array([ [867], [714], [394], [328]])
    bounding_box = np.array([ [915], [742], [298], [229]])

    # Initialize KF
    kf = KalmanFilter(initial_bb=bounding_box, 
                      dt=0.1, covar=0.1, 
                      proc_noise=0.0, alpha=0.98, 
                      r_1_xy=10.0, r_1_wh=5.0, 
                      r_2_xy=0.0, r_2_wh=5.0)
    # Load from MP4. Movement starts at around 27-28s for video 1
    video = loadFrames(DATASET)
    SkipToInteresting(video)

    # It seems to run a bit faster running off of static images
    #  plus you can skip frames to see movement more quickly
    #video = StaticCapture()

    is_frame_good, initial_frame = video.read()

    # Use this for getting bounding box information
    #   DrawBoundingBox(initial_frame)
    
    # Initialize SIFT
    sift = Sift(initial_frame, bounding_box)

    # While testing, don't run on too many frames
    iteration = 0
    while is_frame_good and iteration <= 200:
        # Read a frame
        is_frame_good, frame = video.read()
        if not is_frame_good:
            break
        # To generate static image directory for use with StaticCapture
        #   cv2.imwrite(f'outimgs/out{iteration}.jpg', frame)
        #   iteration += 1
        #   continue

        # Compute Sift predicted
        bounding_box = sift.ProcessImg(frame, bounding_box)

        # Apply Kalman Filter
        bounding_box = kf.update(bounding_box)

        # Convert to int
        bounding_box = bounding_box.astype(np.int64)

        # Draw bounding box
        frame = DrawRectangle(frame, bounding_box)

        # Show image
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        iteration += 1
