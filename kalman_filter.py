import numpy as np

'''
Example: 

# Initialize
initial_bb = np.matrix([[50.], [60.], [10.], [30.]])
kf = KalmanFilter(initial_bb=initial_bb, dt=1.0, covar=1.0, 
                  proc_noise=4., alpha=0.99, 
                  r_1_xy=2., r_1_wh=5., r_2_xy=25., r_2_wh=50.)

# Loop this
next_bb = np.matrix([[48.], [65.], [14.], [33.]])
filtered_prediction = kf.Update(next_bb)
'''

class KalmanFilter:
    def __init__(self, initial_bb, dt, covar, proc_noise, alpha, r_1_xy, r_1_wh, r_2_xy, r_2_wh):
        '''
        Inputs:
            initial_bb: inital bounding box, a 4x1 matrix [[x], [y], [w], [h]]
            dt:         time delta between frames
            covar:      covariance to use along the diagonal of P
            proc_noise: process noise, added to uncertainty
            alpha:      odds of having a `normal` update (without a 'large' jump in xywh)
            r_1_xy:     measurement noise along the xy dimensions normally
            r_1_xy:     measurement noise along the wh dimensions normally
            r_2_xy:     measurement noise along the xy dimensions in execptionally large cases
            r_2_xy:     measurement noise along the wh dimensions in execptionally large cases
        '''
        # Numpy data type to use for computation
        self.dtype = np.float64

        # Time delta between frames
        self.dt = self.dtype(dt)

        # Odds of a normal (not enormous) error
        self.alpha = self.dtype(alpha)

        # Measurement [[x], [y], [w], [h]]
        self.z_k = initial_bb.astype(self.dtype)

        # Prediction from previous state
        self.A_k = (np.eye(8) + np.diagflat([dt, dt, 0., 0., dt, dt], 2)).astype(self.dtype)

        # State space to observation space conversion
        self.H_k = np.matrix([[1., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 1., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 1., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 1., 0., 0.]], dtype=self.dtype)

        # State [x, y, xdot, ydot, w, h, wdot, hdot]
        self.X_k = self.H_k.T @ self.z_k
        
        # Measurement Noise
        self.R_k_1 = np.diagflat([r_1_xy, r_1_xy, r_1_wh, r_1_wh]).astype(self.dtype)
        self.R_k_2 = np.diagflat([r_2_xy, r_2_xy, r_2_wh, r_2_wh]).astype(self.dtype)

        # Covariance
        self.P_k = covar * np.eye(8, dtype=self.dtype)

        # Process covariance matrix
        pn2 = proc_noise ** 2.0
        self.Q_k = np.diagflat([0., 0., pn2, pn2, 0., 0., pn2, pn2]).astype(self.dtype)


    def update(self, z_k):
        '''
        Inputs:
            z_k: the new measurement [[x], [y], [w], [h]]

        Returns:
            Updated estimate of 4x1 xywh bounding box
        '''
        A_k = self.A_k
        P_k = self.P_k
        Q_k = self.Q_k
        H_k = self.H_k
        alpha = self.alpha 
        beta = 1.0 - self.alpha
        R_k_1 = self.R_k_1
        R_k_2 = self.R_k_2

        # Predict without using measurement
        X_k_bar = self.A_k @ self.X_k
        P_k_bar = (A_k @ P_k @ A_k.T) + Q_k


        # Correction using measurement
        hpht = H_k @ P_k_bar @ H_k.T
        pht = P_k_bar @ H_k.T
        K_1 = pht @ np.linalg.pinv(hpht + R_k_1)
        K_2 = pht @ np.linalg.pinv(hpht + R_k_2)
        K_k = (alpha * K_1) + (beta * K_2)
        self.X_k = X_k_bar + K_k @ (z_k - H_k @ X_k_bar)
        B = (K_1 - K_2) @ (z_k - H_k @ X_k_bar)
        self.P_k = ((np.eye(8) - K_k @ H_k) @ P_k_bar) + (alpha * beta * (B @ B.T))
        #print(f' z_k:        {z_k.T}\n Prediction: {(H_k @ X_k_bar).T}\n Correction: {(H_k @ self.X_k).T}')
        return H_k @ self.X_k
