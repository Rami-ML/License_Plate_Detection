"""Module for Kalman filter"""

import numpy as np


class KalmanFilter():
    """Class KalmanFilter"""

    def __init__(self, detection):
        """This function initializes KalmanFilter object.
        Args:
            detection(list: 1x4): Detection of an object, which is used to
                initialize the state vector.
        """

        # initial state vector
        # x = np.matrix([[x_up], [y_up], [x_down], [y_down], [x_vel], [y_vel]])
        self.state = np.array([[detection[0]], [detection[1]],
                               [detection[2]], [detection[3]], [1], [1]])

        # initial process covariance matrix P
        self.p_matrix = np.eye(6, dtype=int)

        # predicted measurement model matrix H
        self.h_matrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]
        ])

        # observation noise matrix R
        self.r_matrix = np.eye(4)

        # initialization of predicted x, P, z
        self.predicted_state = type(None)
        self.predicted_covariance = np.zeros((6, 6))
        self.predicted_measurement = np.zeros((4, 1))

    def predict(self):
        """Function for prediction.

        This function predicts the current state vector using state vector
        from previous step and A matrix. It also predicts the state covariance
        matrix P.

        Return:
            temp[:4]: First 4 elements of predicted_state vector.
                (Predicted coordinates of bounding box)
        """

        # delta time
        delta_t = 0.005
        # state matrix A
        a_matrix = np.array([
            [1, 0, 0, 0, delta_t, 0],
            [0, 1, 0, 0, 0, delta_t],
            [0, 0, 1, 0, delta_t, 0],
            [0, 0, 0, 1, 0, delta_t],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        # process noise covariance matrix Q
        q_matrix = np.eye(6, dtype=int)    # 6x6

        # prediction of state x: x_{k|k-1} = A*x_{k-1|k-1}
        self.predicted_state = np.dot(a_matrix, self.state)

        # prediction of state covariance matrix P:
        # P_{k|k-1} = A*P_{k-1|k-1}*A_transpose + Q
        self.predicted_covariance = np.dot(a_matrix, np.dot(
            self.p_matrix, a_matrix.T)) + q_matrix

        temp = np.array(self.predicted_state)
        return temp[:4]

    def correct(self, current_measurement):
        """It corrects the state and covariance based on measurements.
        Args:
            current_measurement(matrix: 4x1): It is used to correct the
                state vector.
        Return:
            temp[:4]: First 4 elements of state vector.
                (Corrected coordinates of bounding box)
        """

        c_matrix = np.dot(self.h_matrix, np.dot(
            self.predicted_covariance, self.h_matrix.T)) + self.r_matrix
        # Kalman gain K
        kalman_gain = np.dot(self.predicted_covariance, np.dot(
            self.h_matrix.T, np.linalg.inv(c_matrix)))

        # correction of state x and covariance P
        self.state = np.round(self.predicted_state + np.dot(kalman_gain, (
            current_measurement - np.dot(self.h_matrix, self.predicted_state))))

        self.p_matrix = self.predicted_covariance - (np.dot(
            kalman_gain, np.dot(self.h_matrix, self.predicted_covariance)))

        temp = np.array(self.state).reshape(1, 6).flatten()
        return temp[:4]
