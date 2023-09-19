"""Module for Track and Tracker classes"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from src.stabilisierung.kalman import KalmanFilter


class Track():
    """Track class for every object to be tracked."""

    def __init__(self, det, trackId):
        """Initialize variables for an object to track.

        Args:
            det(list: 1x4): Detection of an object, which is used to initialize
                the Kalman filters state vector. Also initializes the
                self.correction and self.prediction arrays of this object.
            trackId: Id number of tracked object.
        Return:
            None
        """

        self.kalman = KalmanFilter(det)
        self.kalman.predict()
        self.correction = self.kalman.correct(np.array(det).reshape(4, 1))
        self.prediction = np.array(det).reshape(4, 1)  # Predicted boundary boxes
        self.track_id_count = trackId
        self.skipped_frames = 0  # number of frames skipped undetected

    def predict(self):
        """Prediction with Kalman filter"""

        self.prediction = np.array(self.kalman.predict()).reshape(1, 4)

    def correct(self, detection):
        """Correction with Kalman filter
        Args:
            detection(list: 1x4): current measurement to be used in 'correct'
            function of Kalman class. State vector(coordinates of a bbox) is
            corrected with this measurement and predicted state vector.
        Return:
            None
        """

        self.correction = self.kalman.correct(np.matrix(detection).reshape(4, 1))


class Tracker():  # pylint: disable=R0903
    """Tracker class that updates coordinates of objects tracked."""

    def __init__(self, dist_thresh, max_frames_to_skip):
        """Initialize variables.
        Args:
            dist_tresh: distance threshold. When exceeds the threshold,
                        track will be deleted and new track is created.
            max_frames_to_skip: maximum allowed frames to be skipped for
                                the track object undetected.
        Return: None
        """

        self.dist_thresh = dist_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.tracks = []
        self.track_id_count = 0

    def update(self, detections):  # pylint: disable=R0912
        """Update tracks-vector using following steps:
            - Create tracks if no tracks-vector found.
            - Calculate cost using sum of square distance
              between predicted vs detected coordinates.
            - Using Hungarian Algorithm assign the correct
              detected measurements to predicted tracks.
            - Identify tracks with no assignment.
            - If tracks are not detected for long time, remove them.
            - Look for unassigned detects.
            - Start new tracks.
            - Update KalmanFilter state and boundary boxes.
        Args:
            detections: detected positions of boundary boxes to be tracked
                        on a frame
        Return:
            None
        """

        # Create tracks if no tracks-vector found.
        if not self.tracks:
            for detection in detections:
                track = Track(detection, self.track_id_count)
                self.track_id_count += 1
                self.tracks.append(track)

        # Calculate cost; using sum of square distance between
        # coordinates from previous step and from current step
        cost = []
        for track in self.tracks:
            diff = np.linalg.norm(track.prediction.flatten()
                                  - np.array(detections).flatten().reshape(-1, 4), axis=1)
            cost.append(diff)

        cost = np.array(cost)*0.5

        # Using Hungarian Algorithm, assign the
        # coordinates with minimum distance together
        # pre ---assign---> current
        row, col = linear_sum_assignment(cost)

        # list of -1 with length of self.tracks from previous step
        assignment = [-1]*len(self.tracks)
        for i, j in enumerate(row):
            assignment[j] = col[i]

        # Find current tracks with no assignment.
        for i, value in enumerate(assignment):
            if value != -1:
                # Check for cost distance threshold.
                # If cost is very high, then unassign the track.
                if cost[i][value] > self.dist_thresh:
                    value = -1
            else:
                self.tracks[i].skipped_frames += 1

        # If tracks are not detected for long time, remove them.
        for i, track in enumerate(self.tracks):
            if track.skipped_frames > self.max_frames_to_skip:
                try:
                    del track
                    del assignment[i]
                except IndexError:
                    pass

        # Look for unassigned detects.
        # If there is any, start new tracks.
        for i, detection in enumerate(detections):
            if i not in assignment:
                track = Track(detection, self.track_id_count)
                self.track_id_count += 1
                self.tracks.append(track)
                track.predict()

        # Update bboxes of existing objects.
        for i, value in enumerate(assignment):
            if value != -1:
                self.tracks[i].skipped_frames = 0
                self.tracks[i].predict()
                self.tracks[i].correct(detections[value])
