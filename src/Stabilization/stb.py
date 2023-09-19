"""Module for stabilization of boundry boxes"""

import os
from os.path import isfile, join
from pathlib import Path
import cv2
import numpy as np
from src.stabilisierung.tracker import Tracker
from src.detect_pedestrians.pedestrianrec import generate_pedestrian_boxes
from src.detect_platings.detect_platings import detect_image
from src.ocr.ocr import reset_plates


def pedestrians(frame, tracker):
    """Function for pedestrians.

    This function uses 'generate_pedestrian_boxes' function to detect pedestrians
    on a frame and update their coordinates using a Tracker object and Track
    objects. It also draws bounding boxes around the objects and blur them.

    Args:
        frame: One frame, on which the pedestrians will be detected and
            their bounding boxes will be drawn.
        tracker: Tracker class object, in which the detected pedestrian-
            object-coordinates are kept and tracked.
    Return:
        frame: One frame with bounding boxes of detected and tracked
            objects(pedestrians). The inner areas of bounding boxes are
            blurred.
    """

    # Detect and return coordinates of the boundary boxes in the frame
    # Return of function for a frame:
    # array_of_bboxes_in_a_frame = [[x_up, y_up, x_down, y_down], [],... []]
    list_pedestrians = generate_pedestrian_boxes(frame)

    # If boundary boxes are detected, track them.
    if len(list_pedestrians) > 0:

        # Track objects using Tracker and Kalman filter
        tracker.update(list_pedestrians)

        for track in tracker.tracks:
            track_pos = np.array(track.correction, dtype=int).flatten()
            # Bounding box coordinates:
            x_up, y_up = track_pos[0], track_pos[1]
            x_down, y_down = track_pos[2], track_pos[3]
            # Draw a bounding box rectangle:
            # Blue for pedestrians
            cv2.rectangle(frame, (x_up, y_up), (x_down, y_down), (255, 0, 0), 3)

    return frame


def platings(frame, tracker):
    """Function for license plates.

    This function uses 'detect_image' function to detect license plates on a
    frame and update their coordinates using a Tracker object and Track
    objects. It also draws bounding boxes around the objects and blur them.

    Args:
        frame: One frame, on which license plates will be detected and
            their bounding boxes will be drawn.
        tracker: Tracker class object, in which the detected license plates
            coordinates are kept and tracked.
    Return:
        frame: One frame with bounding boxes of detected and tracked
            objects(license plates). The inner areas of bounding boxes
            are blurred.
    """

    # Detect and return coordinates of the boundary boxes in the frame
    # return of function for a frame:
    # array_of_bboxes_in_a_frame = [[x_up, y_up, width, height], [],... []]
    list_plates = detect_image(frame)
    for pos in list_plates:
        pos[2] = pos[0] + pos[2]
        pos[3] = pos[1] + pos[3]

    # If boundary boxes are detected, track them.
    if len(list_plates) > 0:

        # Track objects using Tracker and Kalman filter
        tracker.update(list_plates)

        for track in tracker.tracks:
            track_pos = np.array(track.correction, dtype=int).flatten()

            # Blur and draw a rectangle:
            # Red for license plates
            frame[track_pos[1]:track_pos[3], track_pos[0]:track_pos[2]] = cv2.blur(
                frame[track_pos[1]:track_pos[3], track_pos[0]:track_pos[2]], (75, 75))

            cv2.rectangle(frame, (track_pos[0], track_pos[1]),
                          (track_pos[2], track_pos[3]), (0, 0, 255), 3)

    # If there is no detection
    else:
        list_plates = []
        for track in tracker.tracks:
            track.kalman.r_matrix = 0
            list_plates.append(np.array(track.correction, dtype=int).flatten())

        tracker.update(list_plates)

        for track in tracker.tracks:
            track_pos = np.array(track.correction, dtype=int).flatten()

            # Blur and draw a rectangle:
            # Red for license plates
            frame[track_pos[1]:track_pos[3], track_pos[0]:track_pos[2]] = cv2.blur(
                frame[track_pos[1]:track_pos[3], track_pos[0]:track_pos[2]], (75, 75))

            cv2.rectangle(frame, (track_pos[0], track_pos[1]),
                          (track_pos[2], track_pos[3]), (0, 0, 255), 3)

    return frame


def execute():
    """Main function."""

    # dist_thresh, max_frames_to_skip
    pedestrians_tracker = Tracker(150, 30)
    platings_tracker = Tracker(150, 30)

    pfad_in = Path(__file__).parent / 'input_frames'
    pfad_out = Path(__file__).parent / 'output_frames'

    files = [f for f in os.listdir(pfad_in) if isfile(join(pfad_in, f))]
    files.sort(key=lambda x: int(x[5:-4]))

    count = 0

    for fil in files:

        image = cv2.imread(str(pfad_in / fil))
        image = platings(image, platings_tracker)
        image_out = pedestrians(image, pedestrians_tracker)
        cv2.imwrite(str(pfad_out / f'frame{count}.jpg'), image_out)
        count += 1
    # Reset detected plates, so program can be run again
    reset_plates()
