"""Test"""

from copy import deepcopy
import os
from pathlib import Path
import numpy as np
import pytest
from cv2 import cv2
from src.stabilisierung.stb import platings, pedestrians
from src.stabilisierung.tracker import Tracker


img_list_1 = []
img_list_2 = []

pfad = Path(__file__).parent / 'Test Bilder'
files = os.listdir(pfad)
files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))
for image in files:
    if image.endswith('.bmp'):
        img_list_1.append(cv2.imread(str(pfad / image)))
    else:
        img_list_2.append(cv2.imread(str(pfad / image)))

pos1 = [172, 251, 435, 339]
pos2 = [162, 244, 434, 335]


@pytest.mark.parametrize("img", img_list_1)
def test_single_image_pedestrians(img):
    """Test for a single image; pedestrians.

    This test function makes two copies of an image. 'pedestrians'
    function is applied several times on each copy. Both results
    are compared.

    Args:
        img: A pedestrian image from img_list_1.
    """

    img_copy_1 = deepcopy(img)
    img_copy_2 = deepcopy(img)
    tracker_1 = Tracker(160, 5)
    tracker_2 = Tracker(160, 5)
    for _ in range(30):
        img_copy_1 = pedestrians(img_copy_1, tracker_1)
    for _ in range(40):
        img_copy_2 = pedestrians(img_copy_2, tracker_2)
    assert np.all(img_copy_1 == img_copy_2)


@pytest.mark.parametrize("img", img_list_2)
def test_single_image_platings(img):
    """Test for a single image; license plates.

    This test function makes two copies of an image. 'platings'
    function is applied several times on each copy. Both results
    are compared.

    Args:
        img: A license plate image from img_list_2.
    """

    img_copy_1 = deepcopy(img)
    img_copy_2 = deepcopy(img)
    tracker_1 = Tracker(160, 3)
    tracker_2 = Tracker(160, 3)
    for _ in range(10):
        img_copy_1 = platings(img_copy_1, tracker_1)
    for _ in range(30):
        img_copy_2 = platings(img_copy_2, tracker_2)
    assert np.any(img_copy_1 == img_copy_2)


def test_image1_bbox():
    """Test for first image in license plates list.
    (img_list_2)
    The coordinates of bounding box after applying the
    'platings' function 15 times on the image are
    compared with the position list pos1, which is
    known-position of bounding box of the object on
    the frame.
    """

    img = deepcopy(img_list_2[0])
    tracker = Tracker(160, 3)
    for _ in range(15):
        img = platings(img, tracker)
    for track in tracker.tracks:
        track_pos = np.array(track.correction, dtype=int).flatten()
        assert np.all(pos1 == track_pos)


def test_image2_bbox():
    """Test for second image in license plates list.
    (img_list_2)
    The coordinates of bounding box after applying the
    'platings' function 15 times on the image are
    compared with the position list pos2, which is
    known-position of bounding box of the object on
    the frame.
    """

    img = deepcopy(img_list_2[1])
    tracker = Tracker(160, 3)
    for _ in range(15):
        img = platings(img, tracker)
    for track in tracker.tracks:
        track_pos = np.array(track.correction, dtype=int).flatten()
        assert np.all(pos2 == track_pos)
