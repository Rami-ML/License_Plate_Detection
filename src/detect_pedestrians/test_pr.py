#!/usr/bin/env python3

"""Tests for the pedestrian recognizer"""

from collections import namedtuple
import os
import csv
from pathlib import Path
import pytest
import cv2
from src.detect_pedestrians.pedestrianrec import (overlap_between, non_maximum_suppression,
                                                  generate_pedestrian_boxes)

Box = namedtuple('Box', 'x_tl y_tl x_br y_br')
test_boxes = [Box(0, 0, 40, 40), Box(36, 0, 41, 40), Box(39, 0, 44, 40),
              Box(50, 0, 60, 10), Box(10, 10, 20, 20)]

path = Path(__file__).parent/'test_imgs'

img_list = [cv2.imread(str(path/file))
            for file in sorted(os.listdir(path)) if file.endswith('.bmp')]
detected_boxes = [generate_pedestrian_boxes(img) for img in img_list]
detected_boxes = [[Box._make(box) for box in row] for row in detected_boxes]

ground_truth_boxes_str = [list(csv.reader((open(str(path/file), newline='')), delimiter=';'))
                          for file in sorted(os.listdir(path)) if file.endswith('.csv')]
ground_truth_boxes_int = [[[int(element) for element in box] for box in row]
                          for row in ground_truth_boxes_str]
ground_truth_boxes = [[Box._make(box) for box in row] for row in ground_truth_boxes_int]


OverlapTest = namedtuple('OverlapTest', 'big_box test_box overlap')
high_overlap = OverlapTest(test_boxes[0], test_boxes[1], 0.8)
low_overlap = OverlapTest(test_boxes[0], test_boxes[2], 0.2)
no_overlap = OverlapTest(test_boxes[0], test_boxes[3], 0)


@pytest.mark.parametrize("boxes", [high_overlap, low_overlap, no_overlap])
def test_overlap_calculation(boxes):
    """
    Tests overlap calculation.
    Args:
        boxes: list (len=3) of OverlapTest namedtuples, which contain two Box namedtuples
        and the overlap between the two (float)
    """

    overlap = overlap_between(boxes.test_box, boxes.big_box)

    assert overlap == boxes.overlap


NMSTest = namedtuple('NMSTest', 'box_list after_nms')
one_absorbable_box = NMSTest([test_boxes[0], test_boxes[1]], [test_boxes[0]])
no_absorbable_box = NMSTest([test_boxes[0], test_boxes[2]], [test_boxes[0], test_boxes[2]])
two_absorbable_boxes = NMSTest([test_boxes[0], test_boxes[1], test_boxes[4]], [test_boxes[0]])


@pytest.mark.parametrize("boxes", [one_absorbable_box, no_absorbable_box,
                                   two_absorbable_boxes])
def test_nms(boxes):
    """
    Test non_maximum_suppression.
    Args:
        boxes: list (len=3) of NMSTest namedtuples, which contain two lists of Box namedtuples;
        one before the algorithm is applied, and the desired output.
    """

    boxes_after_nms = non_maximum_suppression(boxes.box_list)

    assert sorted(boxes_after_nms) == sorted(boxes.after_nms)


parameter_array = []
DetectionTest = namedtuple('DetectionTest', 'gt_box detected_boxes')
for index, row in enumerate(ground_truth_boxes):
    for gt_box in row:
        parameter_array.append(DetectionTest(gt_box, detected_boxes[index]))


@pytest.mark.parametrize("gt_detection_combo", parameter_array)
def test_boundary_boxes(gt_detection_combo):
    """
    Tests if every single ground truth box has a corresponding detected box by assessing
    overlap between each box and all detected boxes in the image.

    The images used to test detection success contain varying amounts of pedestrians
    in a variety of conditions, including an image that is blurry, a dark one, one
    of people sitting, one image with 3 people walking near one another.
    The images were found online.

    The ground truths are csv files that contain pixel-coordinates of tightly-fitted
    boundary boxes around expected detections. To determine these pixel coordinates
    I drew boxes in Preview (macOS image editor).

    Args:
        gt_detection_combo: list of DetectionTest namedtuples, which contain one
        ground truth Box namedtuple and a list of all the detected Boxes in the image
        to which the ground truth box belongs.
    """
    found = False
    overlap_threshold = 0.7

    for found_box in gt_detection_combo.detected_boxes:
        if overlap_between(gt_detection_combo.gt_box, found_box) > overlap_threshold:
            found = True
            break

    assert found is True


def test_empty_img():
    """
    Tests that an image of a football produces no detections. Because of naming,
    the image and the relevant ground truth csv file are at the end of the list.
    """
    assert detected_boxes[-1] == ground_truth_boxes[-1]
