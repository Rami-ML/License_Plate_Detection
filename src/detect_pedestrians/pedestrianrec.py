#!/usr/bin/env python3

""" Module for extraction of boundary boxes for pedestrians from an image"""

from collections import namedtuple
import cv2


def overlap_between(test_box, bigger_box):
    """
    Calculates ratio between overlapping area and area covered by the test_box
    Args:
        test_box, bigger_box: boundary_box namedtuples, containing pixel coordinates
        of top left and bottom right corners.
    Notes:
        The function does not ensure that input 'test_box' is actually the smaller,
        redundant box that may therefore be absorbed by a larger one; if inputs are
        entered in the wrong order then the outputted ratio will not be meaningful as
        intended, and non_maximum_suppression will fail to achieve the desired results
    """

    x_left = max(test_box.x_tl, bigger_box.x_tl)
    x_right = min(test_box.x_br, bigger_box.x_br)
    y_top = max(test_box.y_tl, bigger_box.y_tl)
    y_bottom = min(test_box.y_br, bigger_box.y_br)

    x_overlap = max(0, x_right - x_left)
    y_overlap = max(0, y_bottom - y_top)
    overlap_area = x_overlap * y_overlap

    test_box_area = (test_box.x_br - test_box.x_tl) * (test_box.y_br - test_box.y_tl)

    overlap_ratio = overlap_area / test_box_area

    return overlap_ratio


def non_maximum_suppression(boxes):
    """
    Returns new list without redundant boxes.
    Args:
        boxes: List of boundary_box namedtuples, containing pixel coordinates of
        top left and bottom right corners
    Notes:
        In order for a box to be discarded, it must, at the presently chosen threshold,
        be at least 50% overlapping with another; a threshold that is too high will
        cause some redundant boxes to not be removed; one that is too low will cause
        boxes for pedestrians close to each other to be removed.
    """

    boxes = sorted(boxes, key=lambda box: box[2]-box[0], reverse=True)
    nms_boxes = []
    overlap_threshold = 0.5

    for box in boxes:
        if not any([overlap_between(box, nms_box) > overlap_threshold for nms_box in nms_boxes]):
            nms_boxes.append(box)

    return nms_boxes


def generate_pedestrian_boxes(image):
    """
    Detects pedestrians and returns list of boundary_boxes for them. Applies
    non_maximum_suppression to decrease redundancy/false detection.
    Args:
        image: image file loaded with cv2.imread()
    Notes:
        If the image is wider than 400 pixels, it is resized to reduce run-time.
        Manipulating winStride and scale affects run time and detection accuracy.
    """

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    min_width = 400
    new_width = min(min_width, image.shape[1])
    scale_factor = min_width/image.shape[1]
    new_height = round(image.shape[0] * scale_factor)

    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    (boxes, _) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8),
                                      scale=1.05)

    if len(boxes) > 0:
        for row in boxes:
            row[2] = row[0] + row[2]
            row[3] = row[1] + row[3]

        Box = namedtuple('Box', 'x_tl y_tl x_br y_br')
        boxes = [Box._make(row) for row in boxes]

        boxes = non_maximum_suppression(boxes)

        boxes = [[int(round(element/scale_factor)) for element in row] for row in boxes]

    return boxes


def show_boxes(img, boundary_boxes, gt_boxes=None):
    """
    Shows the generated and ground-truth boundary boxes in the image.
    This function is not presently called anywhere, but it had its use.
    """

    for (x_tl, y_tl, x_br, y_br) in boundary_boxes:
        cv2.rectangle(img, (x_tl, y_tl),
                      (x_br, y_br),
                      (0, 0, 255), 2)

    if gt_boxes is not None:
        for (x_tl, y_tl, x_br, y_br) in gt_boxes:
            cv2.rectangle(img, (x_tl, y_tl),
                          (x_br, y_br),
                          (0, 255, 0), 2)

    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def print_overlaps(gt_list, det_list):
    """
    Prints overlaps between each ground truth box and all of the detected boxes,
    as well as a list of the highest overlaps calculated for each ground truth box.
    Once again, not called anywhere presently but was useful.
    """

    overlap_list = []
    high = 0
    for i_1, grt in enumerate(gt_list):
        for i_2, det in enumerate(det_list):
            overlap = overlap_between(grt, det)
            print(i_1, i_2, overlap)
            if overlap > high:
                high = overlap
        overlap_list.append(high)
        high = 0

    print(overlap_list)
