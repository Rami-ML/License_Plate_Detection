#!/usr/bin/env python3

"""OCR Module for detecting Characters on License plates"""
import bisect
from collections import Counter
from pathlib import Path
import re
from cv2 import cv2
import numpy as np
import tensorflow.keras.models as tf

# declaring variables
detected_plates = []
german_np = re.compile(r'^[A-Z]{1,5}\d{1,4}[EH]?$')
model = tf.load_model(Path(__file__).parent / 'cnn.model')
# Number of times the same string has to be detected, till the plate gets saved as detected
CONFIDENCE_LVL = 3


def cutout(img, position):
    """Function for cutting out multiple Boundary boxes in an Image
    Args:
        img (numpy 3d array): Input Image from which the cutout should be taken
        position (list of lists): list containing x, y, width, height coordinates

    Returns:
        img (3d numpy array): cut out of images of given position
    """
    return [img[pos[1]: pos[1] + pos[3], pos[0]: pos[0] + pos[2]] for pos in position]


def preprocess_image(img):
    """Function to Preprocess an Image
    Args:
        img (3d numpy array): Input Image

    Returns:
        thresh (2d numpy array): black and white, smoothed input images

    Notes:
        Image first getsturned into grayscale, then it is smoothed and finally thresholded.
        The Adapitvethreshold settings where tested on 70 licens plates white 7 characters this
        setting only let the find_characters function to missread


    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    return thresh


def find_characters(img):
    """Function that singles out characters in a given image of a license plate
        and returns a list of the character images
    Args:
        img (3d numpy array): Input Image

    Returns:
        Bool: If there is a sufficient amount of characters detected
        characters (list): list of single characters images detected on the input image
        img (3d numpy array): input image

    Notes:
        The function uses Multiple filters to find the characters, it assumes
        that bounding boxes must have a similar height and y coordinate. If
        less than 3 character boxes for fill this requirement, than the Image cant
        be recognized correctly anymore, since european plates have at least 3 characters.
        The filters also gets rid of most images, who do not show a license plate. Most
        of the assumptions in this function where fine tuned during the labeling of
        the dataset.
    """
    characters, location = [], []
    # org_img = np.copy(img)
    processed_img = preprocess_image(img)
    height, width = processed_img.shape

    contours, hierarchy = cv2.findContours(processed_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for i, cnt in enumerate(contours):

        # filter really small boxes and boxes inside of characters, e.g. the hole in the number 6
        if cv2.contourArea(cnt) > 25 and not hierarchy[0][i][3] > -1:
            [x_p, y_p, wid, hei] = cv2.boundingRect(cnt)
            # assumption box can only be rectangular and some size asumptions
            if hei > wid > width / 32 and hei > height / 5 and \
                    not filter_blue(img[y_p: y_p + hei, x_p: x_p + wid]):
                # cv2.rectangle(org_img, (x_p, y_p), (x_p + wid, y_p + hei), (0, 255, 0), 1)
                index = bisect.bisect([i[0] for i in location], x_p)
                location.insert(index, [x_p, y_p, wid, hei])
                characters.insert(index, processed_img[y_p: y_p + hei, x_p: x_p + wid])

    # a planed error occurs when less than 3 characters are left in the list
    try:
        filter_small_boxes(location, characters)
        filter_outer_left_right(location, characters)
        # assumption that the y coordinate can only change a 5th of the total Image height
        filter_anomalies(location, 1, processed_img.shape[0] / 5, characters)
        resize_boxes(location, processed_img, characters)
        assert len(characters) > 2
    except (IndexError, ValueError, AssertionError):
        return False, 'Not enough Characters detected', img

    return True, characters, img


def closest_neighbors(length, index):
    """ Function that returns the 2 closet neighbors of a sorted list
    Args:
        length (int): length of array
        index (int): index of current position

    Returns:
        List: the indexes of the two closest neighbors

    Notes:
        For the first element the two closest neighbors are index
        1 and 2. For an element in between it should be index - 1
        and index + 1. For the Last element it should be index -1
        and index - 2. Here the assumption is made that european
        license plate have more or equal to 3 characters, thus
        smaller lists dont have to be handled because, the plate
        cannot be detected correctly
    """
    if index == 0:
        return [1, 2]
    if index == length - 1:
        return [length - 2, length - 3]
    return [index - 1, index + 1]


def filter_anomalies(location, pos, threshold, characters):
    """ Function that filters out anomalies based on the two closed neighbors
    Args:
        location (list with len 4): x position, y position, width, height
        pos (int from [0:3]): index which anomalie of location should be targeted
        threshold (int): how far from the mean is acceptable
        characters (list of numpy arrays): detected characters

    Notes:
        This function calculates the local mean and decides if the box is
        feasible.
    """
    locs = np.array(location)[:, pos]
    del_val = 0
    for i, loc in enumerate(locs):
        neighbors = closest_neighbors(locs.shape[0], i)
        loc_mean = (locs[neighbors[0]] + locs[neighbors[1]]) / 2
        if loc > loc_mean + threshold or loc < loc_mean - threshold:
            del characters[i - del_val], location[i - del_val]
            del_val += 1


def filter_outer_left_right(location, characters):
    """Filters out the outer anaomal characters

    Args:
        location (list with len 4): x position, y position, width, height
        characters (list of numpy arrays): detected characters


    Notes: The outer most characters are often not a character of the license plate
        targeting them with a stricter filter improved the algorithm over all
    """
    height = np.array(location)[:, 3]
    if height[0] > height[1:3].mean() + 8 or height[0] < height[1:3].mean() - 8:
        del characters[0], location[0]
    if height[-1] > height[-3:-1].mean() + 8 or height[-1] < height[-3:-1].mean() - 8:
        del characters[-1], location[-1]


def filter_blue(img):
    """Filters bounding boxes that contain a lot of blue
    Args:
        img (3d numpy array): img cutout

    Returns:
        bool: If the given box contains to much blue

    Notes: European Numberplates have a blue box on the left of the plate.
        The bounding box has similar features as the character box. Making it unfeasible
        for that kind of filtering
    """
    blue, green, red = cv2.split(img)
    blue, green, red = blue.mean(), green.mean(), red.mean()
    return blue > 30 + green and blue > red + 30


def resize_boxes(location, processed_img, characters):
    """ Function that resize boxes to the average
    Args:
        location (list with len 4): x position, y position, width, height
        processed_img (2d numpy array): img after preprocess_image(img)
        characters (list of numpy arrays): detected characters

    Notes:
        Sometimes only the upper or lower part of a characters gets detected,
        this makes sure that the box has the same size as its neighbors. It only
        runs on a really small margin since filter_small_boxes deletes boxes
        that are 80% smaller than the average.

    """
    x_p, y_p, width, height = np.hsplit(np.array(location), 4)
    height_mean = int(height.mean())
    for i, char_height in enumerate(height):
        if char_height < height_mean * 0.9:
            neighbors = closest_neighbors(height.shape[0], i)
            y_mean = int((y_p[neighbors[0]] + y_p[neighbors[1]]) / 2)
            characters[i] = \
                processed_img[y_mean: y_mean + height_mean, x_p[i][0]: x_p[i][0] + width[i][0]]


def recognize_characters(char_found, chars):
    """Function that recognizes detected Characters

    Args:
        char_found (Bool): True or False if characters where been found
        chars (list of numpy arrays): detected characters

    Returns:
        string: Of detected Plate or 'Could not be detected'
    """

    if char_found:
        characters = \
            [cv2.resize(cv2.copyMakeBorder(char, 7, 7, 7, 7, 0), (28, 28)) for char in chars]

        # 0-9, A-Z and the 37 class for german TÜV and state sign
        map_legend = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B',
                               'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
                               'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                               ''])
        # reshaping character images into tensor
        tensor = np.asarray(characters).reshape((len(characters), 28, 28, 1))
        predictions = model.predict(tensor)
        index = [np.argmax(pre) for pre in predictions]
        return ''.join(map_legend[index])

    return 'Could not be detected'


def filter_small_boxes(location, characters):
    """Filters out boxes that are smaller than the average

       Args:
           location (list with len 4): x position, y position, width, height
           characters (list of numpy arrays): detected characters

       Notes: The assumption that boxes shouldn't be smaller than 20% bellow the mean
            worked fairly well during labeling 300+ Images, where the find_characters
            function was called to cut out the single characters. It also made the
            resize boxes almost obsolete
       """
    height = np.array(location)[:, 3]
    del_id = np.where(height < height.mean() * 0.8)
    for i in reversed(del_id[0]):
        del characters[i], location[i]


def read_numberplate(img, boxes, path=Path(__file__).parent / 'Licenseplates'):
    """Bringing everything together, from input frame to a saved plate image with detected
        string name

     Args:
        img (numpy 3d array): whole Frame as input Image
        boxes (list of list): list containing bounding boxes for license plates on given frame
        path (path object): path to where the license plate should be saved to
    """
    path.mkdir(parents=True, exist_ok=True)
    plates = cutout(img, boxes)
    for plate in plates:
        found, characters, schild = find_characters(plate)
        plate_text = recognize_characters(found, characters)
        if filter_confidence(plate_text) is False:
            continue
        cv2.imwrite(str(path / plate_text) + '.jpg', schild)


def filter_confidence(plate_name):
    """Filter out strings that are unlikely to be a numberplate

       Args:
           plate_name (str): string of detected Plate Text

        Returns:
            bool: if detected plate string is likely to be a real license plate

       Notes:
           This filter is specialized to detect german number plates, since
           there is no standardized european syntax. The len(plate_name) > 6
           is an addition to include a high number of european plates, according
           to https://en.wikipedia.org/wiki/Vehicle_registration_plates_of_Europe.
           Unfortunately this does increase wrong predictions, as a trade off.
           The CONFIDENCE_LVL determent's how often the same plate string
           has to be detected before it is confident that its correct
       """

    if german_np.match(plate_name) or 6 < len(plate_name) < 9:
        detected_plates.append(plate_name)
    plate_count = Counter(detected_plates)
    return plate_count[plate_name] == CONFIDENCE_LVL


def reset_plates():
    """Resets detected_plates list

    Notes: This reset is necessary, so the program doesn't have
        to be closed after running once
        """
    print('Process finished')
    detected_plates.clear()
