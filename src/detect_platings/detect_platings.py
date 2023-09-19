'''Detection Module for detecting platings on cars'''
from pathlib import Path
import cv2
import numpy as np
from src.ocr.ocr import read_numberplate


def read_image(image):
    '''Read the image and convert it.

    This function performs a convertion of the image.
    The assignment of this function is to filter the image.

    Args:
        image : Input image as an array

    Return:
        blurred: Image converted into blur-scale
        mean: calculates the mean value of the image

    Notes:
        Image blurring is achieved by convolving the image
        with a low-pass filter kernel
        It is useful for removing noise (removes high frequency)
    '''

    mean = int(np.mean(image))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return blur, mean


def detect_image(img):
    '''Detect the image and find license plates.

    This function performs a detection to find license plates
    on cars. It creates then boundary boxes around the license plate.

    Args:
        img: Input image as an array

    Returns:
        list: contains the coordinates of the rectangle boundary boxes
            from the detected license plates.

    Notes:
        If the mean value of the image is smaller than 65 (darker tone),
        it goes through a filter called thresh
        Thresh applies an adaptive threshold to the blurred image

        The detectMultiScale function has a paramater called minNeighbors
        minNeighbors: Parameter specifying how many neighbours each candidate
                      rectangle should have to retain it
        If the license plate has a big size,
        then minNeighbors should have values from 5 to 10
        If the license plate has a small size (more realistic in a traffic),
        then the value should be between 2 or 3
    '''
    position = []

    blurred, dark = read_image(img)
    classifier = cv2.CascadeClassifier(
        str(Path(__file__).parent / 'haarcascade.xml'))
    if dark < 65:
        thresh = cv2.adaptiveThreshold(
            blurred, 255, 1, 1, 11, 2)
        platings = classifier.detectMultiScale(
            thresh, minNeighbors=3)
    else:
        platings = classifier.detectMultiScale(
            blurred, minNeighbors=6)

    for i, j, wide, height in platings:
        position.append([i, j, wide, height])
        read_numberplate(img, position)
    return position
