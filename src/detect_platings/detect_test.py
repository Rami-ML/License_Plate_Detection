'''Testing the Detector module detect_platings'''
from pathlib import Path
import os
import cv2
import numpy as np
from src.detect_platings.detect_platings import read_image, detect_image


img_list = []
pfad = Path(__file__).parent / 'Test Bilder'
folder = os.listdir(pfad)
folder = sorted(folder, key=lambda x: int(os.path.splitext(x)[0]))
for filename in folder:
    img_list.append(cv2.imread(str(pfad / filename)))


def test_read_image():
    '''Tests if mean value is correct.

       This function tests if the mean value of the first 5 images
       is read correct.
    '''

    mean_1 = int(np.mean(img_list[0]))
    mean_2 = int(np.mean(img_list[1]))
    mean_3 = int(np.mean(img_list[2]))
    mean_4 = int(np.mean(img_list[3]))
    mean_5 = int(np.mean(img_list[4]))
    assert read_image(img_list[0])[1] == mean_1
    assert read_image(img_list[1])[1] == mean_2
    assert read_image(img_list[2])[1] == mean_3
    assert read_image(img_list[3])[1] == mean_4
    assert read_image(img_list[4])[1] == mean_5


def test_detect_image_1():
    '''Tests the approximately position of the boundary boxes.

       This function tests if the output position of the boundary box
       from the first image is correct.
       It assumes if the location of the x and y-coordinate of
       the boundary box is approximately in the lower half of
       the picture (start point of [x,y] is in the upper left corner of the
       rectangular boundary box).

       Notes:
            This test is important, because it proves that our
            detector is capable of detecting yellow license plates
    '''

    shape = img_list[0].shape
    approximated_y = shape[0]/2
    real_y = detect_image(img_list[0])[0][1]
    approximated_x = shape[1]/4
    real_x = detect_image(img_list[0])[0][0]
    assert approximated_y < real_y
    assert approximated_x < real_x


def test_detect_image_2():
    '''Tests the approximately position of the boundary boxes.

       This function tests if the output position of the boundary box
       from the second image is correct.
       It assumes if the location of the x and y-coordinate of
       the boundary box is approximately in the lower half of
       the picture.
    '''

    shape = img_list[1].shape
    approximated_y = shape[0]/2
    real_y = detect_image(img_list[1])[0][1]
    approximated_x = shape[1]/4
    real_x = detect_image(img_list[1])[0][0]
    assert approximated_y < real_y
    assert approximated_x < real_x


def test_detect_image_3():
    '''Tests the approximately position of the boundary boxes.

       This function tests if the output position of the boundary box
       from the second image is correct.
       It assumes if the location of the x and y-coordinate of
       the boundary box is approximately in the lower half of
       the picture.

       Notes:
            This test is important, because it proves that our
            detector is capable of detecting images with darker background
    '''

    shape = img_list[2].shape
    approximated_y = shape[0]/2
    real_y = detect_image(img_list[2])[0][1]
    approximated_x = shape[1]/4
    real_x = detect_image(img_list[2])[0][0]
    assert approximated_y < real_y
    assert approximated_x < real_x


def test_detect_image_4():
    '''Tests the approximately position of the boundary boxes.

       This function tests if the output position of the boundary box
       from the second image is correct.
       It assumes if the location of the x and y-coordinate of
       the boundary box is approximately in the lower half of
       the picture.

       Notes:
            Sometimes black cars are captured as "darker" images and get
            wrong positions of boundary boxes, because they are filtered
            with the threshold function from detect_platings.py instead of
            the blur function
    '''

    shape = img_list[3].shape
    approximated_y = shape[0]/2
    real_y = detect_image(img_list[3])[0][1]
    approximated_x = shape[1]/4
    real_x = detect_image(img_list[3])[0][0]
    assert approximated_y < real_y
    assert approximated_x < real_x


def test_detect_image_5():
    '''Tests the approximately position of the boundary boxes.

       This function tests if the output position of the boundary box
       from the second image is correct.
       It assumes if the location of the x and y-coordinate of
       the boundary box is approximately in the lower half of
       the picture.
    '''

    shape = img_list[4].shape
    approximated_y = shape[0]/2
    real_y = detect_image(img_list[4])[0][1]
    approximated_x = shape[1]/4
    real_x = detect_image(img_list[4])[0][0]
    assert approximated_y < real_y
    assert approximated_x < real_x


def test_detect_image_6():
    '''Tests the approximately position of the boundary boxes.

       This function tests if the output position of the boundary box
       from the second image is correct.
       Here it assumes if the boundary box of the license plate is on the left
       lower side of the picture.

       Notes:
            Here the boundary box of the license plate is on the left
            lower side of the picture, so x must have a value between 50-150
            This is a test if traffic images would cause any problems for
            the detector
    '''

    shape = img_list[5].shape
    approximated_y = shape[0]/2
    real_y = detect_image(img_list[5])[0][1]
    approximated_x = shape[1]/10
    real_x = detect_image(img_list[5])[0][0]
    assert approximated_y < real_y
    assert approximated_x < real_x


def test_detect_image_7():
    '''Tests the approximately position of the boundary boxes.

       This function tests if the output position of the boundary box
       from the second image is correct.
       It assumes if the location of the x and y-coordinate of
       the boundary box is approximately in the lower half of
       the picture.
    '''

    shape = img_list[6].shape
    approximated_y = shape[0]/3
    real_y = detect_image(img_list[6])[0][1]
    approximated_x = shape[1]/4
    real_x = detect_image(img_list[6])[0][0]
    assert approximated_y < real_y
    assert approximated_x < real_x


def test_detect_image_8():
    '''Tests the approximately position of the boundary boxes.

       This function tests if the output position of the boundary box
       from the second image is correct.
       It assumes if the location of the x and y-coordinate of
       the boundary box is approximately in the lower half of
       the picture.
    '''

    shape = img_list[7].shape
    approximated_y = shape[0]/3
    real_y = detect_image(img_list[7])[0][1]
    approximated_x = shape[1]/4
    real_x = detect_image(img_list[7])[0][0]
    assert approximated_y < real_y
    assert approximated_x < real_x


def test_detect_image_9():
    '''Tests the approximately position of the boundary boxes.

       This function tests if the output position of the boundary box
       from the second image is correct.
       It assumes if the location of the x and y-coordinate of
       the boundary box is approximately in the lower half of
       the picture.
    '''

    shape = img_list[8].shape
    approximated_y = shape[0]/2
    real_y = detect_image(img_list[8])[0][1]
    approximated_x = shape[1]/4
    real_x = detect_image(img_list[8])[0][0]
    assert approximated_y < real_y
    assert approximated_x < real_x


def test_detect_image_10():
    '''Tests the approximately position of the boundary boxes.

       This function tests if the output position of the boundary box
       from the second image is correct.
       It assumes if the location of the x and y-coordinate of
       the boundary box is approximately in the lower half of
       the picture.
    '''

    shape = img_list[9].shape
    approximated_y = shape[0]/3
    real_y = detect_image(img_list[9])[0][1]
    approximated_x = shape[1]/4
    real_x = detect_image(img_list[9])[0][0]
    assert approximated_y < real_y
    assert approximated_x < real_x


def test_none_plate():
    '''Tests if detector sees car plates.

       This function tests if the detector detects car plates on images
       without cars on it.
    '''

    nothing = []
    assert nothing == list(detect_image(img_list[10]))
    assert nothing == list(detect_image(img_list[11]))


def test_double_plate():
    '''Tests if car plates are recognized.

       This function tests if the detector is capable of recognizing two
       car plates on an image.
    '''

    shape = img_list[12].shape
    approximated_y1 = shape[0]/3
    real_y1 = detect_image(img_list[12])[0][1]
    approximated_x1 = shape[1]/12
    real_x1 = detect_image(img_list[12])[0][0]
    assert approximated_y1 < real_y1
    assert approximated_x1 < real_x1

    approximated_y2 = shape[0]/3
    real_y2 = detect_image(img_list[12])[1][1]
    approximated_x2 = shape[1]/6
    real_x2 = detect_image(img_list[12])[1][0]
    assert approximated_y2 < real_y2
    assert approximated_x2 < real_x2
