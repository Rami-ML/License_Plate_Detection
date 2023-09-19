#!/usr/bin/env python3

"""Tests for The OCR Module
The 3 czech license plates, used in the pos_img List and the resize_img
where taken from this Dataset:
https://archive.org/details/HDRDataset
"""

import os
import shutil
from pathlib import Path
from mock import patch
import pytest
from cv2 import cv2
import numpy as np
import tensorflow.keras.models as model
import src.ocr.ocr as ocr
import src.ocr.generate_cnn as gc
from src.ocr.label_data import label_data


pfad = Path(__file__).parent / 'Test_Bilder'
path_ori = pfad / 'test_data'
pos_img = [cv2.imread(str(pfad / fil)) for fil in os.listdir(pfad) if fil.endswith('pos_test.jpg')]
neg_img = [cv2.imread(str(pfad / fil)) for fil in os.listdir(pfad) if fil.endswith('neg_test.jpg')]
resize_img = cv2.imread(str(pfad / 'resize_test.jpg'))

anomal_series =\
    [[[10, 39, 24, 62], [133, 12, 33, 57], [169, 40, 33, 58], [251, 41, 32, 80], [288, 41, 32, 57]],
     [[10, 70, 24, 62], [133, 40, 33, 57], [169, 40, 33, 58], [288, 41, 32, 57], [329, 42, 24, 90]],
     [[10, 39, 24, 30], [133, 40, 33, 57], [169, 65, 33, 58], [251, 41, 32, 56], [329, 41, 26, 56]]]

plate_names = \
    ['5C65578', '9B38368', 'DUP2900', 'MAF6608', 'MAU1288', 'MBB9006', 'MKE1104', 'MKT5268']


@pytest.mark.parametrize("img, position", [(pos_img[0], [[10, 15, 15, 8], [2, 3, 3, 4]])])
def test_cutout(img, position):
    """Test tries to cutout a specified area of an Image
    Args:
        img (numpy array): Input Image from which the cutout should be taken
        position (list of list): list containing x, y, width, height coordinates
    """
    height, width, _ = ocr.cutout(img, position)[0].shape
    assert len(ocr.cutout(img, position)) == len(position)
    assert height == position[0][3]
    assert width == position[0][2]


@pytest.mark.parametrize("img", pos_img[:2])
def test_preprocess_image(img):
    """Tests if the Image Matrix gets reduced to 2d and 2 pixel values (0 and 255)
    Args:
        img (numpy array): Input Image from which the cutout should be taken
    """
    processed = ocr.preprocess_image(img)
    assert len(processed.shape) == 2
    processed[processed == 255] = 0
    assert sum(sum(processed)) == 0


@pytest.mark.parametrize("img", pos_img)
def test_find_characters(img):
    """This tests if the appropriate amount of characters is found
    Args:
        img (numpy array): Input Image from which the cutout should be taken

    Notes:
        The test images are license plates which contain 7 characters
    """
    height, _, _ = img.shape
    _, characters, _ = ocr.find_characters(img)
    for char in characters:
        hei, wid = char.shape
        assert height / 5 < hei
        assert hei > wid
    assert len(characters) == 7


@pytest.mark.parametrize("img", neg_img)
def test_neg_img(img):
    """This tests the find_characters() function with images that dont contain numberplates

    Args:
        img (numpy array): Input Image from which the cutout should be taken
    """
    recognizable, _, _ = ocr.find_characters(img)
    assert recognizable is False


def test_closest_neighbors():
    """Tests if the Index of the closest neighbor is correct
    Notes:
        For the first element the two closest neighbors are index
        1 and 2. For an element in between it should be index - 1
        and index + 1. For the Last element it should be index -1
        and index - 2
    """
    assert [8, 7] == ocr.closest_neighbors(10, 9)
    assert [3, 5] == ocr.closest_neighbors(7, 4)
    assert [1, 2] == ocr.closest_neighbors(3, 0)


@pytest.mark.parametrize("anomal", anomal_series)
def test_filter_anomalies(anomal):
    """Tests if anomalies get filtered out
       Args:
        anomal (List): List of Information about the Character bounding box
            (x postion, y_position, width and height)

    Notes:
        This function is fine tuned it fails for large anomalies as it shifts
        the local mean by a lot and filters out the correct neighbors as well.
        But it performed really well on my tests
        A different approach would be, instead of taking the local mean, to take the global mean.
        This on the other hand performs worse since sometime only the upper part of a character is
        found and gets filtered out by the global mean
    """
    no_anomal = [[100, 39, 24, 62], [133, 40, 33, 57], [169, 40, 33, 58]]
    characters_dummy = ['Bild1', 'Bild2', 'Bild3', 'Bild4', 'Bild5']
    ocr.filter_anomalies(anomal, 1, 15, characters_dummy)
    assert len(characters_dummy) == 4
    ocr.filter_anomalies(anomal, 3, 20, characters_dummy)
    assert len(characters_dummy) == 3
    ocr.filter_anomalies(no_anomal, 1, 15, characters_dummy)
    assert len(characters_dummy) == 3
    ocr.filter_anomalies(no_anomal, 1, 15, characters_dummy)
    assert len(characters_dummy) == 3


def test_filter_blue():
    """Test if Blue Images get filtered

    Notes:
        CV2 reads arrays in BGR thus the first layer
        is the blue content of an image
    """
    blue_image_dummy = np.zeros([5, 5, 3])
    blue_image_dummy[:, :, 0] = np.ones([5, 5]) * 180
    blue_image_dummy[:, :, 1] = np.ones([5, 5]) * 100
    blue_image_dummy[:, :, 2] = np.ones([5, 5]) * 120
    assert ocr.filter_blue(blue_image_dummy)
    blue_image_dummy[:, :, 0] = np.ones([5, 5]) * 120
    assert ocr.filter_blue(blue_image_dummy) == 0


@pytest.mark.parametrize("img", pos_img + [resize_img])
def test_filter_small_boxes_and_resize_boxes(img):
    """Test if all small boxes where filtered out and resized properly
     Args:
        img (numpy array): Input Image from which the cutout should be taken
    """
    _, characters, _ = ocr.find_characters(img)
    size = np.ones([len(characters), 2])
    for i, character in enumerate(characters):
        size[i] = character.shape
    average_height = int(size[:, 0].mean()) * 0.9
    for height in size[:, 0]:
        assert average_height < height


def test_generate_data():
    """Test if synthetic Data is created in a new directory

    Notes:
        The test_data directory consists of 4 class folders, each
        containing a single Image. Hence after running the test,
        a new directory with 4 folders and a set amount of images
        in them should exist.
        The created directory is deleted after the test is done,
        since every time the test is run new images would be added
    """

    path_des = pfad / 'test_synthetic'
    # Generating 10 synthetic images per category
    gc.generate_data(path_ori, path_des, 10)
    # checking if category folder with name 00 was created
    assert os.path.isdir(Path(path_des / '00'))
    # checking if four folders where created
    assert len(list(os.listdir(path_des))) == 4
    # checking if folder contains 10 files
    assert len(list(os.listdir(Path(path_des / '00')))) == 10
    assert len(list(os.listdir(Path(path_des / '01')))) == 10
    shutil.rmtree(path_des)


def test_read_data():
    """Test if it reads in images so that they can be used for model fitting

    Notes:
        The CNN requires a tensor and class labels as numbers. The test_data
        folder contains 4 folders 00, 01, 10, 11 corresponding to the classes
        0, 1, A, B. Folder 00 contains 2 images while the others only contain
        1.
    """
    x_data, y_data = gc.read_dataset(path_ori)
    # five images with the following labels 0 + 0 + 1 + 10 + 11 = 22
    assert sum(y_data) == 22
    # test if all five images are one tensor
    assert x_data.shape == (5, 28, 28, 1)


@pytest.mark.parametrize("img", pos_img + [neg_img[0]])
def test_recognize_characters(img):
    """Test if characters get recognized
    Args:
        img (numpy array): Input Image
    """
    found, characters, _ = ocr.find_characters(img)
    plate = ocr.recognize_characters(found, characters)
    if found:
        assert len(plate) == 7
        assert isinstance(plate, str)
        assert plate in plate_names
    else:
        # test for non character image
        assert plate == 'Could not be detected'


@pytest.mark.parametrize("img", [pos_img[0], neg_img[0]])
def test_read_numberplates(img):
    """Test if characters get recognized
    Args:
        img (numpy array): Input Image
    """
    plate_path = pfad / 'Plates'
    dummy_box = [[0, 0, img.shape[1] - 1, img.shape[0] - 1]]

    with patch('src.ocr.ocr.find_characters', return_value=['dummy1', 'dummy', img]):
        with patch('src.ocr.ocr.recognize_characters', return_value='Test'):
            ocr.read_numberplate(img, dummy_box, path=plate_path)
            assert os.path.isfile(plate_path / 'Test.jpg') is False

            with patch('src.ocr.ocr.filter_confidence', return_value=True):
                ocr.read_numberplate(img, dummy_box, path=plate_path)

    assert os.path.isdir(plate_path)
    assert os.path.isfile(plate_path / 'Test.jpg')
    os.remove(plate_path / 'Test.jpg')


def test_label_data():
    """Tests the labeling Function, by simulating key presses

    Notes:
        The cv2.waitKex() function returns the ASCii value of the pressed key,
        label_data converts this to a number between 0 and 36 and creates a folder,
        with that number. If the space bar (ASCii 32) is pressed it skips labeling.
        The keys that get simulated are 0, D, U, space bar, #, 9, 0
    """
    path_train = pfad / 'test_train_data'

    with patch('src.ocr.label_data.cv2.waitKey', side_effect=[48, 117, 32, 35, 57, 48, 100]):
        label_data(pfad / 'label_test', path_train)

    folders = [int(fol) for fol in os.listdir(path_train)]
    # 32 means skip and 48 is pressed twice that's why only 5 Folders should remain
    assert len(folders) == 5
    # 0 + 13 + 30 + 36 + 9 + 0 = 88 (corresponding to folder created by the keys pressed)
    assert sum(folders) == 88
    zeros = list(os.listdir(path_train / '00'))
    # zero was pressed twice hence '00' should contain two files
    assert len(zeros) == 2
    # deleting created folder
    shutil.rmtree(pfad / 'test_train_data')


def test_create_model():
    """Test if a CNN is created

    Notes:
        This function creates a CNN only using a small amount of data. It can not be
        evaluated properly.Since fitting the model accurately, would require a few minutes
        with sufficient amount of RAM.
        """
    cnn_path = Path(__file__).parent / 'test_cnn'
    # Mocks the generate_data function so it doesn't create as much data
    with patch('src.ocr.generate_cnn.generate_data',
               return_value=gc.generate_data(path_ori, pfad / 'Synthetic_Data', 50)):
        # Mocks the read_data function, to change its path to the test folder
        with patch('src.ocr.generate_cnn.read_dataset',
                   return_value=gc.read_dataset(pfad / 'Synthetic_Data')):
            gc.create_model(str(cnn_path), epoch=1)

    # asserts if Model exists
    assert os.path.isdir(cnn_path)
    # try to load model
    try:
        model.load_model(cnn_path)
    except IOError:
        raise Exception('CNN could not be loaded!')
    # remove created data
    shutil.rmtree(cnn_path)
    shutil.rmtree(pfad / 'Synthetic_Data')


@patch('src.ocr.ocr.detected_plates', plate_names)
@pytest.mark.parametrize('plate_text, truth', [[plate_names[0], True], [plate_names[0], False],
                                               ['dummy', False], ['dummy', False], ['M12', False]])
def test_filter_confidence(plate_text, truth):
    """Test the filter_confidence Function

    Args:
        plate_text (str): string of detected plate
        truth (bool): Bool of assert should be true or false

    Notes:
        Im checking the Function with different plates. The confidence level, is
        set to 2 meaning a plate has to be detected exactly twice before it passes
        the filter. It also filters number plates that are not likely to be a numberplate, hence
        the test with dummy

    """
    with patch('src.ocr.ocr.CONFIDENCE_LVL', 2):
        assert ocr.filter_confidence(plate_text) == truth


def test_reset_plates():
    """Tests the reset plates Function"""
    with patch('src.ocr.ocr.detected_plates', plate_names):
        assert len(plate_names) != 0
        ocr.reset_plates()
        assert len(plate_names) == 0
