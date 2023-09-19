"""Since there is no dataset for my CNN im creating my own, this Module helps label data"""
from os import listdir
from pathlib import Path
import time
from cv2 import cv2
import src.ocr.ocr as ocr


PATH_IN = Path(__file__).parent / 'Dataset'
PATH_OUT = Path(__file__).parent / 'Training_Data'


def label_data(path_ori=PATH_IN, path_des=PATH_OUT):
    """This function displays a single character and waits for a Keypress
        to label the character
    Args:
        path_ori (path object): Path to origin folder, that contains images of
        cropped license plate
        path_des (path object): Path to destination directory

    Notes: Im creating a dataset since the accuracy of my character recognition CNN, trained on
        the EMNIST Dataset is pretty low.
        I am taking pictures of license plates than crop them and use my find_characters function
        to cutout the single characters.
    """
    path_des.mkdir(parents=True, exist_ok=True)
    for img_name in listdir(path_ori):
        if not img_name.endswith(('.jpg', '.jpeg', '.png')):
            continue
        print(img_name)
        img = cv2.imread(str(path_ori / img_name))
        found, char, _ = ocr.find_characters(img)
        if not found:
            continue
        cv2.imshow('Original_Image', img)

        for cha in char:

            cv2.imshow('character', cha)
            key = cv2.waitKey(0)
            # changing input keys to labeld folders from 0-36
            if key == 32:
                continue
            # a-z ascii gets changed to 10-36
            if 96 < key < 123:
                key -= 87
            # 0-9 ascii gets changed to 0-9
            elif 47 < key < 58:
                key -= 48
            # 37 class is for the TÜV and state circle on german plates, which
            else:
                key = 36
            # creating a boarder supposedly boosts the accuracy
            image = 255 - cv2.resize(cv2.copyMakeBorder(cha, 7, 7, 7, 7, 0), (28, 28))
            folder_des = path_des / str(key).zfill(2)
            folder_des.mkdir(parents=True, exist_ok=True)
            # getting a unique name so cv2 does not over write the image
            image_name = str(time.time()) + '_.png'
            cv2.imwrite(str(folder_des / image_name), image)

        cv2.destroyAllWindows()
