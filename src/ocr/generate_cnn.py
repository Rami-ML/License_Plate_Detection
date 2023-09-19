"""Function to create synthetic Numberplate Data for CNN Training"""
import os
from pathlib import Path
import random
from cv2 import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.models import Sequential
from src.ocr.ocr import preprocess_image


def generate_data(path_org, path_des, amount):
    """This function creates synthetic Data in a new directory,
        using the keras ImageDataGenerator feature

    Args:
        path_org (Path object): Path to directory that contains image folders
        each representing a class
        path_des (Path object): Path to location where the synthetic Data is saved to
        amount (int): Amount of pictures that are generated per class

    Notes:
        The only use of this function is to create enough data to train the CNN and wont
        be run after that
    """
    path_des.mkdir(parents=True, exist_ok=True)
    aug = ImageDataGenerator(samplewise_center=True,
                             rotation_range=12, width_shift_range=0.15,
                             height_shift_range=0.15, shear_range=0.45,
                             zoom_range=0.2, channel_shift_range=0.2)

    for foldername, _, filenames in os.walk(path_org):

        # The new synthetic data gets labeled as numbers so it can be used by the CNN
        new_path = path_des / os.path.basename(foldername)
        if os.path.basename(new_path) == os.path.basename(path_org):
            continue
        new_path.mkdir(parents=True, exist_ok=True)
        for filename in filenames:

            total = 0
            # calculates how many picture have to be created for each class
            synthetic_amount = int(amount / len(filenames))
            processed_img = preprocess_image(cv2.imread(foldername + '/' + filename))
            # standardizing images for the CNN
            resized_img = (255 - cv2.resize(processed_img, (28, 28))) / 255
            tensor = resized_img.reshape((1, 28, 28, 1))
            # This creates data until the loop is broken
            for _ in aug.flow(tensor, batch_size=1, save_to_dir=new_path,
                              save_prefix="image", save_format="jpg"):
                total += 1
                if total == synthetic_amount:
                    break


def read_dataset(path):
    """This function reads in an image dataset that is labeled in folders

     Args:
        path (Path object): Path to directory that contains image folders
        each representing a class

    Returns:
        x_train (list): Of Character images
        y_train (list): Of labels for the Character images

    """

    training_data, x_train, y_train = [], [], []
    for folder, _, filenames in os.walk(path):
        for file in filenames:
            processed = preprocess_image(cv2.imread(str(Path(folder, file))))
            processed = cv2.resize(processed, (28, 28))
            training_data.append([processed, int(os.path.basename(folder))])

    random.shuffle(training_data)
    for features, label in training_data:
        x_train.append(features.reshape((28, 28, 1)))
        y_train.append(label)
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    return x_train, y_train


def create_model(name_path, epoch=5):
    """This function creates a Convolutional Neural Network based

    Args:
        name_path (str): String path where model should be saved to
        epoch (int): Amount of Epochs on which the model should be trained

    Note:
    The CNN is highly based on the following:
    https://www.kaggle.com/cdeotte/25-million-images-0-99757-mnist
    This CNN has the best score for the MNIST dataset a set
    that contains handwritten digits. I modified it slightly
    to predict more than 10 classes and trained it on
    the EMNIST dataset which includes handwritten digits and
    letters (in total 47 Classes) . There it scored a performance of
    above 95 %. Thats why i choose this model to try and predict
    my 36 classes.
    """
    path = Path(__file__).parent
    # creating Data for training
    generate_data(path / 'Training_Data', path / 'Synthetic_Data', 3000)
    x_train, y_train = read_dataset(path / 'Synthetic_Data')

    model = Sequential()

    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(128, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=5, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, kernel_size=4, activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(len(y_train), activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=epoch)

    # x_val, y_val = read_dataset(path / 'Training_Data')
    # val_loss, val_acc = model.evaluate(x_val, y_val)
    # print(f'The Validation Loss is {val_loss} and the Validation accuracy is {val_acc}')

    model.save(name_path)
