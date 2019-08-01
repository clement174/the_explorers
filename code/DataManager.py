# Data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from os.path import join, isdir
import cv2
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# Variables

# Ne PAS mettre le path vers les données dans le repo local.
# Faire une copie des fichiers ailleurs et mettre ce path là.
# Plus prudent :)
data_path = "C:/Users/simplon/Documents/Code/Simplon/Python/machine_learning/The_explorers/data/categs_clean"



# CLASS

class DataManager():

    def __init__(self, path):

        self._path = path
        self._label_to_int = {}
        self._int_to_label = {}


    def get_data(self, img_size=None, split=False, model_class=None):
        """
            Main function used for data importation.
            - If img_size, resize all image to a square of shape (img_size, img_size)
            - If split is True, return X_train, X_test, y_train, y_test
                else return X, y
            - If transfer learning, put model class (which has the .preprocess() method) in transfer_model
        """
        X, y = self.import_data(img_size)
        X = self.preprocess_data(X, model_class)

        if split:
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=7)
            return X_train, X_test, y_train, y_test

        else:
            return X, y


    def get_label_name(label_id):
        """
            Retrieve the category name of an image based on the label used by model
        """
        if self._int_to_label == {}:
            print("ERROR")
            print("Need to import data first")
        else:
            label_name = self._int_to_label[label_id]

            return label_name


    def get_label_id(label_name):
        """
            Retrieve the label of image used by model from his category name
        """
        if label_to_int == {}:
            print("ERROR")
            print("Need to import data first")
        else:
            label_id = self._label_to_int[label_name]

            return label_id

    #----------------------------------------------------------------------------------------------

    # PRIVATE

    def import_data(self, img_size):
        """
            Import images from dir
            (Used by get_data())
        """
        path = self._path
        images = []
        labels = []

        categs_name = [filename for filename in os.listdir(path)]
        for categ in categs_name:
            if isdir(join(path, categ)):

                for img_name in os.listdir(join(path, categ)):

                    if ".jpg" in img_name:

                        img_name = self.correct_filename(img_name, categ)
                        img_path = join(path, categ, img_name)
                        img = cv2.imread(img_path)

                        if img_size:
                            dim = (img_size, img_size)
                            try:
                                img = cv2.resize(img, dim)
                            except:
                                print(img_name, "has not been loaded.")
                                continue

                        images.append(img)
                        labels.append(categ)

        X = np.array(images)
        y = self.transform_labels(labels)

        return X, y


    def correct_filename(self, img_name, categ):
        """
            Check if wrong character in image name
            If so, change filename in dir and return correct filename for cv2
        """
        path = self._path

        # Change wrong characters in filename
        wrong_char = [char for char in img_name if char in [" ", "(", ")", "é", "©"]]
        if len(wrong_char) > 0:

            new_img_name = img_name
            for char in [" ", "(", ")", "©"]:
                new_img_name = new_img_name.replace(char, "")
            new_img_name = new_img_name.replace("é", "e")

            os.rename(join(path, categ, img_name), join(path, categ, new_img_name))
            img_name = new_img_name

        return img_name


    def transform_labels(self, categs):
        """
            One hot encode categs (Used by get_data())
            Store label_to_int and int_to_label dictionary
                used by methods get_label_name() and get_label_id
        """
        label_encoder = LabelEncoder()
        enc_label = label_encoder.fit_transform(categs)

        self._label_to_int = { categ: label_encoder.transform([categ])[0] for categ in categs }
        self._int_to_label = { label_encoder.transform([categ])[0]: categ for categ in categs }

        y = np_utils.to_categorical(enc_label)

        return y


    def preprocess_data(self, X, model_class):
        """
            Preprocess data. If transfer learning, need class of model to call .preprocess_input()
            If not, normalize
        """
        if model_class:

            X = model_class.preprocess_input(X)
        else:
            X = X / 255

        return X


    def split_data(self):
        """
            Split data and make sure we all have the same random_state
            Used by get_data()
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=7)

        return X_train, X_test, y_train, y_test





# DEMO IMPORT

# Select image size
img_size = 224

# Instantiate manager
manager = DataManager(data_path)

# Load data
from keras.applications import xception
X_train, X_test, y_train, y_test = manager.get_data(img_size=img_size, split=True, model_class=xception)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)