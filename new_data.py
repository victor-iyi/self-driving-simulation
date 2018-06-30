"""Helper file for working with training dataset.

   @author
     Victor I. Afolabi
     Artificial Intelligence & Software Engineer.
     Email: javafolabi@gmail.com
     GitHub: https://github.com/victor-iyiola

   @project
     File: new_data.py
     Created on 08 June, 2018 @ 8:27 PM.

   @license
     MIT License
     Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""

import os

import cv2
import numpy as np
import pandas as pd


class Data(object):

    def __init__(self):
        pass

    def __call__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self):
        pass

    def __iter__(self):
        pass


class Dataset(Data):

    # CSV File header names.
    COL_NAMES = [
        'center_path', 'left_path', 'right_path',  # Input
        'steering_angle', 'throttle', 'brake', 'speed'  # Controls.
    ]

    def __init__(self, filename, **kwargs):
        # Extract keyword arguments.
        feature_col = kwargs.get('feature_col') or self.COL_NAMES[0]
        label_col = kwargs.get('label_col') or self.COL_NAMES[-1]
        self._img_size = kwargs.get('img_size') or (200, 200)

        if not os.path.isfile(filename):
            raise FileNotFoundError('{} was not found!'.format(filename))

        # Read csv into a pandas DataFrame object.
        self._df = pd.read_csv(filename, names=self.COL_NAMES)

        # Features & labels.
        img_paths = self._df[feature_col].astype(str).values
        label_paths = self._df[label_col].astype(np.float32).values

        self._images, self._labels = self._create_data(img_paths, label_paths)

    def load_data(self, train: bool=True, size: float=0.3, valid_portion: float=None):

        if 0 >= size < 1:
            raise ValueError('Size must be between 0 & 1.')

        test_size = int(len(self._images) * size)
        train = self._images[:-test_size], self._labels[:-test_size]
        test = self._images[-test_size:], self._labels[-test_size:]

        if valid_portion is not None:
            if 0 >= valid_portion < 1:
                raise ValueError('`valid_portion` must be between 0 & 1.')

            # Calculate the validation size.
            valid_size = int(len(train[0]) * valid_portion)

            # Split validation from training set.
            train = train[0][:-valid_size], train[1][:-valid_size]
            valid = test[0][-valid_size:], test[1][-valid_size:]

            return train, test, valid

        return train, test

    def __repr__(self):
        return '<Dataset {} - {}>'.format(self._images.shape, self._labels.shape)

    def __len__(self):
        return len(self._df)

    def _create_data(self, img_paths, labels):
        assert len(img_paths) == len(labels), 'Lengths don\'t match!'

        # Process images.
        images = [self._process_img(path) for path in img_paths]
        images = np.asarray(images, dtype=np.float32)

        # Process labels.
        labels = np.asarray(labels, dtype=np.float32)

        return images, labels

    def _process_img(self, img_path: str):
        # Load image from path.
        image = cv2.imread(img_path)

        # Resize image.
        image = cv2.resize(image, self._img_size)

        # Convert to RGB colors.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    @property
    def df(self):
        return self._df

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels


if __name__ == '__main__':
    data = Dataset('simulations/driving_log.csv')

    train, test = data.load_data()
    X_train, y_train = train
    X_test, y_test = test
