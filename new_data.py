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

        print(self._images.shape, self._labels.shape)

    def load_data(self, train=True, size=0.2, **kwargs):
        normalize = kwargs.get('normalize', False)

    def __repr__(self):
        return '<Dataset {} - {}>'.format(self._images.shape, self._labels.shape)

    def __len__(self):
        return len(self._df)

    @classmethod
    def fromArray(cls, arr):
        pass

    def _create_data(self, img_paths, labels):
        assert len(img_paths) == len(labels), 'Lengths don\'t match!'

        # Process images.
        images = [self._process_img(path) for path in imag_paths]
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
