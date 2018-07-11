"""Base class for manimulating and working with dataset.

   @author
     Victor I. Afolabi
     Artificial Intelligence & Software Engineer.
     Email: javafolabi@gmail.com
     GitHub: https://github.com/victor-iyiola

   @project
     File: base.py
     Created on 11 July, 2018 @ 8:01 PM.

   @license
     MIT License
     Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""


class BaseData(object):
    def __init__(self, **kwargs):
        self.train = BaseLoader()
        self.test = BaseLoader()
        self.val = BaseLoader()

    def __repr__(self):
        return NotImplemented

    def __str__(self):
        return self.__repr__()

    def __iter__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def load(self, inputs, **kwargs):
        pass

    def save(self, X, y, **kwargs):
        pass

    def split(self, test_size=0.1, val_size=None):
        pass


class BaseLoader(object):
    """Base class for train, tsest and validation set.

    Methods:
        def next_batch(self, batch_size):
            pass

    Properties:
        features
        labels
    """

    def __init__(self, **kwargs):
        self._features, self._labels = None, None

    def __repr__(self):
        return NotImplemented

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        pass

    def __iter__(self):
        pass

    def __getitem__(self, idx):
        pass

    def next_batch(self, batch_size):
        pass

    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels
