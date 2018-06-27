"""Based on Nvidia's Paper on End-to-end learning for self driving car.

   @author
     Victor I. Afolabi
     Artificial Intelligence & Software Engineer.
     Email: javafolabi@gmail.com
     GitHub: https://github.com/victor-iyiola/

   @project
     File: model.py
     Created on 18 May, 2018 @ 5:26 PM.

   @license
     MIT License
     Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""

import tensorflow as tf
from new_data import Dataset


class Model(object):
    def __init__(self, sess: tf.Session, data: Dataset):
        self._sess = sess
        self._data = data

    def __call__(self, X, **kwargs):
        return self.predict(X, **kwargs)

    def train(self):
        pass

    def predict(self):
        pass

    def evaluate(self):
        pass

    def build_graph(self):
        pass

    def build_eval_graph(self):
        pass

    def _predict(self, X, **kwargs):
        pass

    class Result(object):
        def __init__(self):
            pass

        def __len__(self):
            pass

        def __iter__(self):
            pass

        def __getitem__(self, idx):
            pass

    @property
    def sess(self):
        return self._sess

    @property
    def data(self):
        return self._data
