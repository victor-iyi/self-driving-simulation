"""Dataset utility class for loading & pre-processing data for models.

   @author
     Victor I. Afolabi
     Artificial Intelligence & Software Engineer.
     Email: javafolabi@gmail.com
     GitHub: https://github.com/victor-iyiola

   @project
     File: dataset.py
     Created on 11 July, 2018 @ 8:00 PM.

   @license
     MIT License
     Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""


class Dataset(object):
    def __init__(self, **kwargs):
        pass

    def __repr__(self):
        return NotImplemented

    def __str__(self):
        return self.__repr__()

    def predict(self, inputs, **kwargs):
        pass

    def train(self, X, y, **kwargs):
        pass
