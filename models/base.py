"""Base class for Machine Learning Models to be used for this project.

   @author
     Victor I. Afolabi
     Artificial Intelligence & Software Engineer.
     Email: javafolabi@gmail.com
     GitHub: https://github.com/victor-iyiola

   @project
     File: base.py
     Created on 11 July, 2018 @ 8:00 PM.

   @license
     MIT License
     Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""


class ModelBase(object):
    def __init__(self, **kwargs):
        pass

    def __repr__(self):
        return NotImplemented

    def __str__(self):
        return self.__repr__()

    def __call__(self, inputs, **kwargs):
        return self.predict(inputs, **kwargs)

    def predict(self, inputs, **kwargs):
        pass

    def train(self, X, y, **kwargs):
        pass
