"""Frozen Model class for Machine Learning Models to be used for this project.

   @author
     Victor I. Afolabi
     Artificial Intelligence & Software Engineer.
     Email: javafolabi@gmail.com
     GitHub: https://github.com/victor-iyiola

   @project
     File: frozen.py
     Created on 11 July, 2018 @ 8:00 PM.

   @license
     MIT License
     Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""
from models.base import BaseModel


class Frozen(BaseModel):
    def __init__(self, **kwargs):
        pass

    def __repr__(self):
        return 'models.Frozen()'

    def predict(self, inputs, **kwargs):
        pass

    def load(self):
        pass
