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
from models.base import Base


class Frozen(Base):
    def __init__(self, **kwargs):
        super(Frozen, self).__init__(name='Frozen')

    def __repr__(self):
        return 'models.Frozen()'

    def call(self, inputs, **kwargs):
        pass
