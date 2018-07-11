"""Base class for working with dataset to be used for this project.

   @author
     Victor I. Afolabi
     Artificial Intelligence & Software Engineer.
     Email: javafolabi@gmail.com
     GitHub: https://github.com/victor-iyiola

   @project
     File: __init__.py
     Created on 11 July, 2018 @ 8:01 PM.

   @license
     MIT License
     Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""

from data.dataset import Dataset
from data.loader import DataLoader

__all___ = [
    'Dataset',
    'DataLoader'
]
