"""Frozen model utility package.

   @author 
     Victor I. Afolabi
     Artificial Intelligence & Software Engineer.
     Email: javafolabi@gmail.com
     GitHub: https://github.com/victor-iyiola
  
   @project
     File: __init__.py.py
     Created on 09 June, 2018 @ 4:37 PM.
  
   @license
     MIT License
     Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""
from frozen_model.freeze import freeze
from frozen_model.load import load

__all__ = [
    'freeze',  # For freezing TensorFlow model to a single protobuf file.
    'load',  # For loading frozen protobuf model into a graph.
]
