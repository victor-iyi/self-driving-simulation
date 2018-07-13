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

from tensorflow import keras


class ModelBase(keras.Model):

    def __init__(self, **kwargs):
        super(ModelBase, self).__init__(name='Model')

        # Extract keyword arguments.
        self._verbose = kwargs.get('verbose', 1)

    def __repr__(self):
        return f'ModelBase(verbose={self._verbose})'

    def __str__(self):
        return self.__repr__()

    def call(self, inputs, **kwargs):
        return NotImplemented

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)
