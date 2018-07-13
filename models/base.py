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


class Base(keras.Model):
    """`Model` groups layers into an object with training and inference features.

    There are two ways to instantiate a `Model`:

    1 - With the "functional API", where you start from `Input`,
    you chain layer calls to specify the model's forward pass,
    and finally you create your model from inputs and outputs:

    ```python
    import tensorflow as tf
    from tensorflow import keras

    inputs = keras.Input(shape=(3,))
    x = keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
    outputs = keras.layers.Dense(5, activation=tf.nn.softmax)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    ```

    2 - By subclassing the `Model` class: in that case, you should define your
    layers in `__init__` and you should implement the model's forward pass
    in `call`.

    ```python
    import tensorflow as tf
    from models import Base

    class MyModel(Base):

      def __init__(self):
        super(MyModel, self).__init__(name='MyModel')

        self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)

      def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

    model = MyModel()
    ```

    If you subclass `Model`, you can optionally have
    a `training` argument (boolean) in `call`, which you can use to specify
    a different behavior in training and inference:

    ```python
    import tensorflow as tf
    from tensorflow import keras
    from models import Base

    class MyModel(Base):

      def __init__(self):
        super(MyModel, self).__init__(name='MyModel')

        self.dense1 = keras.layers.Dense(4, activation=tf.nn.relu)
        self.dense2 = keras.layers.Dense(5, activation=tf.nn.softmax)
        self.dropout = keras.layers.Dropout(0.5)

      def call(self, inputs, training=False):
        x = self.dense1(inputs)
        if training:
          x = self.dropout(x, training=training)
        return self.dense2(x)

    model = MyModel()
    ```
    """

    def __init__(self, **kwargs):
        super(Base, self).__init__(name=kwargs.get('name', 'Base'))

        # Extract keyword arguments.
        self._verbose = kwargs.get('verbose', 1)

    def __repr__(self):
        return f'models.Base(verbose={self._verbose})'

    def __str__(self):
        return self.__repr__()

    def call(self, inputs, **kwargs):
        """Calls the model on new inputs.

        In this case `call` just reapplies all ops in the graph to the new inputs
        (e.g. build a new computational graph from the provided inputs).

        Arguments:
            inputs: A tensor or list of tensors.
            training: Boolean or boolean scalar tensor, indicating whether to run
            the `Network` in training mode or inference mode.
            mask: A mask or list of masks. A mask can be
                either a tensor or None (no mask).

        Returns:
            A tensor if there is a single output, or
            a list of tensors if there are more than one outputs.
        """
        return NotImplemented

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)
