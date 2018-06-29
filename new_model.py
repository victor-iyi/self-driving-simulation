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


class Mode(object):
    train = 'TRAIN'
    predict = 'PREDICT'
    evaluate = 'EVALUATE'


class Model(object):
    def __init__(self, sess: tf.Session, data: Dataset, **kwargs):
        # Extract keyword arguments.
        self._verbose = kwargs.get('verbose', 1)
        self._dropout = kwargs.get('dropout', 0.5)
        self._save_dir = kwargs.get('save_dir', 'saved/')
        self._img_dim = kwargs.get('img_dim', (200, 200, 3))

        # Session & data.
        self._sess, self._data = sess, data

        # Placeholders.
        self.X_plhd = tf.placeholder(dtype=tf.float32,
                                     shape=(None, *self._img_dim))
        self.y_plhd = tf.placeholder(dtype=tf.float32, shape=(None,))

        # Model Hyperparameters.
        self._mode = Mode.predict
        self._num_classes = 1
        self._global_step = tf.train.get_or_create_global_step()

        self.build_graph()
        # self.build_eval_graph()

        # self._saver = tf.train.Saver()
        self.restore()

    def __call__(self, X, **kwargs):
        return self.predict(X, **kwargs)

    def __repr__(self):
        return '<Model mode={}>'.format(self._mode)

    def train(self):
        pass

    def predict(self):
        pass

    def evaluate(self):
        pass

    def build_graph(self):
        with tf.name_scope('Model'):
            with tf.name_scope('FeatureExtraction'):
                # Layer 1.
                net = tf.layers.conv2d(net, filters=16, kernel_size=5,
                                       activation=tf.nn.relu, name="conv1")
                net = tf.layers.max_pooling2d(net, pool_size=2, strides=2,
                                              padding='same', name="pool1")
                # Layer 2.
                net = tf.layers.conv2d(net, filters=32, kernel_size=5,
                                       activation=tf.nn.relu, name="conv2")
                net = tf.layers.max_pooling2d(net, pool_size=2, strides=2,
                                              padding='same', name="pool2")
                # Layer 3.
                net = tf.layers.conv2d(net, filters=64, kernel_size=3,
                                       activation=tf.nn.relu, name="conv3")
                net = tf.layers.max_pooling2d(net, pool_size=2, strides=2,
                                              padding='same', name="pool3")

            with tf.name_scope('FullyConnected'):
                # Flatten & Dropout.
                net = tf.layers.flatten(net)
                net = tf.layers.dropout(net, rate=self._dropout,
                                        training=self._mode == Mode.train)
                # Fully connected layers.
                net = tf.layers.dense(net, units=1024, activation='relu',
                                      name='dense1')
                net = tf.layers.dense(net, units=128, activation='relu')

            with tf.name_scope('Prediction'):
                # Prediction layer.
                self.y_pred = tf.layers.dense(net, units=self._num_classes,
                                              name='logits')

    def build_eval_graph(self):
        pass

    def restore(self):
        # Restore checkpoint properly.
        if tf.gfile.IsDirectory(self._save_dir):
            try:
                last_ckpt = tf.train.latest_checkpoint(self._save_dir)
                self._saver.restore(sess=self._sess, save_path=last_ckpt)
                print('Restored model checkpoint: {}'.format(last_ckpt))
                return
            except Exception:
                print('WARN: Could not load checkpoint.')
        else:
            print('INFO: Creating checkpoint directory: {}'.format(ckpt_dir))
            tf.gfile.MakeDirs(self._save_dir)

        print('Initializing global variables instead.')
        self._sess.run(tf.global_variables_initializer())

    def _predict(self, **kwargs):
        pass

    @property
    def sess(self):
        return self._sess

    @property
    def data(self):
        return self._data


if __name__ == '__main__':
    filename = 'simulations/driving_log.csv'
    data = Dataset(filename)
    with tf.Graph().as_default(), tf.Session() as sess:
        model = Model(sess=sess, data=data)
