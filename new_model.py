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
    def __init__(self, sess: tf.Session, data: Dataset, **kwargs):
        self._sess = sess
        self._data = data

        self._verbose = kwargs.get('verbose', 1)
        self._save_dir = kwargs.get('save_dir', 'saved/')
        self._global_step = tf.train.get_or_create_global_step()

        self.X_plhd = tf.placeholder(dtype=tf.float32, shape=(None,))
        self.y_plhd = tf.placeholder(dtype=tf.float32, shape=(None,))

        # self.build_graph()
        # self.build_eval_graph()

        # self._saver = tf.train.Saver()
        self.restore()

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
