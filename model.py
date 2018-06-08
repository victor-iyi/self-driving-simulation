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
import argparse
import tensorflow as tf

import data


# noinspection PyAbstractClass
class Model(tf.keras.Model):
    def __init__(self, args):
        super(Model, self).__init__()

        # Command line arguments.
        self.args = args

        # Convolutional Layers / Feature Extraction
        self.conv1 = tf.keras.layers.Conv2D(filters=24, kernel_size=5,
                                            padding='same', activation='elu')
        self.conv2 = tf.keras.layers.Conv2D(filters=36, kernel_size=5,
                                            padding='same', activation='elu')
        self.conv3 = tf.keras.layers.Conv2D(filters=48, kernel_size=5,
                                            padding='same', activation='elu')
        self.conv4 = tf.keras.layers.Conv2D(filters=64, kernel_size=3,
                                            padding='same', activation='elu')
        self.conv5 = tf.keras.layers.Conv2D(filters=64, kernel_size=3,
                                            padding='same', activation='elu')

        # Flatten & apply dropout.
        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(rate=self.args.dropout)

        # Fully connected / Dense layers.
        self.dense1 = tf.keras.layers.Dense(units=100, activation='elu')
        self.dense2 = tf.keras.layers.Dense(units=50, activation='elu')
        self.dense3 = tf.keras.layers.Dense(units=10, activation='elu')

        # Prediction / Output layer.
        self.pred_layer = tf.keras.layers.Dense(units=1)

    def call(self, inputs, **kwargs):
        with tf.name_scope('model'):
            # Network layers.
            with tf.name_scope('layers'):
                # Input layer.
                with tf.name_scope('input'):
                    net = tf.reshape(inputs,
                                     shape=(self.args.batch_size, self.args.img_size,
                                            self.args.img_size, self.args.img_depth),
                                     name='reshape')

                # Convolutional layers.
                with tf.name_scope('feature_extraction'):
                    net = self.conv2(self.conv1(net))
                    net = self.conv4(self.conv3(net))
                    net = self.conv5(net)

                # Fully connected / Dense layers.
                with tf.name_scope('fully_connected'):
                    net = self.flatten(net)
                    net = self.dense1(self.dropout(net))
                    net = self.dense3(self.dense2(net))

                # Prediction / Output layer.
                with tf.name_scope('prediction'):
                    net = self.pred_layer(net)

        return net


def train(args):
    dataset = data.load_data()
    # model = Model(args)
    print(dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data & checkpoint arguments.
    parser.add_argument('-d', dest='data_dir', type=str, default='./simulations/',
                        help='Directory where simulated data is stored.')
    parser.add_argument('-s', dest='save_path', type=str, default='./saved/model.ckpt',
                        help='Checkpoint saved path.')

    # Training arguments.
    parser.add_argument('-e', dest='epochs', type=int, default=10000,
                        help='Number of training epochs.')
    parser.add_argument('-b', dest='batch_size', type=int, default=128,
                        help='Mini-batch size.')
    parser.add_argument('-lr', dest='learning_rate', type=float, default=1e-2,
                        help='Optimizer\'s learning rate.')

    # Parsed arguments.
    args = parser.parse_args()

    print('{0}\n{1:^45}\n{0}'.format('-' * 45, 'Command Line Arguments'))
    for k, v in vars(args).items():
        print('{:<20} = {:>20}'.format(k, v))
    print('{}\n'.format('-' * 45))

    train(args=args)
