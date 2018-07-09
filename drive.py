"""Remote Driving for Simulator in "Autonomous Mode".

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
import numpy as np
import cv2

import socketio

from base64 import b64decode
from io import BytesIO
from PIL import Image

import eventlet.wsgi
from flask import Flask

import data

# Helper file to load frozen model.
from frozen_model import load

# Global objects.
sio, driver = socketio.Server(), None


class Drive:
    def __init__(self, frozen_file, **kwargs):
        self.img_size = kwargs.get('img_size') or data.img_size
        # Desired speed limit.
        self._max_speed = kwargs.get('max_speed') or 25
        self._min_speed = kwargs.get('min_speed') or 10
        self._speed_limit = self._max_speed

        self.graph = load(frozen_file=frozen_file)
        self.sess = tf.Session(graph=self.graph)

    def connect(self, sid, env):
        # Connection info.
        print('Socket ID:', sid)
        print('Host: {HTTP_HOST} Interface: {GATEWAY_INTERFACE}'.format(**env))

        # Drive.
        self.drive(0, self._speed_limit)

    def drive(self, steering_angle, throttle):
        self._speed_limit = (self._min_speed if throttle > self._speed_limit
                             else self._max_speed)
        # (1 - A^2) - (s/L)^2
        throttle = 1.0 - steering_angle ** 2 - \
            (throttle / self._speed_limit) ** 2

        sio.emit(event="steer",
                 data={
                     "steering_angle": str(steering_angle),
                     "throttle": str(throttle)
                 }, skip_sid=True)

    def telemetry(self, sid, data):
        print('Socket ID:', sid)
        # Collect data.
        # steering_angle = float(data['steering_angle'])
        # throttle = float(data['throttle'])
        image = data['image']

        pred_angle = self.predict(image)
        print('Prediction:', pred_angle)

        # Drive the car with these parameters.
        # self.drive(steering_angle, throttle)

    def _img_preprocess(self, image):
        # Load base64 image into a NumPy array of pixels.
        image = BytesIO(b64decode(image))
        image = np.asarray(Image.open(image), dtype=np.float32)

        # Crop the image (removing the sky at the top and the car front at the bottom).
        image = image[60:-25, :, :]

        # Resize the image to the input shape used by the network model.
        image = cv2.resize(
            image, (self.img_size, self.img_size), cv2.INTER_AREA)

        # Convert the image from RGB to YUV (This is what the NVIDIA model does)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

        # Expand dimension to [1, height, width, channel]
        image = np.expand_dims(image, axis=0)
        return image

    def predict(self, image):
        # Apply image pre-processing.
        # image = self._img_preprocess(image)
        # _fake_label = np.zeros(shape=(1,), dtype=np.float32)

        # # Get image placeholder & prediction tensor.
        img_plhd = self.graph.get_tensor_by_name('nvidia/placeholders/image:0')
        output = self.graph.get_tensor_by_name(
            'nvidia/model/layers/output/BiasAdd:0')
        init = self.graph.get_operation_by_name(
            'nvidia/data/initializer/train_data')
        # label_plhd = self.graph.get_tensor_by_name('nvidia/placeholders/labels:0')

        # print(img_plhd.shape, label_plhd.shape)
        # print(image.shape, _fake_label.shape)
        dataset = data.make_dataset(img_plhd, labels=None)
        iterator = dataset.make_one_shot_iterator()
        items = iterator.get_next()

        feed_dict = {img_plhd: np.array([image], dtype=np.string_)}
        self.sess.run(init, feed_dict=feed_dict)
        o = self.sess.run(output)
        print(o)
        # iterator = self.graph.get_tensor_by_name('nvidia/data/iterator/Iterator:0')
        # elements = self.graph.get_tensor_by_name('nvidia/data/iterator/IteratorGetNext:0')
        # print(self.sess.run(elements, feed_dict={img_plhd: image, label_plhd: _fake_label}))
        # for op in self.graph.get_operations():
        #   print(op.name)

        return 3.14


@sio.on("connect")
def connect(sid, env):
    driver.connect(sid, env)


@sio.on("telemetry")
def telemetry(sid, data):
    driver.telemetry(sid, data)


if __name__ == '__main__':
    # Command line arguments.
    parser = argparse.ArgumentParser(
        description='Remote Driving for Simulator in "Autonomous Mode".',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('-m', '--model', dest='frozen_file', type=str,
                        default='saved/frozen/nvidia.pb',
                        help='Frozen model to be used. Must have a `.pb` extension. '
                             '(default: saved/frozen/nvidia.pb)')

    # Parse known arguments.
    args = parser.parse_args()

    # Log Parsed arguments.
    print('{0}\n{1:^55}\n{0}'.format('-' * 55, 'Command Line Arguments'))
    for k, v in vars(args).items():
        print('{:<20} = {:>30}'.format(k, str(v)))
    print('{}\n'.format('-' * 55))

    # Driver object.
    driver = Drive(frozen_file=args.frozen_file)
    driver.predict('')

    # Flask app.
    # app = Flask(__name__)
    #
    # # SocketIO as a middleware.
    # app = socketio.Middleware(socketio_app=sio, wsgi_app=app)
    #
    # # Start eventlet server, listen on port 4567.
    # eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
