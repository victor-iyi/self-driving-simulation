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

import socketio
import eventlet.wsgi
from flask import Flask

# Helper file to load frozen model.
from frozen_model import load
from data import create_tiny_dataset

# Global objects.
sio, driver = socketio.Server(), None


class Drive:
    def __init__(self, model_path, **kwargs):
        # Desired speed limit.
        self._max_speed = kwargs.get('max_speed') or 25
        self._min_speed = kwargs.get('min_speed') or 10
        self._speed_limit = self._max_speed

        self.graph = load(frozen_file=model_path)
        self.sess = tf.Session(graph=self.graph)

    def connect(self, sid, env):
        # Connection info.
        print('Socket ID:', sid)
        print('Host: {HTTP_HOST} Interface: {GATEWAY_INTERFACE}'
              .format(**env))

        # Drive.
        self.drive(0, self._speed_limit)

    def drive(self, steering_angle, throttle):
        self._speed_limit = (self._min_speed if throttle > self._speed_limit
                             else self._max_speed)
        # (1 - A^2) - (s/L)^2
        throttle = 1.0 - steering_angle ** 2 - (throttle / self._speed_limit) ** 2

        sio.emit(
            event="steer",
            data={
                "steering_angle": str(steering_angle),
                "throttle": str(throttle)
            },
            skip_sid=True
        )

    def telemetry(self, sid, data):
        # Collect data.
        steering_angle = float(data['steering_angle'])
        throttle = float(data['throttle'])

        # Drive the car with these parameters.
        self.drive(steering_angle, throttle)

    def predict(self, image):
        # out_tensor_name = "nvidia/model/layers/output/BiasAdd:0"
        # prediction = self.graph.get_tensor_by_name(out_tensor_name)
        # feed_dict = {}
        # self.sess.run(prediction, feed_dict=feed_dict)
        # print(prediction)
        # dataset = create_tiny_dataset((image,))
        # print(dataset)

        # iter_op = self.graph.get_operation_by_name('nvidia/Iterator')
        # print(self.sess.run(iter_op.values()))
        # # print(iter_op.values())
        #
        # get_next = self.graph.get_operation_by_name('nvidia/IteratorGetNext')
        # print(get_next.values())
        # # iter_val = iter_op.values()[0]
        # img_tensor = get_next.values()[0]
        # print(self.sess.run(img_tensor))

        for op in self.graph.get_all_collection_keys():
            print('{}\n'.format(op))


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

    parser.add_argument('-m', '--model', dest='model_path', type=str,
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
    driver = Drive(model_path=args.model_path)
    driver.predict('')

    # # Flask app.
    # app = Flask(__name__)
    #
    # # SocketIO as a middleware.
    # app = socketio.Middleware(socketio_app=sio, wsgi_app=app)
    #
    # # Start eventlet server, listen on port 4567.
    # eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
