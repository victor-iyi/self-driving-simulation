"""Helper file for working with training dataset.

   @author
     Victor I. Afolabi
     Artificial Intelligence & Software Engineer.
     Email: javafolabi@gmail.com
     GitHub: https://github.com/victor-iyiola

   @project
     File: data.py
     Created on 08 June, 2018 @ 8:27 PM.

   @license
     MIT License
     Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""
import os

import numpy as np
import pandas as pd

import tensorflow as tf

from utils import Keys

# Path to where training data is collected (from the simulator).
data_dir = os.path.join(os.path.dirname(os.path.curdir), 'simulations')

# CSV File generated by the Simulator.
CSV_FILENAME = os.path.join(data_dir, 'driving_log.csv')
IMG_DIR = os.path.join(data_dir, 'IMG')

# CSV File header names.
FILE_NAMES = [
    'center_path', 'left_path', 'right_path', 'steering_angle', 'throttle',
    'brake', 'speed'
]

# Image dimensions.
img_size, channels = 32, 3


# Use standard TensorFlow operations to resize the image to a fixed shape.
def _img_func(row: dict):
  """Use standard TensorFlow ops to resize image to a fixed shape.

    Args:
        row (dict): Containing base64 image.

    Returns:
        dict: {Keys.IMAGES: tf.Tensor}
            Cropped and/or padded image. If `images` was 4-D, a 4-D float
                Tensor of shape `[batch, new_height, new_width, channels]`.
                If `images` was 3-D, a 3-D float Tensor of shape
                `[new_height, new_width, channels]`.
    """
  image_string = tf.decode_base64(row[Keys.IMAGES])
  image_decoded = tf.image.decode_image(image_string)
  image_resized = tf.image.resize_image_with_crop_or_pad(
      image=image_decoded, target_height=img_size, target_width=img_size
  )
  # Cast tf.uint8 into tf.float32
  image_cast = tf.cast(image_resized, tf.float32)

  return {Keys.IMAGES: image_cast, Keys.LABELS: row[Keys.LABELS]}


def _parser(row: dict):
  """Reads an image from a file, decodes it into dense Tensor.

    Args:
        row (dict): Dictionary containing features & labels.

    Returns:
        dict: {Keys.FILENAMES: tf.float32, Keys.LABEL: tf.float32}
            decoded image & reshaped label.

    """
  # Read the contents in filename as a string.
  image_string = tf.read_file(row[Keys.IMAGES])

  # Decode string into dense Tensor
  image_decoded = tf.image.decode_image(image_string)

  # Crop and/or pads an image to a target width and height.
  image_resized = tf.image.resize_image_with_crop_or_pad(
      image=image_decoded, target_height=img_size, target_width=img_size
  )
  # Cast tf.uint8 into tf.float32
  image_cast = tf.cast(image_resized, tf.float32)

  # Reshape label.
  label_reshape = tf.reshape(row[Keys.LABELS], shape=(1,))

  # Return parsed image & label.
  return {Keys.IMAGES: image_cast, Keys.LABELS: label_reshape}


def make_dataset(features: np.ndarray, labels: np.ndarray = None, **kwargs):
  """Returns a dataset object from tensor slices.

    Args:
        features (np.ndarray): Features (filenames) or a image for prediction.
        labels (np.ndarray): List of associated labels to features.

    Keyword Args:
        shuffle (bool): Maybe shuffle dataset.
            (default {True})
        buffer_size (int): Amount of data to shuffle randomly at a time.
            (default {1000})
        batch_size (int): Mini-batch size.
            (default {128})

    Returns:
        tf.data.Dataset: Dataset object.
    """
  # Extract keyword arguments.
  shuffle = kwargs.get('shuffle', True)
  buffer_size = kwargs.get('buffer_size') or 1000
  batch_size = kwargs.get('batch_size') or 128

  # Change map function depending on parameters.
  map_fn = _parser if labels is not None else _img_func

  # Read CSV file into dataset object.
  tensors = {Keys.IMAGES: features, Keys.LABELS: labels}
  dataset = tf.data.Dataset.from_tensor_slices(tensors)
  dataset = dataset.map(map_fn)

  # Apply transformation steps...
  dataset = dataset.batch(batch_size=batch_size)
  if shuffle:
    dataset = dataset.shuffle(buffer_size=buffer_size)

  return dataset


def create_tiny_dataset(image_string):
  dataset = tf.data.Dataset.from_tensor_slices(image_string)
  dataset.apply(dataset)
  return dataset


def load_data(filename: str, **kwargs):
  """Helper method for loading image filenames & labels from CSV file.

    Args:
        filename (str): Path to a CSV file. Containing image paths.

    Keyword Args:
        header (list): List of CSV headers.
        feature_cols (list or str): (optional) Names of feature columns.
        label_col (str): Name of label column.

    Raises:
        FileNotFoundError: `filename` was not found!

    Returns:
        tuple: (image filenames, labels)
            Filenames and Labels are both NumPy arrays.
    """
  # Extract keyword arguments.
  header = kwargs.get('header') or FILE_NAMES
  feature_cols = kwargs.get('feature_cols') or FILE_NAMES[0]
  label_col = kwargs.get('label_col') or FILE_NAMES[-1]

  if not os.path.isfile(filename):
    raise FileNotFoundError('{} was not found!'.format(filename))

  # Read dataset from cvs file if there are no features.
  df = pd.read_csv(filename, names=header)

  # Extract features & labels.
  img_filenames = df[feature_cols].astype(str).values
  labels = df[label_col].astype(np.float32).values

  return img_filenames, labels
