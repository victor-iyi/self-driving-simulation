"""Load a frozen model.

   @author
     Victor I. Afolabi
     Artificial Intelligence & Software Engineer.
     Email: javafolabi@gmail.com
     GitHub: https://github.com/victor-iyiola

   @project
     File: load.py
     Created on 09 June, 2018 @ 4:38 PM.

   @license
     MIT License
     Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""
import argparse

import tensorflow as tf


def load(frozen_file: str, **kwargs):
  """Returns a graph loaded from a protobuf file (.pb).

    Args:
        frozen_file (str): Path to a protobuf file (.pb) otherwise it appends it.
            (default {'saved/frozen/model.pb'}

    Keyword Args:
        input_map: A dictionary mapping input names (as strings) in `graph_def`
            to `Tensor` objects. The values of the named input tensors in the
            imported graph will be re-mapped to the respective `Tensor` values.

        return_elements: A list of strings containing operation names in
            `graph_def` that will be returned as `Operation` objects; and/or
            tensor names in `graph_def` that will be returned as `Tensor` objects.

        prefix: (Optional.) A prefix that will be prepended to the names in
            `graph_def`. If not provided, it'll be inferred from `froze_file` path.
            Note that this does not apply to imported function names.
            (default {"import"}).

        producer_op_list: (Optional.) An `OpList` proto with the (possibly stripped)
            list of `OpDef`s used by the producer of the graph. If provided,
            unrecognized attrs for ops in `graph_def` that have their default value
            according to `producer_op_list` will be removed. This will allow some more
            `GraphDef`s produced by later binaries to be accepted by earlier binaries.

    Raises:
        TypeError: If `graph_def` is not a `GraphDef` proto,
            `input_map` is not a dictionary mapping strings to `Tensor` objects,
            or `return_elements` is not a list of strings.

        ValueError: If `input_map`, or `return_elements` contains names that
            do not appear in `graph_def`, or `graph_def` is not well-formed (e.g.
            it refers to an unknown tensor).

    Examples:
        ```python
        >>> frozen_file = 'saved/frozen/model.pb'
        >>> graph = load(frozen_file=frozen_file)
        >>> ops = len(graph.get_operations())
        >>> print('{:,} ops in graph.'.format(ops))
        95 ops in graph.
        ```

    Returns:
        tf.Graph: TF graph loaded from `frozen_file`.
    """
  if not tf.gfile.Exists(frozen_file):
    raise FileNotFoundError('{} was not found.'.format(frozen_file))

  # Prefix to node names.
  prefix = kwargs.get('prefix') or frozen_file.split('/')[-1].split('.')[0]

  # Read the protobuf graph
  with tf.gfile.GFile(frozen_file, mode='rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

  # Load graph_def into default graph
  with tf.Graph().as_default() as graph:
    # Import graph def to default graph.
    tf.import_graph_def(
        graph_def=graph_def, return_elements=[], name=prefix, **kwargs)

  return graph


if __name__ == '__main__':
  # Command line argument parser.
  parser = argparse.ArgumentParser(
      description='Load frozen TensorFlow model '
      '(protobuf binary) into a TensorFlow graph.',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # File & directory arguments.
  parser.add_argument(
      '-f',
      dest='frozen_file',
      type=str,
      default='../saved/frozen/nvidia.pb',
      help='Path to a protobuf file (.pb), where frozen model is saved.')

  # Parse known arguments.
  args = parser.parse_args()

  print('{0}\n{1:^55}\n{0}'.format('-' * 55, 'Command Line Arguments'))
  for k, v in vars(args).items():
    print('{:<20} = {:>30}'.format(k, v))
  print('{}\n'.format('-' * 55))

  # Usage:
  graph = load(frozen_file=args.frozen_file)

  ops = len(graph.get_operations())
  print('{:,} ops in graph.'.format(ops))
