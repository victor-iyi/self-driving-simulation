"""Free Model to a single Protocol Buffer (or ProtoBuf) `.pb` file.

   @author 
     Victor I. Afolabi
     Artificial Intelligence & Software Engineer.
     Email: javafolabi@gmail.com
     GitHub: https://github.com/victor-iyiola
  
   @project
     File: freeze.py
     Created on 09 June, 2018 @ 2:13 PM.
  
   @license
     MIT License
     Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""
import argparse
import os

import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants


def freeze(ckpt_dir: str, output_nodes: list, **kwargs):
    """Freeze a given model from it's checkpoint.

    Args:
        ckpt_dir (str): Path to a checkpoint directory. Checkpoint directory
            must contain: "*.data-XXX-of-XXX", "*.index" and "*.meta" files.

        output_nodes (list): A list of (at least one) valid node names.
            Operations after this node will be pruned. A good rule of thumb is to
            specify nodes for output predictions, and/or accuracy.
            Eg: ['model/model/layers/prediction/dense/BiasAdd']

    Keyword Args:
        frozen_file (str): Path to a protobuf file (.pb) otherwise it appends it.
            (default {'saved/frozen/model.pb'}
        clear_devices (bool): Allow TensorFlow to control loading, where it wants
            operations to be calculated. (default {True})

    Examples:
        ```python
        >>> ckpt_dir = 'saved/models/'
        >>> output_nodes = ['model/model/layers/prediction/dense/BiasAdd']
        >>> frozen_file = 'saved/frozen/model.pb'
        >>> freeze(ckpt_dir=ckpt_dir,
        ...        output_nodes=output_nodes,
        ...        frozen_file=frozen_file)
        >>>
        Converted 18 variables to const ops.
        Frozen model saved to "./saved/frozen/model.pb".
        95 nodes (ops) in the final output graph.
        ```

    Raises:
        NotADirectoryError:
            Directory does not exist! `ckpt_dir`

    """
    # Make sure `ckpt_dir` is a directory & exists.
    if not tf.gfile.IsDirectory(ckpt_dir):
        raise NotADirectoryError('Directory does not exist! {}'.format(ckpt_dir))

    # Check `output_nodes` isn't empty.
    if output_nodes is []:
        raise ValueError('`output_nodes` must contain at least one valid node name.')

    # Allow TensorFlow to control on loading, where it wants operations to be calculated.
    clear_devices = kwargs.get('clear_devices') or True

    # Destination frozen file name.
    frozen_file = kwargs.get('frozen_file') or 'saved/frozen/model.pb'
    frozen_file = os.path.abspath((frozen_file if frozen_file.endswith('.pb')
                                   else '{}.pb'.format(frozen_file)))

    # If the frozen path is not a directory, create it.
    if not tf.gfile.IsDirectory(os.path.dirname(frozen_file)):
        tf.gfile.MakeDirs(os.path.dirname(frozen_file))

    # Get the latest checkpoint.
    ckpt_path = os.path.abspath(tf.train.latest_checkpoint(ckpt_dir))

    # File holding graph metadata.
    meta_file = '{}.meta'.format(ckpt_path)

    # Import meta graph & retrieve Saver object.
    saver = tf.train.import_meta_graph(meta_file,
                                       clear_devices=clear_devices)

    # Retrieve protobuf graph definition.
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with tf.Session() as sess:
        saver.restore(sess=sess, save_path=ckpt_path)
        try:
            # Convert variables to constants.
            output_graph_def = convert_variables_to_constants(sess=sess,
                                                              input_graph_def=input_graph_def,
                                                              output_node_names=output_nodes)

            # Write serialized string into frozen protobuf file.
            with tf.gfile.GFile(frozen_file, mode='wb') as f:
                f.write(output_graph_def.SerializeToString())
            print('Frozen model saved to "{}".'.format(frozen_file))
            print('{:,} nodes (ops) in the final output graph.'.format(len(output_graph_def.node)))
        except AssertionError as e:
            print('ERROR: {}.'.format(e))


def _str2list(string: str):
    """Command line arg for parsing string to list"""
    return string.split(', ') or string.split(',') or string.split(': ') or string.split(':')


if __name__ == '__main__':
    # Command line arguments.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-d', dest='ckpt_dir', type=str, default='./saved/models/',
                        help='Directory containing checkpoint files.')
    parser.add_argument('-f', dest='frozen_file', type=str, default='./saved/frozen/nvidia.pb')

    parser.add_argument('-o', dest='output_nodes', type=_str2list,
                        default='model/model/layers/prediction/dense/BiasAdd',
                        help='What are the names of useful output nodes for inference (or metrics). '
                             'NOTE: Output nodes must be separated by (",", ", ", ":" or ": ").')
    parser.add_argument('-c', dest='clear_devices', type=bool, default=True,
                        help='Allow TensorFlow to control on loading, where '
                             'it wants operations to be calculated.')

    # Parse known arguments.
    args = parser.parse_args()

    print('{0}\n{1:^55}\n{0}'.format('-' * 55, 'Command Line Arguments'))
    for k, v in vars(args).items():
        print('{:<20} = {:>30}'.format(k, str(v)))
    print('{}\n'.format('-' * 55))

    freeze(ckpt_dir=args.ckpt_dir,
           output_nodes=args.output_nodes,
           frozen_file=args.frozen_file,
           clear_devices=args.clear_devices)
