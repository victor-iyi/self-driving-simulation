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


def main(args):
    print(args)


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
        print('{:<15} = {:>20}'.format(k, v))
    print('{}'.format('-' * 45))

    main(args=args)
