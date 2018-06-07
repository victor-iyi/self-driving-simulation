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

    parser.add_argument('-d', dest='data_dir', type=str, default='simulated_data',
                        help='Directory where simulated data is stored.')

    args = parser.parse_args()

    print('{0}\nArguments\n{0}'.format('-' * 45))
    for k, v in vars(args).items():
        print('{:<20} = {}'.format(k, v))
    print('{}'.format('-' * 45))

    main(args=args)
