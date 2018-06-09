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


def main(args):
    print(args)


if __name__ == '__main__':
    # Command line arguments.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-c', dest='ckpt_dir', type=str, default='./saved/models',
                        help='Directory containing checkpoint files.')

    # Parse known arguments.
    args = parser.parse_args()

    print('{0}\n{1:^55}\n{0}'.format('-' * 55, 'Command Line Arguments'))
    for k, v in vars(args).items():
        print('{:<20} = {:>30}'.format(k, v))
    print('{}\n'.format('-' * 55))

    main(args=args)
