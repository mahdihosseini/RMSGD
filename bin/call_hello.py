#! /usr/bin/env python3

'''
Just print a hello world!
'''

from argparse import ArgumentParser

from imrsv.production import hello


argument_parser = ArgumentParser(description=__doc__)
argument_parser.add_argument('what')


if __name__ == '__main__':
    args = argument_parser.parse_args()
    hello(args.what)
