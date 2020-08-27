"""
MIT License

Copyright (c) 2020 Mahdi S. Hosseini and Mathieu Tuli

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from argparse import ArgumentParser
# import logging

from .train import args as train_args, main as train_main

parser = ArgumentParser(description=__doc__)
# parser.add_argument(
#     '-vv', '--very-verbose', action='store_true',
#     dest='very_verbose',
#     help="Set verbose. In effect, set --log-level to DEBUG.")
# parser.add_argument(
#     '-v', '--verbose', action='store_true',
#     dest='verbose',
#     help="Set verbose. In effect, set --log-level to INFO.")
# parser.set_defaults(verbose=False)
# parser.set_defaults(very_verbose=False)
# parser.add_argument('--log-level', type=LogLevel.__getitem__,
#                     default=LogLevel.INFO,
#                     choices=LogLevel.__members__.values(),
#                     dest='log_level',
#                     help="Log level.")
subparser = parser.add_subparsers(dest='command')
train_subparser = subparser.add_parser(
    'train', help='Train commands')
train_args(train_subparser)

args = parser.parse_args()
# if str(args.log_level) == 'DEBUG' or args.very_verbose:
#     logging.root.setLevel(logging.DEBUG)
# elif str(args.log_level) == 'INFO' or args.verbose:
#     logging.root.setLevel(logging.INFO)
# elif str(args.log_level) == 'WARNING':
#     logging.root.setLevel(logging.WARNING)
# elif str(args.log_level) == 'ERROR':
#     logging.root.setLevel(logging.ERROR)
# elif str(args.log_level) == 'CRITICAL':
#     logging.root.setLevel(logging.CRITICAL)
# else:
#     logging.root.setLevel(logging.INFO)
#     args.log_level = 'INFO'
#     logging.warning(
#         f"Frontend: Log level \"{args.log_level}\" unknown, defaulting" +
#         " to INFO.")
# logging.info(f"AdaS: Log Level set to {str(args.log_level)}")

# logging.info("AdaS: Main")

if str(args.command) == 'train':
    train_main(args)
# if str(args.command) == 'lrrt':
#     lrrt_main(args)
else:
    # logging.critical(f"AdaS: Unknown subcommand {args.command}")
    ...
