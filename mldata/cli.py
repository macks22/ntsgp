import os
import logging
import argparse


def add_mldata_arguments(parser):
    parser.add_argument(
        '-c', '--config-file',
        help='path to feature guide configuration file')
    parser.add_argument(
        '-d', '--data-file',
        help='path to data file (csv format)')
    parser.add_argument(
        '-rcs', '--remove-cold-start',
        action='store_true', default=False,
        help='boolean flag to remove cold start records; false by default')
    parser.add_argument(
        '-ws', '--window-size',
        type=int, default=4,
        help='limit on number of previous terms to include in training set')

    try: # Add verbosity flag if not present in parser.
        parser.add_argument(
            '-v', '--verbose',
            type=int, default=1,
            help='adjust verbosity of logging output')
    except argparse.ArgumentError:
        pass


def default_parser():
    parser = argparse.ArgumentParser(
        description='Perform machine learning experiments using Pandas')
    add_mldata_arguments(parser)
    return parser


def parse_and_setup(parser):
    args = parser.parse_args()

    # Setup logging.
    logging.basicConfig(
        level=(logging.DEBUG if args.verbose == 2 else
               logging.INFO if args.verbose == 1 else
               logging.ERROR),
        format="[%(asctime)s][%(levelname)s][%(processName)s][%(process)d]:"
               " %(message)s")
    return args


def mkdir_ifnexists(dirname):
    try:
        os.mkdir(dirname)
    except OSError:
        pass

