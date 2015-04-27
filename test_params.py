import os
import subprocess as sub
import argparse
import logging

import numpy as np


LIBFM = 'libfm-1.42.src/bin/libFM'


class LibFMFailed(Exception):
    """LibFM returned a non-zero exit code."""
    pass


def compose_libfm_args(train, test, iter=20, std=0.2, dim=8, bias=False,
                       outfile='', bin=LIBFM):
    """Put together libFM args in a list suitable for use with Popen.

    :param str train:   Path for training data file, in libFM format.
    :param str test:    Path for test data file, in libFM format.
    :param int iter:    Number of iterations to run learning algorithm for.
    :param int dim:     Dimensionality of low-rank approximation to use.
    :param bool bias:   Use global and per-feature bias terms.
    :param str outfile: Output predictions to a file with this name.
    :return:            List of args to use for running libFM.

    """
    bias = int(bias)
    args = [
        bin,
        '-task', 'r',
        '-train', train,
        '-test', test,
        '-iter', str(iter),
        '-dim', '1,%d,%d' % (bias, dim),
        '-init_stdev', str(std)
    ]
    if outfile:
        args.append('-out')
        args.append(outfile)
    return args


def run_libfm(train, test, iter=20, std=0.2, dim=8, bias=False,
              outfile=''):
    """Run libFM and return final train/test results.

    :return: Final error for (train, test) sets.
    """
    kwargs = {k: v for k, v in locals().items() if not k in ['train', 'test']}
    args = compose_libfm_args(train, test, **kwargs)
    cmd = ' '.join(args)
    logging.debug(cmd)

    proc = sub.Popen(cmd, shell=True, stdout=sub.PIPE)
    retcode = proc.wait()
    if retcode:
        raise LibFMFailed("libFM failed to execute.\n%s" % cmd)

    output = proc.communicate()[0]
    lines = output.split('\n')
    rows = [row.split('\t')[1:] for row in lines[-iter:] if row]
    train_err = '%.6f' % float(rows[-1][0].split('=')[1])
    test_err = '%.6f' % float(rows[-1][1].split('=')[1])
    return [train_err, test_err]


def libfm_predict(train, test, outfile, iter=20, std=0.2, dim=8, bias=False):
    """Run libFM, output predictions to file, read file and return predictions
    as an array of floats.

    :rtype: numpy.ndarray of floats.
    :return: The predictions for the test examples.
    """
    kwargs = {k: v for k, v in locals().items()
              if not k in ['train', 'test']}
    kwargs['outfile'] = outfile
    args = compose_libfm_args(train, test, **kwargs)
    cmd = ' '.join(args)
    logging.debug(cmd)

    proc = sub.Popen(cmd, shell=True, stdout=sub.PIPE)
    retcode = proc.wait()
    if retcode:
        raise LibFMFailed("libFM failed to execute.\n%s" % cmd)

    # read output and return predictions
    with open(outfile) as f:
        return np.array([float(num) for num in f if num])


def test_dim(start, end, *args, **kwargs):
    """Run libFM regression once for each dimension value in range(start, end+1).
    See `run_libfm` for *args and **kwargs.

    :param int start: The first dimension in the range.
    :param int end: The last dimension in the range (inclusive).
    :return: List of (dim, train_err, test_err) for all dimensions tested.
    """
    results = []
    for dim in range(start, end+1):
        out = run_libfm(*args, dim=dim, **kwargs)
        out.insert(0, str(dim))
        logging.info('\t'.join(out))
        results.append(out)

    top = list(sorted(results, key=lambda l: l[2]))
    top5 = [t[0] for t in top[:5]]
    logging.info('best results with dim=%s' % ','.join(map(str, top5)))
    return results


def make_parser():
    parser = argparse.ArgumentParser(
        description='try out various libFM params to fine-tune')

    parser.add_argument(
        '-v', '--verbose', action='store_true', default=False,
        help='enable verbose logging output')
    parser.add_argument(
        '-b', '--bias', action='store_true',
        help='bias terms will be used if this flag is given')
    parser.add_argument(
        '-i', '--iter', type=int, default=200,
        help='number of iterations to run learning algorithm for')
    parser.add_argument(
        '-d', '--dimensionality', default='5-20',
        help='dimensionality of low-rank approximation; try all in range')
    # higher values can speed up learning, but too high can slow it
    parser.add_argument(
        '-s', '--init-stdev', type=float, default=0.2,
        help='initial standard deviation for initial Gaussian dist')
    parser.add_argument(
        '-tr', '--train',
        help='path for training file, in libFM format')
    parser.add_argument(
        '-te', '--test',
        help='path for test file, in libFM format')

    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s][%(levelname)s]: %(message)s')

    dim_start, dim_end = map(int, args.dimensionality.split('-'))
    logging.info('running libFM for %d iterations' % args.iter)
    logging.info('testing dimension range: %s' % args.dimensionality)
    results = test_dim(dim_start, dim_end, args.train, args.test, args.iter,
                       std=args.init_stdev, bias=args.bias)
    print '\n'.join(['\t'.join(result) for result in results])
