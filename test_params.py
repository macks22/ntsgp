import os
import subprocess as sub
import argparse
import logging


LIBFM = 'libfm-1.42.src/bin/libFM'


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


def run_libfm(train, test, iter, std=0.2, dim=8, bias=False, bin=LIBFM):
    """Run libFM and return final train/test results.

    :param str train: Path for training data file, in libFM format.
    :param str test:  Path for test data file, in libFM format.
    :param int iter:  Number of iterations to run learning algorithm for.
    :param int dim:   Dimensionality of low-rank approximation to use.
    :rtype:  (float, float)
    :return: Final error for (train, test) sets.
    """
    bias = int(bias)
    args = [
        bin,
        '-task', 'r',
        '-train', train,
        '-test', test,
        '-iter', str(iter),
        '-dim', '%d,%d,%d' % (bias, bias, dim),
        '-init_stdev', str(std)
    ]
    logging.debug(' '.join(args))
    proc = sub.Popen(args, stdout=sub.PIPE)
    output = proc.communicate()[0]
    lines = output.split('\n')
    rows = [row.split('\t')[1:] for row in lines[-iter:] if row]
    train_err = rows[-1][0].split('=')[1]
    test_err = rows[-1][1].split('=')[1]
    return [train_err, test_err]


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
