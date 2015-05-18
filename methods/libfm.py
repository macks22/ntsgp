import os
import subprocess as sub
import argparse
import logging
import datetime
import time

import numpy as np
from sklearn import preprocessing

from writer import write_libfm
from scaffold import *


LIBFM = '../libfm-1.42.src/bin/libFM'


class LibFMFailed(Exception):
    """LibFM returned a non-zero exit code."""
    pass


def compose_libfm_args(train, test, iter=20, std=0.2, dim=8, bias=False,
                       outfile='', task='r', bin=LIBFM):
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
        '-task', task,
        '-train', train,
        '-test', test,
        '-iter', str(iter),
        '-dim', '%d,%d,%d' % (bias, bias, dim),
        '-init_stdev', str(std)
    ]
    if outfile:
        args.append('-out')
        args.append(outfile)
    return args


def run_libfm(train, test, iter=20, std=0.2, dim=8, bias=False,
              outfile='', task='r'):
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


def libfm_predict(train, test, outfile, iter=20, std=0.2, dim=8, bias=False,
                  task='r'):
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


class FM(object):
    """Factorization machine (wraps libFM functionality using subprocess."""

    def __init__(self, train, test, outdir='tmp', iter=20, std=0.2, dim=8,
                 bias=False, task='r', userid='sid', itemid='cid',
                 target='grdpts', cvals=None, rvals=None, pcgrades=False):
        """
        :param DataFrame train: Training data.
        :param DataFrame test: Test data.
        :param str outfile: Path of file to write predictions to.
        :param int iter: Number of iterations to learn for.
        :param float std: Standard deviation of Gaussian noise for
            model initialization.
        :param int dim: Dimensionaltiy of model.
        :param bool bias: Use global and per-feature bias if True.
        :param str task: r=regression, c=classification.

        """
        tmpdir = '.tmp'
        try:
            os.mkdir(tmpdir)
        except OSError:
            pass

        self.outdir = os.path.join(tmpdir, outdir)
        try:
            os.mkdir(self.outdir)
        except OSError:
            pass

        self.outfile = os.path.join(self.outdir, 'predict.csv')
        self.ftrain = os.path.join(self.outdir, 'train.csv')
        self.ftest = os.path.join(self.outdir, 'test.csv')

        # Write data in libFM format to temporary files.
        with open(self.ftrain, 'w') as ftrain, open(self.ftest, 'w') as ftest:
            write_libfm(ftrain, ftest, train, test, target, userid,
                        itemid, cvals, rvals, pcgrades)

        self.iter = iter
        self.std = std
        self.dim = dim
        self.bias = bias
        self.task = task

    def run(self):
        return libfm_predict(self.ftrain, self.ftest, self.outfile, self.iter,
                             self.std, self.dim, self.bias, self.task)


def fm_mcmc(train, test, outdir='tmp', iter=20, std=0.2, dim=8, bias=False,
            task='r', userid='sid', itemid='cid', target='grdpts', cvals=None,
            rvals=None, pcgrades=False):
    fm = FM(train, test, outdir, iter, std, dim, bias, task, userid, itemid,
            target, cvals, rvals, pcgrades)
    return fm.run()


def libfm_model(train, test, *args, **kwargs):
    train = train.copy()
    to_predict = test.copy()
    logging.info('predicting %d values using libFM' % len(to_predict))
    to_predict['grdpts'] = fm_mcmc(train, test, *args, **kwargs)
    return to_predict


def scale(df, colname):
    df[colname] = (df[colname] - df[colname].mean()) / df[colname].std(ddof=0)


def libfm_model2(train, test, *args, **kwargs):
    """Scale data first."""
    train = train.copy()
    to_predict = test.copy()

    # Scale real-valued attributes.
    # Note: we must scale both train/test at the same time in order to avoid
    # distorting the importance of the attributes in one or the other.
    for colname in kwargs.get('rvals', []):
        vals = pd.concat((train[colname], test[colname]))
        mean = vals.mean()
        std = vals.std(ddof=0)
        train[colname] = (train[colname] - mean) / std
        test[colname] = (test[colname] - mean) / std

    logging.info('predicting %d values using libFM' % len(to_predict))
    to_predict['grdpts'] = fm_mcmc(train, test, *args, **kwargs)
    return to_predict


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


def gen_ts():
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
    return st


CVALS = ['itenure', 'irank', 'iclass', 'iid',
         'major', 'hs', 'sex', 'zip', 'srace', 'cohort',
         'cdisc']
RVALS = ['age', 'hsgpa', 'sat', 'lterm_gpa', 'lterm_cum_gpa',
         'chrs', 'clevel', 'total_chrs', 'num_enrolled', 'total_enrolled',
         'lterm_cgpa', 'lterm_cum_cgpa', 'term_chrs']

def make_parser():
    parser = argparse.ArgumentParser(
        description='try out various libFM params to fine-tune')
    parser.add_argument(
        'data_file', action='store')
    parser.add_argument(
        '-v', '--verbose', action='store_true', default=False,
        help='enable verbose logging output')
    parser.add_argument(
        '-b', '--bias', action='store_true', default=False,
        help='bias terms will be used if this flag is given')
    parser.add_argument(
        '-i', '--iter', type=int, default=200,
        help='number of iterations to run learning algorithm for')
    parser.add_argument(
        '-d', '--dimension', type=int, default=5,
        help='dimensionality of low-rank approximation')
    # higher values can speed up learning, but too high can slow it
    parser.add_argument(
        '-s', '--init-stdev', type=float, default=0.2,
        help='initial standard deviation for initial Gaussian dist')
    parser.add_argument(
        '-p', '--pcgrades', action='store_true', default=False)
    parser.add_argument(
        '-t', '--task',
        action='store', choices=('r', 'c'), default='r')
    parser.add_argument(
        '-o', '--outdir',
        action='store', default=None)

    # Add all possible features.
    for featname in CVALS + RVALS:
        name = '--%s' % featname
        parser.add_argument(
            name, action='store_true', default=False)

    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s][%(levelname)s]: %(message)s')

    outdir = args.outdir if args.outdir else gen_ts()

    data = read_data(args.data_file)

    cvals = [cval for cval in CVALS if getattr(args, cval)]
    rvals = [rval for rval in RVALS if getattr(args, rval)]

    def eval_fm(train, test):
        return libfm_model2(
            train, test, outdir=outdir, iter=args.iter, std=args.init_stdev,
            dim=args.dimension, bias=args.bias, task=args.task, cvals=cvals,
            rvals=rvals, pcgrades=args.pcgrades)

    print 'RMSE: %.5f' % eval_method(data, eval_fm, False)['all']['rmse']
