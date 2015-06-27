"""
Wrap up LibFM using subprocess calls. This requires that the train/test files be
written to disk before use and that the results be written to disk after
predictions. Naturally, this can be quite slow in some cases. It is an
expeditious solution with an acceptable offline runtime at this time.

NOTE: the current implementation does not support binary data format, meta
groupings or relations. It also does not support SGDA (since that requires
extending the writer to write a validation set).

"""
import os
import time
import shutil
import logging
import datetime
import subprocess as sub

import numpy as np

from writer import write_libfm


def silent_mkdir(dirname):
    """Create a directory if it does not exist, else do nothing."""
    try:
        os.mkdir(dirname)
    except OSError:
        pass


class LibFMFailed(Exception):
    """LibFM returned a non-zero exit code."""
    pass


def gen_ts():
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
    return st


LIBFM = '/home/msweene2/ers-data/libfm-1.42.src/bin/libFM'
class FM(object):
    """Factorization machine (wraps libFM functionality using subprocess)."""

    def __init__(self, method='mcmc', task='r', iter=100, std=0.1, dim=8,
                 fbias=True, gbias=True, lrate=0.1, r0=0.0, r1=0.0, r2=0.0,
                 outdir='', logfile='', verbosity=0, bin=LIBFM):
        """
        Args:
            method (str): Learning method for inference, mcmc by default.
            task (str): r=regression c=classification, set to r by default.
            iter (int): Number of iterations to learn for, 100 by default.
            std (float): Standard deviation of Gaussian noise used for model
                initialization, 0.1 by default.
            dim (int): Dimensionality (rank) of the FM. This is the number of
                latent factors learned, 8 by default.
            fbias (bool): If True, use per-interaction bias terms, True by
                default.
            gbias (bool): If True, use a global bias term (central tendency),
                True by default.
            lrate (float): Learning rate for SGD and SGDA, 0.1 by default.
            r0 (float): Bias regularization term, 0.0 by default.
            r1 (float): 1-way regularization term, 0.0 by default.
            r2 (float): 2-way regularization term 0.0 by default.
            outdir (str): Name of directory to write LibFM-formatted input to,
                as well as the prediction outputs.
            logfile (str): Name of the logfile for LibFM logging, which can be
                made more verbose by increasing the level with the `verbosity`
                parameter. By default, no logfile is written.
            verbosity (int): Verbosity level. Higher integers are more verbose.
                Set to 0 by default.

        """
        for (k, v) in locals().iteritems():
            if k != 'self':
                setattr(self, k, v)

        self.setup(outdir)

    def setup(self, outdir):
        """Ensure output directory is present in cwd.

        This wrapper requires writing input data frames to disk before calling
        LibFM and retrieving the output of LibFM from disk in order to return it
        back in memory. The outdir holds these temporary files, and it must
        exist before using the wrapper.

        """
        tmpdir = '.tmp'
        silent_mkdir(tmpdir)
        if not outdir:
            outdir = gen_ts()
        self.outdir = os.path.join(tmpdir, outdir)
        silent_mkdir(self.outdir)

    def teardown(self):
        shutil.rmtree(self.outdir)

    def write(self, train, test, userid='sid', itemid='cid', target='grdpts',
             cvals=None, rvals=None, previous=False, outfile='predict.csv',
             train_file='train.csv', test_file='test.csv'):
        """
        Args:
            train (DataFrame): Training dataset; take care that it has columns
                for `userid`, `itemid`, and `target`, as well as all `cvals`
                and `rvals`.
            test (DataFrame): Testing dataset; same precuations as `train`.
            validation (DataFrame): Validation dataset (only for SGDA).
            userid (str): Name of the user ID column.
            itemid (str): Name of the item ID column.
            target (str): Name of the column with the target (dependent) var.
            cvals (list): Names of the categorical variables to use as features.
                This is in addition to `userid` and `itemid` and should not
                include them (doing so causes undefined behavior).
            rvals (list): Names of the real-valued variables to use as features.
                Do not include `target` here.
            previous (bool): If True, previous item ratings will be used as
                features.
            outfile (str): Filename for predictions output file.
            train_file (str): Filename for temporary LibFM-formatted file for
                training data.
            test_file (str): Filename for temporary LibFM-formatted file for
                testing data.

        """
        self.outfile = os.path.join(self.outdir, outfile)
        self.train_file = os.path.join(self.outdir, train_file)
        self.test_file = os.path.join(self.outdir, test_file)

        # Write data in libFM format to temporary files.
        with open(self.train_file, 'w') as ftrain,\
             open(self.test_file, 'w') as ftest:
            write_libfm(ftrain, ftest, train, test, target, userid,
                        itemid, cvals, rvals, previous)

    @property
    def cmd_args(self):
        args = [
            self.bin,
            '-method', self.method,
            '-task', self.task,
            '-train', self.train_file,
            '-test', self.test_file,
            '-iter', str(self.iter),
            '-dim', '%d,%d,%d' % (self.gbias, self.fbias, self.dim),
            '-init_stdev', str(self.std),
            '-out', self.outfile,
            '-verbosity', '%d' % self.verbosity
        ]
        if self.logfile:
            args += ['-rlog', self.logfile]
        if self.method != 'mcmc':
            args += ['-regular', '%f,%f,%f' % (self.r0, self.r1, self.r2)]
        if self.method == 'sgd':
            args += ['-learn_rate', str(self.lrate)]
        return args

    @property
    def cmd(self):
        return ' '.join(self.cmd_args)

    def run(self):
        """Note that probabilities are returned for classification."""
        logging.debug(self.cmd)

        logging.info('spawning libFM subprocess')
        proc = sub.Popen(self.cmd, shell=True, stdout=sub.PIPE)
        retcode = proc.wait()
        logging.info('libFM finished running')
        if retcode:
            raise LibFMFailed("libFM failed to execute.\n%s" % self.cmd)

        # read output and return predictions
        with open(self.outfile) as f:
            logging.info('reading predictions output by libFM')
            predictions = np.array([float(num) for num in f if num])
            logging.info('done reading predictions')
            return predictions


if __name__ == "__main__":

    # Read the data.
    import pandas as pd
    data = pd.read_csv('data/preprocessed-cs-students.csv')
    train = data[data.termnum < 5]
    test = data[data.termnum == 5]

    # Predict new grades.
    fm = FM(method='als')
    fm.write(train, test)
    predictions = fm.run()

    # Evaluate predictions.
    err = (test.grdpts.values - predictions)
    abserr = abs(err)
    mae = abserr.sum() / len(err)
    mae_std = abserr.var()
    sqerr = err ** 2
    mse = sqerr.sum() / len(sqerr)
    rmse = np.sqrt(mse)

    print 'MAE:  %.4f' % mae
    print 'STD:  %.4f' % mae_std
    print 'RMSE: %.4f' % rmse
