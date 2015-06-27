import os
import subprocess as sub
import argparse
import logging

import numpy as np
from sklearn import preprocessing

from writer import write_libfm
from scaffold import *
from wlibfm import FM


def libfm(train, test, userid='sid', itemid='cid', target='grdpts',
          cvals=None, rvals=None, previous=False, outfile='predict.csv',
          train_file='train.csv', test_file='test.csv',
          method='mcmc', task='r', iter=100, std=0.1, dim=8,
          fbias=True, gbias=True, lrate=0.1, r0=0.0, r1=0.0, r2=0.0,
          outdir='tmp', logfile=''):
    fm = FM(method, task, iter, std, dim, fbias, gbias, lrate, r0, r1, r2,
            outdir, logfile)
    fm.write(train, test, userid, itemid, target, cvals, rvals, previous,
             outfile, train_file, test_file)
    predictions = fm.run()
    fm.teardown()
    return predictions


def libfm_model(train, test, *args, **kwargs):
    """Scale data first."""
    target = kwargs.get('target', 'grdpts')
    task = kwargs.get('task', 'r')
    task_name = 'regression' if task == 'r' else 'classification'

    train = train.copy()
    to_predict = test.copy()

    # Scale real-valued attributes.
    # Note: we must scale both train/test at the same time in order to avoid
    # distorting the importance of the attributes in one or the other.
    rvals = kwargs.get('rvals', [])
    rvals = [] if rvals is None else rvals
    for colname in rvals:
        vals = pd.concat((train[colname], test[colname]))
        mean = vals.mean()
        std = vals.std(ddof=0)
        train[colname] = (train[colname] - mean) / std
        test[colname] = (test[colname] - mean) / std

    logging.info('predicting %d %s values using libFM %s' % (
        len(to_predict), target, task_name))
    to_predict[target] = libfm(train, test, *args, **kwargs)
    return to_predict


def make_parser():
    parser = base_parser('try out various libFM params to fine-tune')
    parser.add_argument(
        '-m', '--method',
        choices=('mcmc', 'als', 'sgd'), default='mcmc',
        help='inference method')
    parser.add_argument(
        '-t', '--task',
        action='store', choices=('r', 'c'), default='r')
    parser.add_argument(
        '-i', '--iter', type=int, default=100,
        help='number of iterations to run learning algorithm for')
    parser.add_argument(
        '-s', '--init-stdev', type=float, default=0.2,
        help='initial standard deviation for initial Gaussian dist;'
             ' higher values speed up learning, too high may slow it')
    parser.add_argument(
        '-d', '--dimension', type=int, default=5,
        help='dimensionality of low-rank approximation')
    parser.add_argument(
        '-fb', '--fbias',
        action='store_true', default=False)
    parser.add_argument(
        '-gb', '--gbias',
        action='store_true', default=False)
    parser.add_argument(
        '-l', '--lrate',
        type=float, default=0.1,
        help='learning rate for SGD')
    parser.add_argument(
        '-p', '--pcgrades',
        action='store_true', default=False,
        help='use previous course grades as features')
    parser.add_argument(
        '-r0', type=float, default=0.0)
    parser.add_argument(
        '-r1', type=float, default=0.0)
    parser.add_argument(
        '-r2', type=float, default=0.0)
    parser.add_argument(
        '-o', '--outdir',
        action='store', default='')

    parser.add_argument(
        '-tw', '--train_window',
        type=int, default=None,
        help='how many terms to include in train set, starting from test term'
             '; default is use all prior terms')

    # Add all possible features.
    for featname in CVALS + RVALS:
        name = '--%s' % featname
        parser.add_argument(
            name, action='store_true', default=False)

    return parser


def eval_params(method, std, iter, dim, fbias, gbias, task, lrate, r0, r1, r2,
                target, cvals=None, rvals=None, previous=False):
    """Get total RMSE for all terms using a certain parameter setting.

    This function is for use in grid searches.
    """
    def eval_fm(train, test):
        return libfm_model(
            train, test, method=method, std=std, iter=iter, dim=dim,
            fbias=fbias, gbias=gbias, task=task, lrate=lrate, r0=r0, r1=r1,
            r2=r2, target=target, cvals=cvals, rvals=rvals, previous=previous)

    results = method_error(data, eval_fm, False)
    return results['all']['rmse']


if __name__ == "__main__":
    args = setup(make_parser)
    outdir = args.outdir if args.outdir else gen_ts()

    data = pd.read_csv(args.data_file).sort(['sid', 'termnum'])
    cvals = [cval for cval in CVALS if getattr(args, cval)]
    rvals = [rval for rval in RVALS if getattr(args, rval)]

    target = 'grdpts' if args.task == 'r' else 'pass'
    if not target in data.columns:
        print "cannot perform %s task without target attribute: %s" % (
            'regression' if args.task == 'r' else 'classification', target)
        sys.exit()

    def eval_fm(train, test):
        return libfm_model(
            train, test, method=args.method, lrate=args.lrate, r0=args.r0,
            r1=args.r1, r2=args.r2, outdir=outdir, iter=args.iter,
            std=args.init_stdev, dim=args.dimension, fbias=args.fbias,
            gbias=args.gbias, task=args.task, target=target, cvals=cvals,
            rvals=rvals, previous=args.pcgrades)

    # Regression task.
    if args.task == 'r':
        quiet_delete(data, 'alevel')

        results = method_error(
            data, eval_fm, dropna=False, predict_cold_start=args.cold_start,
            train_window=args.train_window)
        evaluation = eval_results(
            results, by='sterm' if args.plot == 'sterm' else 'termnum')
        print evaluation

        if args.cold_start:
            print eval_results(results, by='cs')

        if args.plot == 'pred':
            g1, g2 = plot_predictions(results)
        elif args.plot in ['termnum', 'sterm', 'cohort']:
            ax1, ax2 = plot_error_by(args.plot, results)

    # Classification task.
    else:
        quiet_delete(data, 'grdpts')

        num_labels = len(data.alevel.unique())
        results = pd.DataFrame()
        tokeep = ['sid', 'cid', 'alevel', 'major', 'sterm']

        terms = list(sorted(data['termnum'].unique()))
        for termnum in terms:
            logging.info("making predictions for termnum %d" % termnum)

            train, test = split_train_test(data, termnum)
            test = remove_cold_start(train, test)
            if len(test) == 0:
                continue

            # How many labels are we actually working with?
            uniq_labels = np.unique(test[target])
            nlabels = len(uniq_labels)
            if nlabels == 1:
                print 'only one label for term %d' % termnum
                continue
            elif nlabels == 2:
                predicted = eval_fm(train, test)
            else:
                raise NotImplementedError(
                    "multi-class classification not implemented")
                sys.exit()

            df = test[tokeep].copy()
            df['predicted'] = predicted[target]
            df['test'] = test[target]
            results = pd.concat((results, df))

            print '\nTERM %d:' % termnum
            print_eval(df['test'], df['predicted'])

        print '\nTOTALS'
        print_eval(results['test'], results['predicted'])
