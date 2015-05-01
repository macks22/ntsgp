import logging
import warnings
import argparse

import numpy as np
import pandas as pd
import scipy as sp
from scipy.sparse import linalg as splinalg


def rmse(predicted, actual, userid='sid', itemid='cid', value='grdpts'):
    """Compute root mean squared error between the predicted values and the
    actual values. This assumes some values are missing and only incorporates
    error measurements from the values present in the actual values.

    `actual` and `predicted` should be DataFrame objects with feature vectors
    for each data instance. The rmse is computed on the column named by `value`.
    """
    name_x = '%s_x' % value
    name_y = '%s_y' % value
    to_eval = actual.merge(predicted, how='left', on=[userid, itemid])
    to_eval = to_eval[[name_x, name_y]]  # filter down only to values

    # Sanity check; we don't want nan values here; that indicates no predictions
    # were made for some records.
    if to_eval.isnull().sum().sum() > 0:
        raise ValueError("predictions must be made for all missing values")

    # Compute error.
    error = (to_eval[name_x] - to_eval[name_y]).values
    mse = (error ** 2).sum() / len(error)
    return np.sqrt(mse)


def uniform_random_baseline(train, test, value='grdpts'):
    """Fill values in `value` column of `test` data frame using values drawn
    randomly from a uniform distribution.
    """
    logging.info(
        'predicting %d values using uniform random baseline' % len(test))
    to_predict = test.copy()
    pmin, pmax = train[value].min(), train[value].max()
    to_predict[value] = np.random.uniform(pmin, pmax, len(test))
    return to_predict


def global_mean_baseline(train, test, value='grdpts'):
    """Fill values in `value` column of `test` data frame using mean of all
    values in `train` set.
    """
    logging.info('predicting %d values using global mean baseline' % len(test))
    to_predict = test.copy()
    to_predict[value] = train[value].mean()
    return to_predict


def gm_normal_baseline(train, test, value='grdpts'):
    """Fill values in `value` column of `test` data frame using values drawn
    from a normal distribution with the mean set to the global mean and the
    standard deviation set to the standard deviation from the global mean.
    """
    logging.info('predicting %d values using normal baseline' % len(test))
    to_predict = test.copy()
    mu = train[value].mean()
    std = train[value].std() / np.sqrt(len(train))
    to_predict[value] = np.random.normal(mu, std, len(test))
    return to_predict


def mean_of_means_baseline(train, test, value='grdpts', userid='sid',
                           itemid='cid'):
    """Fill values in `value` column of `test` data frame using values
    calculated from the per-student, per-course, and global mean.
    """
    logging.info(
        'predicting %d values using mean of means baseline' % len(test))
    to_predict = test.copy()
    global_mean = train[value].mean()
    user_means = train.groupby(userid)[value].mean()
    item_means = train.groupby(itemid)[value].mean()
    to_predict[value] = to_predict.apply(
        lambda s: (
            global_mean +
            user_means[s[userid]] +
            item_means[s[itemid]]) / 3,
        axis=1)
    return to_predict


#TODO: FINISH IMPLEMENTING
def svd_baseline(train, test, userid='sid', itemid='cid', value='grdpts', k=5):
    """Fill values in `value` column of `test` data frame using values
    calculated using a basic SVD.
    """
    logging.info(
        'predicting %d values using SVD baseline' % len(test))
    to_predict = test.copy()
    alldata = pd.concat((train, test))[[userid, itemid, value]]
    matrix = alldata.pivot(index=userid, columns=itemid, values=value)
    mat = matrix.values
    U, s, V = splinalg.svds(mat, k)

    R = np.dot(U, V)
    R[R < 0] = 0
    R[R > 0] = 0

    df = pd.DataFrame(R)
    df.index = matrix.index
    df.columns = matrix.columns
    df = df.unstack().reset_index().rename(columns={0: 'grdpts'})

    mask = ((df[userid].isin(to_predict[userid])) &
            (df[itemid].isin(to_predict[itemed])))
    return df[mask]


def make_parser():
    parser = argparse.ArgumentParser(
        description='run baseline methods on dataset')
    parser.add_argument(
        '-d', '--data-file', action='store',
        help='data file to run methods on')
    parser.add_argument(
        '-nt', '--ntest',
        type=int, default=2,
        help='number of records for each student to test on; default 2')
    parser.add_argument(
        '-v', '--verbose',
        action='store_true', default=False,
        help='enable verbose logging output')
    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    logging.basicConfig(
        format='[%(asctime)s]: %(message)s',
        level=logging.INFO if args.verbose else logging.CRITICAL)
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Read in necessary columns from data.
    logging.info("reading data from: %s" % args.data_file)
    cols = ['sid', 'cid', 'grdpts', 'termnum']
    cs = pd.read_csv(args.data_file, usecols=cols)
    cs = cs.sort(['sid', 'termnum'])

    # Now let's take some data for testing. For each student, we take the last 2
    # courses, or take 1 course if only has 1, or take none if only has 1.
    logging.info("splitting trian/test data")
    test = cs.groupby('sid').tail(args.ntest)
    train = cs[~cs.index.isin(test.index)]

    # Transfer cold-start records back to train set.
    for key in ['sid', 'cid']:
        diff = np.setdiff1d(test[key].values, train[key].values)
        logging.info(
            "moving %d %s ids back to train set: %s" % (
                len(diff), key, ' '.join(map(str, diff))))
        cold_start = test[key].isin(diff)
        train = pd.concat((train, test[cold_start]))
        test = test[~cold_start]

    # Now fill grdpts with nan values to get prediction dataset.
    to_predict = test.copy()
    to_predict['grdpts'] = np.nan

    # Make predictions using baseline methods.
    logging.info('making predictions with baseline methods')

    ur_base = uniform_random_baseline(train, to_predict)
    gm_base = global_mean_baseline(train, to_predict)
    ngm_base = gm_normal_baseline(train, to_predict)
    mom_base = mean_of_means_baseline(train, to_predict)

    print 'uniform random baseline rmse: %.5f' % rmse(ur_base, test)
    print 'global mean baseline rmse:    %.5f' % rmse(gm_base, test)
    print 'normal baseline rmse:         %.5f' % rmse(ngm_base, test)
    print 'mean of means baseline rmse:  %.5f' % rmse(mom_base, test)

    results = svd_baseline(train, to_predict)
