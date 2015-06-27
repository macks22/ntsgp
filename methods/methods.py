import os
import sys
import time
import logging
import warnings
import argparse
import datetime
import multiprocessing as mp

try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np
import pandas as pd
import scipy as sp
from scipy.sparse import linalg as splinalg
from sklearn import svm, linear_model, ensemble, tree, neighbors, preprocessing

from scaffold import *
from libfm import libfm_model, CVALS, RVALS


def gen_ts():
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
    return st

def eval_fm(train, test):
    outdir = gen_ts()
    return libfm_model(
        train, test, method='mcmc', std=0.2, iter=200, dim=8,
        fbias=True, gbias=True, task='r', target='grdpts',
        cvals=None, rvals=None)


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
    # user_means = train.groupby(userid).apply(
    #     lambda df: (df[value] - global_mean).mean())
    # item_means = train.groupby(itemid).apply(
    #     lambda df: (df[value] - global_mean).mean())
    to_predict[value] = to_predict.apply(
        lambda s: (
            global_mean +
            user_means.get(s[userid], 0) +
            item_means.get(s[itemid], 0)) / 3,
        axis=1)
    return to_predict


def mean_of_means_fillna(matrix):
    """Fill missing values in the np.ndarray using the mean of (1) the global
    mean, (2) row-wise means, and (3) column-wise means. We assume a 2D array.
    """
    nan_mask = np.isnan(matrix)
    masked = np.ma.masked_array(matrix, nan_mask)
    global_mean = masked.mean()
    row_means = masked.mean(axis=1)
    col_means = masked.mean(axis=0)

    n, m = matrix.shape
    row_means = np.tile(row_means.reshape(n, 1), m)
    col_means = np.tile(col_means.reshape(m, 1), n)
    means = global_mean + row_means + col_means.T
    matrix[nan_mask] = means


def _closest_neighbor(V):
    """Find k-nearest neighbors for each item in the matrix. We assume the
    items are represented by the columns of `matrix`. We return a vector with
    the closest neighbor for each movie.
    """
    sim = np.dot(V.T, V) / (sp.linalg.norm(V) * 2)
    return sim.argmax(1)


def svd_baseline(train, test, userid='sid', itemid='cid', value='grdpts', k=5,
                 knn_post=False):
    """Fill values in `value` column of `test` data frame using values
    calculated using a basic SVD (no regularization, no bias terms).

    If `knn_post` is True, use kNN post-processing. First find the most similar
    item in the V matrix. So cosine similarity is calculated between all items
    using the latent factor composition of each item. Then we reconstruct the
    original matrix using `k` factors. Finally, we fill the value for each cell
    with the value of the user's rating on the nearest neighbor of that item.

    """
    logging.info(
        'predicting %d values using SVD baseline' % len(test))
    to_predict = test.copy()
    alldata = pd.concat((train, to_predict))[[userid, itemid, value]]
    matrix = alldata.pivot(index=userid, columns=itemid, values=value)

    # Fill missing values with mean of means.
    # mean_of_means_fillna(matrix)  # produces poor results.
    matrix = matrix.fillna(matrix[~matrix.isnull()].mean().mean())

    # Perform SVD on the matrix.
    U, S, V = splinalg.svds(matrix.values, k)

    if knn_post:
        # Find most similar item for each item.
        closest_item = _closest_neighbor(V)
        closest = pd.Series(closest_item, matrix.columns)

    # Reconstruct original matrix.
    S = sp.linalg.diagsvd(S, U.shape[1], V.shape[0])
    R = np.dot(np.dot(U, S), V)
    R[R < 0] = 0
    R[R > 4] = 4

    # Convert back to Dataframe.
    df = pd.DataFrame(R)
    df.index = matrix.index
    df.columns = matrix.columns

    if knn_post:
        # Use knn postprocessing here.
        to_predict['grdpts'] = to_predict.apply(
            lambda s: df.ix[int(s['sid'])].iloc[closest[s['cid']]], axis=1)
        return to_predict
    else:
        # Now convert back to original data frame, with missing values filled.
        df = df.unstack().reset_index().rename(columns={0: value})
        del to_predict[value]
        return to_predict.merge(df, how='left', on=[userid, itemid])


def svd_knn(train, test, userid='sid', itemid='cid', value='grdpts', k=5):
    return svd_baseline(train, test, userid, itemid, value, k, knn_post=True)


def svd_range(train, test, k_start, k_end, userid='sid', itemid='cid',
              value='grdpts'):
    """Fill values in `value` column of `test` data frame using values
    calculated using a basic SVD. Reconstruct the matrix with a range of k
    values, for comparison.
    """
    logging.info('predicting %d values using SVD baseline' % len(test))
    k_vals = range(k_start, k_end + 1)
    logging.info('using k values: %s' % ' '.join(map(str, k_vals)))
    to_predict = test.copy()
    alldata = pd.concat((train, to_predict))[[userid, itemid, value]]
    matrix = alldata.pivot(index=userid, columns=itemid, values=value)

    # Fill missing values with mean of means.
    # mean_of_means_fillna(matrix)  # produces poor results.
    matrix = matrix.fillna(matrix[~matrix.isnull()].mean().mean())

    # Perform SVD on the matrix.
    U, s, V = splinalg.svds(matrix.values, k_end)

    # Reverse so singular values are in descending order.
    n = len(s)
    U[:,:n] = U[:, n-1::-1]   # reverse the n first columns of U
    s = s[::-1]               # reverse s
    V[:n, :] = V[n-1::-1, :]  # reverse the n first rows of vt

    del to_predict[value]
    for k in k_vals:
        logging.info('reconstructing matrix using k=%d' % k)
        # Reconstruct original matrix.
        S = sp.linalg.diagsvd(s[:k], k, k)
        R = np.dot(np.dot(U[:, :k], S), V[:k, :])
        R[R < 0] = 0
        R[R > 4] = 4

        # Convert back to Dataframe.
        df = pd.DataFrame(R)
        df.index = matrix.index
        df.columns = matrix.columns
        df = df.unstack().reset_index().rename(columns={0: value})

        # Use SVD-filled values to fill in missing grdpts.
        predicted = to_predict.merge(df, how='left', on=[userid, itemid])
        yield (k, predicted)


def split_xy(data, target='grdpts'):
    return data.drop(target, axis=1).values, data[target].values

SKLEARN_MODELS = {}
def sklearn_model(model_class, *args, **kwargs):
    def model(train, test, value='grdpts'):
        to_predict = test.copy()
        logging.info(
            'predicting %d values using %s' % (
                len(test), model_class.__name__))
        clf = model_class(*args, **kwargs)

        # Split up predictors/targets.
        train_X, train_y = split_xy(train, value)
        test_X, test_y = split_xy(test, value)
        scaler = preprocessing.StandardScaler().fit(train_X)
        train_X = scaler.transform(train_X)
        test_X = scaler.transform(test_X)

        # Learn model and make predictions.
        clf = clf.fit(train_X, train_y)
        predicted = clf.predict(test_X)
        to_predict[value] = predicted
        return to_predict

    SKLEARN_MODELS[model_class.__name__] = model
    return model


logging.info('setting up scikit-learn model functions')
decision_tree_baseline = sklearn_model(
    tree.DecisionTreeRegressor, max_depth=4)

linear_regression_baseline = sklearn_model(
    linear_model.LinearRegression)

sgd_regression_baseline = sklearn_model(
    linear_model.SGDRegressor, n_iter=15, eta0=0.001, penalty='l1')

knn_regression_baseline = sklearn_model(
    neighbors.KNeighborsRegressor, n_neighbors=20)

random_forest_baseline = sklearn_model(
    ensemble.RandomForestRegressor, n_estimators=100, max_depth=10,
    n_jobs=4)

boosted_decision_tree_baseline = sklearn_model(
    ensemble.AdaBoostRegressor,
    base_estimator=tree.DecisionTreeRegressor(max_depth=11),
    n_estimators=100)

svm_baseline = sklearn_model(
    svm.SVR, C=1.7, epsilon=0.6, gamma=0.02, cache_size=5000)


def make_parser():
    parser = argparse.ArgumentParser(
        description='run baseline methods on dataset')
    parser.add_argument(
        'data_file', action='store',
        help='data file to run methods on')
    parser.add_argument(
        '-v', '--verbose',
        action='store_true', default=False,
        help='enable verbose logging output')
    parser.add_argument(
        '-tw', '--train_window',
        type=int, default=None,
        help='how many terms to include in train set, starting from test term'
             '; default is use all prior terms')
    parser.add_argument(
        '-c', '--cold-start',
        action='store_true', default=False,
        help='include cold-start records in the test set; dropped by default')
    parser.add_argument(
        '-n', '--njobs',
        type=int, default=4)
    return parser


# def eval_method(method_name, result_dict, data, method, dropna=False,
#                 predict_cold_start=False, train_window=None):
#     results = method_error(
#         data, method, dropna, predict_cold_start=predict_cold_start,
#         train_window=train_window)
#     result_dict[method_name] = eval_results(results, by='termnum')
#     return method_name
# 
# def print_result(method_name):
#     print '%s:\t%.4f' % (method_name, results[method_name].ix['all']['rmse'])


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    logging.basicConfig(
        format='[%(asctime)s]: %(message)s',
        level=logging.INFO if args.verbose else logging.CRITICAL)
    warnings.simplefilter(action='ignore', category=FutureWarning)

    random_forest_baseline = sklearn_model(
        ensemble.RandomForestRegressor, n_estimators=100, max_depth=10,
        n_jobs=args.njobs)

    logging.info('reading data from %s' % args.data_file)
    tokeep = ['grdpts', 'sid', 'cid', 'termnum', 'major', 'sterm', 'cohort',
              'cs']
    tokeep += RVALS
    data = pd.read_csv(args.data_file, usecols=tokeep).sort(['sid', 'termnum'])
    logging.info('read %d records' % data.shape[0])

    logging.info('making predictions with simple baselines')
    basic_methods = {
        'uniform random': uniform_random_baseline,
        'global mean': global_mean_baseline,
        'normal': gm_normal_baseline,
        'mean of means': mean_of_means_baseline
    }

    if args.cold_start:
        logging.info('predicting cold start records')

    results = {}  # hold method_name: final_rmse
    def print_result(method_name):
        result = results[method_name]

        print '\t'.join([method_name] +
                        ['%.4f' % num for num in result.ix['all']])
        # print '%s:\t%.4f' % (method_name, result.ix['all']['rmse'])

    def evaluate_method(method, dropna=False):
        results = method_error(
            data, method, dropna, predict_cold_start=args.cold_start,
            train_window=args.train_window)
        by = 'cs' if args.cold_start else 'termnum'
        return eval_results(results, by=by)

    for method_name, func in basic_methods.items():
        results[method_name] = evaluate_method(func, False)
        print_result(method_name)

    for method_name, func in SKLEARN_MODELS.items():
        results[method_name] = evaluate_method(func, True)
        print_result(method_name)


    # pool = mp.Pool(processes=args.njobs)
    # manager = mp.Manager()
    # results = manager.dict()

    # for method_name, func in basic_methods.items():
    #     arglist = [method_name, results, data, func, False, args.cold_start,
    #                args.train_window]
    #     thing = pool.apply_async(eval_method, arglist, callback=print_result)
    #     thing.get()

    # # These functions can't be pickled because they were created at runtime.
    # for method_name, func in SKLEARN_MODELS.items():
    #     arglist = [method_name, results, data, func, True, args.cold_start,
    #                args.train_window]
    #     thing = pool.apply_async(eval_method, arglist, callback=print_result)
    #     thing.get()

    # sys.exit()

    # Evaluate using SVD for a variety of k values.
    logging.info('making predictions using SVD...')
    k_start = 4
    best = (k_start, np.inf)  # best is start with max rmse to start
    for k in range(k_start, 7):
        key = 'svd (k=%d)' % k
        def _svd_baseline(train, test):
            return svd_baseline(train, test, k=k)

        result = evaluate_method(svd_baseline)
        results[key] = result
        err = result['rmse']['all']
        if err < best[1]:  # new best RMSE from SVD
            best = (k, err)
        print_result(key)

    # Now use kNN post-processing using best SVD results.
    best_k = best[0]
    key = 'svd-knn (k=%d)' % best_k
    results[key] = evaluate_method(svd_knn)
    print_result(key)

    # Finally, train/evaluate libFM.
    # _libfm = 'LibFM'
    # results[_libfm] = evaluate_method(eval_fm)
    # print_result(_libfm)

    parts = []
    if args.cold_start:
        parts.append('c')
    parts.append(os.path.splitext(os.path.basename(args.data_file))[0])
    fname = '%s.pickle' % '-'.join(parts)
    with open(fname, 'w') as f:
        pickle.dump(results, f)

    # Find top results.
    all_results = pd.DataFrame({
        name: df.ix['all'] for (name, df) in results.items()})\
            .transpose()\
            .sort('rmse')
    print '\nAll Methods Sorted From Best to Worst:'
    print all_results
