import sys
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

def decision_tree_baseline(train, test, value='grdpts', max_depth=4):
    """Fill values in `value` column of `test` data frame using the
    DecisionTreeRegressor from scikit-learn.
    """
    to_predict = test.copy()
    logging.info(
        'predicting %d values using decision tree baseline' % len(test))

    from sklearn import tree
    clf = tree.DecisionTreeRegressor(max_depth=max_depth)

    # Split up predictors/targets.
    train_X, train_y = split_xy(train, value)
    test_X, test_y = split_xy(to_predict, value)

    # Learn model and make predictions.
    clf = clf.fit(train_X, train_y)
    predicted = clf.predict(test_X)
    to_predict[value] = predicted
    return to_predict


def linear_regression_baseline(train, test, value='grdpts'):
    """Fill values in `value` column of `test` data frame using the
    LinearRegression from scikit-learn.
    """
    to_predict = test.copy()
    logging.info(
        'predicting %d values using decision tree baseline' % len(test))

    from sklearn import linear_model
    clf = linear_model.LinearRegression()

    # Split up predictors/targets.
    train_X, train_y = split_xy(train, value)
    test_X, test_y = split_xy(to_predict, value)

    # Learn model and make predictions.
    clf = clf.fit(train_X, train_y)
    predicted = clf.predict(test_X)
    to_predict[value] = predicted
    return to_predict


def remove_cold_start(train, test, userid='sid', itemid='cid'):
    """Remove users/items from the test set that are not in the training set.
    """
    for key in [userid, itemid]:
        diff = np.setdiff1d(test[key].values, train[key].values)
        logging.info(
            "removing %d %s ids from the test set: %s" % (
                len(diff), key, ' '.join(map(str, diff))))
        cold_start = test[key].isin(diff)
        test = test[~cold_start]

    return test


def eval_method(data, method, *args, **kwargs):
    """Evaluate a particular baseline method `method` on the next-term
    prediction task with the given `data`. We assume the `termnum` column is
    present in the data and make predictions for each term by using all previous
    terms as training data. Additional argument will be passed to the `method`
    func.

    """
    results = {}  # key=termnum, val={'count': #, 'rmse': #}
    for termnum in sorted(data['termnum'].unique()):
        logging.info("making predictions for termnum %d" % termnum)
        train = data[data['termnum'] < termnum]
        test = data[data['termnum'] == termnum].copy()
        test = remove_cold_start(train, test)
        if len(test) == 0:
            results[termnum] = {'count': 0, 'rmse': 0}
            continue

        to_predict = test.copy()
        to_predict['grdpts'] = np.nan
        predictions = method(train, to_predict, *args, **kwargs)
        term_rmse = rmse(predictions, test)
        results[termnum] = {'count': len(test), 'rmse': term_rmse}

    sqerror = sum((result['rmse'] ** 2) * result['count']
                  for result in results.values()
                  if result['count'] > 0)
    final_count = sum(result['count'] for result in results.values())
    final_rmse = np.sqrt(sqerror / final_count)
    results['all'] = {'count': final_count, 'rmse': final_rmse}
    return results


def read_data(fname):
    """Read in necessary columns from data."""
    logging.info("reading data from: %s" % fname)
    cols = ['sid', 'cid', 'grdpts', 'termnum']
    data = pd.read_csv(fname, usecols=cols)
    return data.sort(['sid', 'termnum'])


def make_parser():
    parser = argparse.ArgumentParser(
        description='run baseline methods on dataset')
    parser.add_argument(
        'data_file', action='store',
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

    data = read_data(args.data_file)

    # Make predictions using baseline methods.
    logging.info('making predictions with baseline methods')
    results = {}  # hold method_name: final_rmse

    methods = {
        'uniform random': uniform_random_baseline,
        'global mean': global_mean_baseline,
        'normal': gm_normal_baseline,
        'mean of means': mean_of_means_baseline,
        'decision tree': decision_tree_baseline,
        'linear regression': linear_regression_baseline
    }

    compute_rmse = lambda method: eval_method(data, method)['all']['rmse']

    for method_name, func in methods.items():
        results[method_name] = compute_rmse(func)
        print '%s baseline rmse:\t%.5f' % (method_name, results[method_name])

    # Evaluate using SVD for a variety of k values.
    logging.info('making predictions using SVD...')
    k_start = 3
    best = (k_start, np.inf)  # best is start with max rmse to start
    for k in range(k_start, 8):
        key = 'svd (k=%d)' % k
        err = eval_method(data, svd_baseline, k=k)['all']['rmse']
        results[key] = err
        if err < best[1]:  # new best RMSE from SVD
            best = (k, err)
        print 'svd baseline rmse (k=%d):\t%.5f' % (k, results[key])

    # Now use kNN post-processing using best SVD results.
    best_k = best[0]
    key = 'svd-knn (k=%d)' % best_k
    results[key] = eval_method(data, svd_knn)['all']['rmse']
    print 'svd-knn baseline rmse (k=%d):\t%.5f' % (best_k, results[key])

    # Find top results.
    pairs = results.items()
    pairs.sort(key=lambda tup: tup[1])
    print '\nTop 5 baselines:'
    for name, err in pairs[:5]:
        print '%s\t%.5f' % (name, err)
