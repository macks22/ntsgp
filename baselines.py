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
    calculated using a basic SVD. If `knn_post` is True, use kNN
    post-processing. First find the most similar item in the V matrix. So cosine
    similarity is calculated between all items using the latent factor
    composition of each item. Then we reconstruct the original matrix using `k`
    factors. Finally, we fill the value for each cell with the value of the
    user's rating on the nearest neighbor of that item.

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
        df = df.apply(
            lambda s: map(lambda cid: s.iloc[closest[cid]], s.index.values),
            axis=1)

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

    # Read in necessary columns from data.
    logging.info("reading data from: %s" % args.data_file)
    cols = ['sid', 'cid', 'grdpts', 'termnum']
    cs = pd.read_csv(args.data_file, usecols=cols)
    cs = cs.sort(['sid', 'termnum'])

    # Now let's take some data for testing. For each student, we take the last
    # 2 courses, or take 1 course if only has 1, or take none if only has 1.
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
    results = {}  # hold method_name: rmse

    ur_base = uniform_random_baseline(train, to_predict)
    results['uniform random'] = rmse(ur_base, test)
    gm_base = global_mean_baseline(train, to_predict)
    results['global mean'] = rmse(gm_base, test)
    ngm_base = gm_normal_baseline(train, to_predict)
    results['normal'] = rmse(ngm_base, test)
    mom_base = mean_of_means_baseline(train, to_predict)
    results['mean of means'] = rmse(mom_base, test)

    print 'uniform random baseline rmse:   %.5f' % results['uniform random']
    print 'global mean baseline rmse:      %.5f' % results['global mean']
    print 'normal baseline rmse:           %.5f' % results['normal']
    print 'mean of means baseline rmse:    %.5f' % results['mean of means']

    # Evaluate using SVD for a variety of k values.
    logging.info('making predictions using SVD...')
    k_start = 1
    best = (k_start, np.inf)  # best is start with max rmse to start
    predicted = svd_range(train, to_predict, k_start=k_start, k_end=10)
    for k, svd_base in predicted:
        key = 'svd (k=%d)' % k
        err = rmse(svd_base, test)
        results[key] = err
        if err < best[1]:  # new best RMSE from SVD
            best = (k, err)
        print 'svd baseline rmse (k=%d):\t%.5f' % (k, results[key])

    # Now use kNN post-processing using best SVD results.
    svd_knn_base = svd_knn(train, to_predict, k=best[0])
    key = 'svd-knn (k=%d)' % best[0]
    results[key] = rmse(svd_knn_base, test)
    print 'svd-knn baseline rmse (k=%d):\t%.5f' % (best[0], results[key])

    # Find top results.
    pairs = results.items()
    pairs.sort(key=lambda tup: tup[1])
    print '\nTop 5 baselines:'
    for name, err in pairs[:5]:
        print '%s\t%.5f' % (name, err)
