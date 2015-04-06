import time
import multiprocessing as mp
import pandas as pd
import numpy as np


def matrix_approx(U, M, low=0, high=20):
    A = np.dot(U.T, M)
    A[A < low] = low
    A[A > high] = high
    return A


def batch_svd(train_data, learn_rate=0.00001, regc_u=0.015, regc_m=0.015,
              low=0, high=20, dim=50, momentum=0.9, threshold=0.001,
              debug=False):
    """Batch Singular Value Decomposition with Momentum. Terminates when an
    update fails to improve RMSE by `threshold`, which defaults to 0.001.
    """

    # compute indicator for training data
    I = train_data.copy()
    I[~I.isnull()] = 1
    I = I.fillna(0).values

    # set initial values of user and item matrices (U & M)
    global_mean = train_data.mean().mean()  # ignoring nan values
    train_data[train_data.isnull()] = global_mean
    init = np.sqrt((global_mean - low) / dim)

    # U is the user matrix, so f x n
    n = len(train_data)
    U = init + np.random.uniform(-1, 1, (dim, n))

    # M is the item matrix, so f x m
    m = len(train_data.columns)
    M = init + np.random.uniform(-1, 1, (dim, m))

    # init momentum matrices
    u_mom = np.zeros((dim, n))
    m_mom = np.zeros((dim, m))

    # run gradient descent
    A = np.dot(U.T, M)
    err = train_data - A
    mse = (err**2 * I).mean().mean()
    rmse = np.sqrt(mse)
    rmse_record = [rmse]
    improvement = 1
    iterations = 0

    # for _ in xrange(iterations):
    while iterations < 10 or improvement >= threshold:

        # update momentums
        u_mom = momentum * u_mom
        m_mom = momentum * m_mom

        # calculate partial derivatives
        u_deriv = I.T[:dim,:] * (low + np.dot(M, err.T)) - regc_u * U
        m_deriv = I[:dim,:m] * (low + np.dot(U, err)) - regc_m * M

        # perform updates and calculate prediction matrix A
        u_mom += learn_rate * u_deriv
        m_mom += learn_rate * m_deriv
        U += u_mom
        M += m_mom
        A = np.dot(U.T, M)

        # bound the values and compute error
        A[A > high] = high
        A[A < low] = low
        err = train_data - A

        mse = (err**2 * I).mean().mean()
        rmse =  np.sqrt(mse)
        improvement = rmse_record[-1] - rmse
        rmse_record.append(rmse)
        iterations += 1

    if debug:
        return U, M, rmse_record
    else:
        return U, M


def incremental_svd(train_data, learn_rate=0.0025, regc_u=0.015, regc_m=0.015,
                    low=0, high=20, dim=50, threshold=0.001, debug=False):
    """Complete Incremental Singular Value Decomposition.
    Terminates when an update fails to improve RMSE by `threshold`, which
    defaults to 0.001.

    Normally this method would be able to converge faster with a higher learning
    rate than used in batch SVD. Unfortunately, this is not as useful in Python,
    since the vectorized computations possible in the batch version are
    drastically faster than the non-vectorized computations in this one.
    """

    # compute indicator for training data
    I = train_data.copy()
    I[~I.isnull()] = 1
    I = I.fillna(0).values

    # perform mean value imputation
    global_mean = train_data.mean().mean()
    train_data[train_data.isnull()] = global_mean

    # set initial values of user and item matrices (U & M)
    init = np.sqrt((global_mean - low) / dim)

    # U is the user matrix, so f x n
    n = len(train_data)
    U = init + np.random.uniform(-1, 1, (dim, n))

    # M is the item matrix, so f x m
    m = len(train_data.columns)
    M = init + np.random.uniform(-1, 1, (dim, m))

    # run gradient descent
    A = np.dot(U.T, M)
    err = train_data - A
    mse = (err**2 * I).mean().mean()
    rmse = np.sqrt(mse)
    rmse_record = [rmse]
    improvement = 1

    while improvement >= threshold:

        # Perform updates value-wise
        for (i, j), val in np.ndenumerate(train_data):

            # ignore missing values
            if not I[i,j]:
                continue

            # calculate partial derivatives
            U_i = U[:,i]
            M_j = M[:,j]
            predict = low + np.dot(U_i, M_j)
            err = val - predict
            u_deriv = err * M_j - regc_u * U_i
            m_deriv = err * U_i - regc_m * M_j

            # perform updates
            U[:,i] += learn_rate * u_deriv
            M[:,j] += learn_rate * m_deriv

        # calculate prediction matrix, bound the values, and compute error
        A = np.dot(U.T, M)
        A[A > high] = high
        A[A < low] = low
        err = train_data - A

        mse = (err**2 * I).mean().mean()
        rmse =  np.sqrt(mse)
        improvement = rmse_record[-1] - rmse
        rmse_record.append(rmse)

    if debug:
        return U, M, rmse_record
    else:
        return U, M


def biased_incremental_svd(train_data, learn_rate=0.0025, regc_u=0.02,
                           regc_m=0.02, regc_b=0.05, low=0, high=20, dim=50,
                           threshold=0.005, debug=False):
    """Complete Incremental Singular Value Decomposition with bias terms.
    Terminates when an update fails to improve RMSE by `threshold`, which
    defaults to 0.001.
    """
    # TODO: incorporate global mean as global bias, in learning and prediction.

    # compute indicator for training data
    I = train_data.copy()
    I[~I.isnull()] = 1
    I = I.fillna(0).values

    # perform mean value imputation
    global_mean = train_data.mean().mean()
    train_data[train_data.isnull()] = global_mean

    # set initial values of user and item matrices (U & M)
    init = np.sqrt((global_mean - low) / dim)

    # U is the user matrix, so f x n
    n = len(train_data)
    U = init + np.random.uniform(-1, 1, (dim, n))

    # M is the item matrix, so f x m
    m = len(train_data.columns)
    M = init + np.random.uniform(-1, 1, (dim, m))

    # initialize the bias terms
    a = init + np.random.uniform(-1, 1, n)  # user bias
    b = init + np.random.uniform(-1, 1, m)   # item bias

    # run gradient descent
    A = np.dot(U.T, M)
    err = train_data - A
    mse = (err**2 * I).mean().mean()
    rmse = np.sqrt(mse)
    rmse_record = [rmse]
    improvement = 1

    while improvement >= threshold:

        # Perform updates value-wise
        for (i, j), val in np.ndenumerate(train_data):

            # ignore missing values
            if not I[i,j]:
                continue

            # calculate partial derivatives
            U_i = U[:,i]
            M_j = M[:,j]
            predict = low + np.dot(U_i, M_j) + a[i] + b[j]
            err = val - predict
            u_deriv = err * M_j - regc_u * U_i
            m_deriv = err * U_i - regc_m * M_j
            a_deriv = err - regc_b * a[i]
            b_deriv = err - regc_b * b[j]

            # perform updates
            U[:,i] += learn_rate * u_deriv
            M[:,j] += learn_rate * m_deriv
            a[i] += learn_rate * a_deriv
            b[j] += learn_rate * b_deriv

        # calculate prediction matrix, bound the values, and compute error
        A = np.dot(U.T, M)
        A[A > high] = high
        A[A < low] = low
        err = train_data - A

        mse = (err**2 * I).mean().mean()
        rmse =  np.sqrt(mse)
        improvement = rmse_record[-1] - rmse
        rmse_record.append(rmse)

    if debug:
        return U, M, a, b, rmse_record
    else:
        return U, M, a, b


def rmse(actual, predicted, low=0):
    indicator = actual.copy()
    indicator[indicator >= low] = 1
    indicator = actual.copy().fillna(0)
    se = (actual - predicted)**2 * indicator
    mse = se.sum().sum() / indicator.values.sum()
    return np.sqrt(mse)


def extract_test_set(data, num_test=3):
    """Extract a test set from the data and return it, modifying the original
    data in place to remove the test data extracted.

    :param DataFrame data: The data to split into test/train.
    :param int num_test: Number of test ratings to sample for each user.
    :return: test DataFrame, which has the same shape as `data`.
    """
    # make sure index and columns are both ints
    data.index = data.index.astype(np.int)
    data.columns = data.columns.astype(np.int)

    # randomly select 3 ratings for each user to use as test data
    # for now, just assume we have 3 valid options
    valid = data.notnull()
    test = pd.DataFrame(np.zeros(data.shape))
    test.columns = data.columns
    test.index = data.index
    for row in xrange(len(data)):
        data_row = data.ix[row]
        options = data_row[valid.ix[row]]
        choices = np.random.choice(options.index, replace=False, size=num_test)
        test.ix[row].ix[choices] = data_row.ix[choices]
        data_row.ix[choices] = np.nan

    return test


def load_jester_data(fname='data/jester-data-all.csv', nrows=1000):
    data = pd.read_csv(fname, index_col=0, header=False, nrows=nrows)
    data += 10  # ratings are from -10 to 10, lets make them 0 to 20
    test = extract_test_set(data)
    return data, test


def test_batch_svd(nrows=1000):
    train, test = load_jester_data(nrows=nrows)
    # these parameters were tuned through a number of test runs
    start = time.time()
    U, M, rmse_record = batch_svd(train, learn_rate=0.000012, dim=80,
                                  threshold=0.0001, debug=True)
    elapsed = time.time() - start

    A = matrix_approx(U, M, 0, 20)
    print "Batch SVD: %.5f (%.3fs)" % (
        rmse(test, A, low=0), elapsed)


def test_incremental_svd(nrows=1000):
    train, test = load_jester_data(nrows=nrows)
    # these parameters were tuned through a number of test runs
    start = time.time()
    U, M, rmse_record = incremental_svd(
        train, learn_rate=0.0025, dim=80, threshold=0.001, debug=True)
    elapsed = time.time() - start

    A = matrix_approx(U, M, 0, 20)
    print "Incremental SVD: %.5f (%.3fs)" % (
        rmse(test, A, low=0), elapsed)


def test_biased_incremental_svd(nrows=1000):
    train, test = load_jester_data(nrows=nrows)
    # these parameters were tuned through a number of test runs
    start = time.time()
    U, M, a, b, rmse_record = biased_incremental_svd(
        train, learn_rate=0.0025, dim=80, threshold=0.001, debug=True)
    elapsed = time.time() - start

    dims = (len(a), len(b))
    A = matrix_approx(U, M, low=0, high=20)
    A += (np.repeat(a,len(b)).reshape(dims) +
          np.repeat(b, len(a)).reshape(dims, order='F'))
    print "Biased Incremental SVD: %.5f (%.3fs)" % (
        rmse(test, A, low=0), elapsed)


if __name__ == "__main__":
    jobs = []
    for func in [test_batch_svd, test_incremental_svd,
                 test_biased_incremental_svd]:
        p = mp.Process(target=func)
        jobs.append(p)
        p.start()

    for job in jobs:
        job.join()
